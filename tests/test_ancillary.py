import io
import os
import platform
import subprocess as sp
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

import spatialist.ancillary as anc

HAS_PATHOS = hasattr(anc, "mp")
CPU_COUNT = anc.mp.cpu_count() if HAS_PATHOS else 1


@pytest.fixture
def file_tree(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    
    (root / "root.txt").write_text("root", encoding="utf-8")
    
    sub1 = root / "sub1"
    sub1.mkdir()
    (sub1 / "file1.txt").write_text("one", encoding="utf-8")
    
    nested = sub1 / "nested"
    nested.mkdir()
    (nested / "file3.txt").write_text("three", encoding="utf-8")
    
    sub2 = root / "sub2"
    sub2.mkdir()
    (sub2 / "file2.log").write_text("two", encoding="utf-8")
    
    return root


@pytest.fixture
def zip_archive(tmp_path):
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("folder/file1.txt", "one")
        archive.writestr("folder/nested/file2.dat", "two")
        archive.writestr("root.txt", "root")
    return path


@pytest.fixture
def tar_archive(tmp_path):
    path = tmp_path / "archive.tar"
    
    with tarfile.open(path, "w") as archive:
        for dirname in ["folder", "folder/nested"]:
            info = tarfile.TarInfo(dirname)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        
        for name, payload in [
            ("folder/file1.txt", b"one"),
            ("folder/nested/file2.dat", b"two"),
            ("root.txt", b"root"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    
    return path


# ---------------------------------------------------------------------------
# HiddenPrints
# ---------------------------------------------------------------------------

def test_hiddenprints_returns_self():
    context = anc.HiddenPrints()
    with context as entered:
        assert entered is context


def test_hiddenprints_suppresses_stdout(capsys):
    print("before")
    
    with anc.HiddenPrints():
        print("hidden")
    
    print("after")
    
    captured = capsys.readouterr().out
    assert "before" in captured
    assert "after" in captured
    assert "hidden" not in captured


def test_hiddenprints_restores_stdout_after_exception(capsys):
    with pytest.raises(RuntimeError, match="boom"):
        with anc.HiddenPrints():
            print("hidden")
            raise RuntimeError("boom")
    
    print("visible")
    
    captured = capsys.readouterr().out
    assert "hidden" not in captured
    assert "visible" in captured


def test_hiddenprints_supports_nesting(capsys):
    with anc.HiddenPrints():
        print("outer")
        with anc.HiddenPrints():
            print("inner")
        print("outer-again")
    
    print("visible")
    
    assert capsys.readouterr().out == "visible\n"


# ---------------------------------------------------------------------------
# dictmerge
# ---------------------------------------------------------------------------

def test_dictmerge_combines_disjoint_dictionaries():
    assert anc.dictmerge(
        {"a": 1, "b": 2},
        {"c": 3, "d": 4},
    ) == {
               "a": 1,
               "b": 2,
               "c": 3,
               "d": 4,
           }


def test_dictmerge_second_dictionary_takes_precedence():
    assert anc.dictmerge(
        {"a": 1, "shared": "first"},
        {"b": 2, "shared": "second"},
    ) == {
               "a": 1,
               "b": 2,
               "shared": "second",
           }


def test_dictmerge_does_not_modify_inputs():
    first = {"a": 1}
    second = {"b": 2}
    
    result = anc.dictmerge(first, second)
    result["a"] = 99
    
    assert first == {"a": 1}
    assert second == {"b": 2}


def test_dictmerge_handles_empty_dictionaries():
    assert anc.dictmerge({}, {}) == {}
    assert anc.dictmerge({"a": 1}, {}) == {"a": 1}
    assert anc.dictmerge({}, {"b": 2}) == {"b": 2}


# ---------------------------------------------------------------------------
# dissolve
# ---------------------------------------------------------------------------

def test_dissolve_flattens_nested_lists():
    assert anc.dissolve([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert anc.dissolve([[[1]]]) == [1]


def test_dissolve_flattens_nested_tuples():
    assert anc.dissolve(
        ((1, 2, (3, 4)), (5, (6, 7))),
    ) == [1, 2, 3, 4, 5, 6, 7]


def test_dissolve_flattens_mixed_lists_and_tuples():
    assert anc.dissolve(
        [(1, [2, (3,)]), [4, (5, 6)]],
    ) == [1, 2, 3, 4, 5, 6]


def test_dissolve_preserves_order_and_duplicates():
    assert anc.dissolve(
        ((1, 2), (1, 2)),
    ) == [1, 2, 1, 2]


def test_dissolve_treats_strings_as_atomic():
    assert anc.dissolve(
        [["abc"], ("def",)],
    ) == ["abc", "def"]


def test_dissolve_treats_non_list_tuple_iterables_as_atomic():
    value = {"a", "b"}
    
    result = anc.dissolve([value])
    
    assert result == [value]


def test_dissolve_empty_input():
    assert anc.dissolve([]) == []
    assert anc.dissolve(()) == []


# ---------------------------------------------------------------------------
# archive path helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path, expected",
    [
        ("file.txt", []),
        ("a/file.txt", ["a/"]),
        ("a/b/file.txt", ["a/", "a/b/"]),
        ("a/b/c/file.txt", ["a/", "a/b/", "a/b/c/"]),
    ],
)
def test_parent_dirs(path, expected):
    assert list(anc.parent_dirs(path)) == expected


def test_namelist_with_implicit_dirs_adds_missing_directories(zip_archive):
    with zipfile.ZipFile(zip_archive) as archive:
        names = set(anc.namelist_with_implicit_dirs(archive))
    
    assert names == {
        "folder/",
        "folder/file1.txt",
        "folder/nested/",
        "folder/nested/file2.dat",
        "root.txt",
    }


def test_namelist_with_implicit_dirs_deduplicates_explicit_directories(tmp_path):
    archive_path = tmp_path / "explicit.zip"
    
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("folder/", "")
        archive.writestr("folder/file.txt", "data")
    
    with zipfile.ZipFile(archive_path) as archive:
        names = anc.namelist_with_implicit_dirs(archive)
    
    assert set(names) == {
        "folder/",
        "folder/file.txt",
    }
    assert len(names) == 2


# ---------------------------------------------------------------------------
# finder: directories
# ---------------------------------------------------------------------------

def test_finder_directory_recursive_files(file_tree):
    result = anc.finder(
        str(file_tree),
        ["*.txt"],
    )
    
    assert result == sorted([
        str(file_tree / "root.txt"),
        str(file_tree / "sub1" / "file1.txt"),
        str(file_tree / "sub1" / "nested" / "file3.txt"),
    ])


def test_finder_directory_nonrecursive_files(file_tree):
    result = anc.finder(
        str(file_tree),
        ["*.txt"],
        recursive=False,
    )
    
    assert result == [str(file_tree / "root.txt")]


def test_finder_directory_files_and_folders(file_tree):
    result = anc.finder(
        str(file_tree),
        ["*"],
        foldermode=1,
    )
    
    assert len(result) == 7
    assert str(file_tree / "sub1") in result
    assert str(file_tree / "sub2") in result
    assert str(file_tree / "sub1" / "nested") in result
    assert str(file_tree / "root.txt") in result


def test_finder_directory_folders_only(file_tree):
    result = anc.finder(
        str(file_tree),
        ["*"],
        foldermode=2,
    )
    
    assert result == sorted([
        str(file_tree / "sub1"),
        str(file_tree / "sub1" / "nested"),
        str(file_tree / "sub2"),
    ])


def test_finder_directory_regex(file_tree):
    result = anc.finder(
        str(file_tree),
        [r"^file[13]\.txt$"],
        regex=True,
    )
    
    assert result == sorted([
        str(file_tree / "sub1" / "file1.txt"),
        str(file_tree / "sub1" / "nested" / "file3.txt"),
    ])


def test_finder_directory_multiple_patterns(file_tree):
    result = anc.finder(
        str(file_tree),
        ["root.txt", "*.log"],
    )
    
    assert result == sorted([
        str(file_tree / "root.txt"),
        str(file_tree / "sub2" / "file2.log"),
    ])


def test_finder_list_of_targets(file_tree):
    result = anc.finder(
        [
            str(file_tree / "sub1"),
            str(file_tree / "sub2"),
        ],
        ["*.txt"],
    )
    
    assert result == [
        str(file_tree / "sub1" / "file1.txt"),
        str(file_tree / "sub1" / "nested" / "file3.txt"),
    ]


@pytest.mark.parametrize("foldermode", [-1, 3, 99])
def test_finder_rejects_invalid_foldermode(file_tree, foldermode):
    with pytest.raises(
            ValueError,
            match="foldermode",
    ):
        anc.finder(
            str(file_tree),
            ["*"],
            foldermode=foldermode,
        )


def test_finder_rejects_missing_string_target(tmp_path):
    missing = tmp_path / "missing"
    
    with pytest.raises(
            RuntimeError,
            match="directory or a file",
    ):
        anc.finder(
            str(missing),
            ["*"],
        )


def test_finder_rejects_non_archive_file(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("data", encoding="utf-8")
    
    with pytest.raises(
            RuntimeError,
            match="zip or tar archive",
    ):
        anc.finder(
            str(path),
            ["*"],
        )


@pytest.mark.parametrize("target", [1, 1.5, (), {}])
def test_finder_rejects_invalid_target_type(target):
    with pytest.raises(
            TypeError,
            match="str or list",
    ):
        anc.finder(
            target,
            ["*"],
        )


# ---------------------------------------------------------------------------
# finder: ZIP archives
# ---------------------------------------------------------------------------

def test_finder_zip_files(zip_archive):
    result = anc.finder(
        str(zip_archive),
        ["*"],
        foldermode=0,
    )
    
    assert len(result) == 3
    assert all(not path.endswith("/") for path in result)
    assert any(path.endswith("folder/file1.txt") for path in result)
    assert any(path.endswith("folder/nested/file2.dat") for path in result)
    assert any(path.endswith("root.txt") for path in result)


def test_finder_zip_files_and_implicit_folders(zip_archive):
    result = anc.finder(
        str(zip_archive),
        ["*"],
        foldermode=1,
    )
    
    assert len(result) == 5
    assert any(path.endswith("folder") for path in result)
    assert any(path.endswith("folder/nested") for path in result)


def test_finder_zip_folders_only(zip_archive):
    result = anc.finder(
        str(zip_archive),
        ["*"],
        foldermode=2,
    )
    
    assert len(result) == 2
    assert any(path.endswith("folder") for path in result)
    assert any(path.endswith("folder/nested") for path in result)


def test_finder_zip_pattern_matches_basename(zip_archive):
    result = anc.finder(
        str(zip_archive),
        ["file*"],
    )
    
    assert len(result) == 2
    assert all("file" in os.path.basename(path) for path in result)


def test_finder_zip_regex(zip_archive):
    result = anc.finder(
        str(zip_archive),
        [r"^file\d\.(txt|dat)$"],
        regex=True,
    )
    
    assert len(result) == 2


def test_finder_zip_folder_paths_remain_absolute(zip_archive):
    result = anc.finder(
        str(zip_archive.resolve()),
        ["*"],
        foldermode=2,
    )
    
    assert all(
        os.path.isabs(path)
        for path in result
    )


# ---------------------------------------------------------------------------
# finder: TAR archives
# ---------------------------------------------------------------------------

def test_finder_tar_files(tar_archive):
    result = anc.finder(
        str(tar_archive),
        ["*"],
        foldermode=0,
    )
    
    assert len(result) == 3
    normalized = [path.replace("\\", "/") for path in result]
    assert any(path.endswith("folder/file1.txt") for path in normalized)
    assert any(path.endswith("folder/nested/file2.dat") for path in normalized)
    assert any(path.endswith("root.txt") for path in normalized)


def test_finder_tar_files_and_folders(tar_archive):
    result = anc.finder(
        str(tar_archive),
        ["*"],
        foldermode=1,
    )
    
    assert len(result) == 5


def test_finder_tar_folders_only(tar_archive):
    result = anc.finder(
        str(tar_archive),
        ["*"],
        foldermode=2,
    )
    
    assert len(result) == 2
    assert all(os.path.isabs(path) for path in result)


def test_finder_tar_pattern_matches_basename(tar_archive):
    result = anc.finder(
        str(tar_archive),
        ["file*"],
    )
    
    assert len(result) == 2


def test_finder_tar_regex(tar_archive):
    result = anc.finder(
        str(tar_archive),
        [r"^file\d\.(txt|dat)$"],
        regex=True,
    )
    
    assert len(result) == 2


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

def test_add():
    assert anc.add(1, 2, 3) == 6


# ---------------------------------------------------------------------------
# ExceptionWrapper
# ---------------------------------------------------------------------------

def test_exceptionwrapper_reraises_original_exception():
    try:
        raise ValueError("boom")
    except ValueError as error:
        wrapper = anc.ExceptionWrapper(error)
    
    with pytest.raises(
            ValueError,
            match="boom",
    ):
        wrapper.re_raise()


def test_exceptionwrapper_stores_exception():
    try:
        raise RuntimeError("problem")
    except RuntimeError as error:
        wrapper = anc.ExceptionWrapper(error)
    
    assert isinstance(wrapper.ee, RuntimeError)
    assert str(wrapper.ee) == "problem"
    assert wrapper.tb is not None


# ---------------------------------------------------------------------------
# multicore
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not HAS_PATHOS,
    reason="pathos is required for multicore processing",
)
@pytest.mark.parametrize(
    "multiargs, singleargs, expected",
    [
        (
                {"x": [1, 2]},
                {"y": 5, "z": 9},
                [15, 16],
        ),
        (
                {"x": [1, 2], "y": [5, 6]},
                {"z": 9},
                [15, 17],
        ),
    ],
)
def test_multicore_returns_results(
        multiargs,
        singleargs,
        expected,
):
    result = anc.multicore(
        anc.add,
        cores=2,
        multiargs=multiargs,
        **singleargs,
    )
    
    assert result == expected


@pytest.mark.skipif(
    not HAS_PATHOS,
    reason="pathos is required for multicore processing",
)
def test_multicore_reduces_cores_to_number_of_jobs():
    result = anc.multicore(
        anc.add,
        cores=32,
        multiargs={"x": [1, 2]},
        y=5,
        z=9,
    )
    
    assert result == [15, 16]


def test_multicore_rejects_unknown_multi_argument():
    with pytest.raises(
            AttributeError,
            match="incompatible multi arguments",
    ):
        anc.multicore(
            anc.add,
            cores=2,
            multiargs={"foobar": [1, 2]},
            y=5,
            z=9,
        )


def test_multicore_rejects_unknown_single_argument():
    with pytest.raises(
            AttributeError,
            match="incompatible single arguments",
    ):
        anc.multicore(
            anc.add,
            cores=2,
            multiargs={"x": [1, 2]},
            y=5,
            foobar=9,
        )


def test_multicore_rejects_different_multiarg_lengths():
    with pytest.raises(
            AttributeError,
            match="different length",
    ):
        anc.multicore(
            anc.add,
            cores=2,
            multiargs={
                "x": [1, 2],
                "y": [5, 6, 7],
            },
            z=9,
        )


def test_multicore_rejects_empty_multiarg_values():
    with pytest.raises(
            RuntimeError,
            match="did not get any multiargs",
    ):
        anc.multicore(
            anc.add,
            cores=2,
            multiargs={"x": []},
            y=5,
            z=9,
        )


def test_multicore_rejects_empty_multiargs_dictionary():
    with pytest.raises(
            RuntimeError,
            match="did not get any multiargs",
    ):
        anc.multicore(
            anc.add,
            cores=2,
            multiargs={},
            y=5,
            z=9,
        )


@pytest.mark.skipif(
    not HAS_PATHOS,
    reason="pathos is required for multicore processing",
)
def test_multicore_returns_none_when_all_jobs_return_none():
    commands = [
        [sys.executable, "-c", "pass"],
        [sys.executable, "-c", "pass"],
    ]
    
    result = anc.multicore(
        anc.run,
        cores=2,
        multiargs={"cmd": commands},
        void=True,
    )
    
    assert result is None


@pytest.mark.skipif(
    not HAS_PATHOS or platform.system() == "Windows",
    reason="this exception-context branch belongs to the non-Windows ProcessPool implementation",
)
def test_multicore_worker_exception_contains_call_context():
    with pytest.raises(TypeError) as exc:
        anc.multicore(
            anc.add,
            cores=1,
            multiargs={"x": ["not-a-number"]},
            y=5,
            z=9,
        )
    
    message = str(exc.value)
    assert "called function 'add'" in message
    assert "not-a-number" in message


@pytest.mark.skipif(
    not HAS_PATHOS or platform.system() == "Windows",
    reason="progress-bar handling is implemented in the non-Windows ProcessPool branch",
)
def test_multicore_progressbar_finishes(monkeypatch):
    class FakeProgress:
        def __init__(self):
            self.started = False
            self.finished = False
            self.updates = []
        
        def start(self):
            self.started = True
            return self
        
        def update(self, value):
            self.updates.append(value)
        
        def finish(self):
            self.finished = True
    
    progress = FakeProgress()
    
    monkeypatch.setattr(
        anc.pb,
        "ProgressBar",
        lambda *args, **kwargs: progress,
    )
    
    result = anc.multicore(
        anc.add,
        cores=2,
        multiargs={"x": [1, 2]},
        y=5,
        z=9,
        pbar=True,
    )
    
    assert result == [15, 16]
    assert progress.started
    assert progress.finished


# ---------------------------------------------------------------------------
# parse_literal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected, expected_type",
    [
        ("1", 1, int),
        ("-2", -2, int),
        ("1.5", 1.5, float),
        ("1e3", 1000.0, float),
        ("foobar", "foobar", str),
        (b"1", 1, int),
        (b"1.5", 1.5, float),
        (b"foobar", b"foobar", bytes),
    ],
)
def test_parse_literal_scalar(
        value,
        expected,
        expected_type,
):
    result = anc.parse_literal(value)
    
    assert result == expected
    assert isinstance(result, expected_type)


def test_parse_literal_list():
    assert anc.parse_literal(
        ["1", "2.2", "a", b"3"],
    ) == [
               1,
               2.2,
               "a",
               3,
           ]


@pytest.mark.parametrize(
    "value",
    [1, 1.5, None, ("1",), {"value": "1"}],
)
def test_parse_literal_rejects_invalid_input(value):
    with pytest.raises(
            TypeError,
            match=r"expected str\|bytes",
    ):
        anc.parse_literal(value)


# ---------------------------------------------------------------------------
# rescale
# ---------------------------------------------------------------------------

def test_rescale_default_range():
    assert anc.rescale(
        [1000, 2000, 3000],
    ) == [
               0.0,
               0.5,
               1.0,
           ]


def test_rescale_custom_range():
    assert anc.rescale(
        [1000, 2000, 3000],
        (1, 3),
    ) == [
               1.0,
               2.0,
               3.0,
           ]


def test_rescale_supports_descending_new_range():
    assert anc.rescale(
        [0, 5, 10],
        (1, -1),
    ) == pytest.approx([
        1,
        0,
        -1,
    ])


def test_rescale_preserves_input_order():
    assert anc.rescale(
        [10, 0, 5],
    ) == pytest.approx([
        1,
        0,
        0.5,
    ])


def test_rescale_rejects_single_unique_value():
    with pytest.raises(
            RuntimeError,
            match="only one unique value",
    ):
        anc.rescale(
            [1000, 1000],
        )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def test_run_returns_stdout_and_stderr_when_void_false():
    returncode, out, err = anc.run(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        ],
        void=False,
    )
    
    assert returncode == 0
    assert out == "out\n"
    assert err == "err\n"


def test_run_returns_none_when_void_true():
    result = anc.run(
        [
            sys.executable,
            "-c",
            "print('out')",
        ],
        void=True,
    )
    
    assert result is None


def test_run_flattens_nested_command_arguments():
    returncode, out, err = anc.run(
        [
            sys.executable,
            [
                "-c",
                "print('nested')",
            ],
        ],
        void=False,
    )
    
    assert returncode == 0
    assert out == "nested\n"
    assert err == ""


def test_run_converts_command_arguments_to_strings():
    returncode, out, err = anc.run(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1])",
            123,
        ],
        void=False,
    )
    
    assert returncode == 0
    assert out == "123\n"
    assert err == ""


def test_run_passes_stdin_lines():
    script = (
        "import sys; "
        "data = sys.stdin.read(); "
        "print(data.replace(chr(10), '|'))"
    )
    
    returncode, out, err = anc.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        inlist=[
            "first",
            "second",
        ],
        void=False,
    )
    
    assert returncode == 0
    assert out == "first|second|\n"
    assert err == ""


def test_run_uses_requested_working_directory(tmp_path):
    returncode, out, err = anc.run(
        [
            sys.executable,
            "-c",
            "import os; print(os.getcwd())",
        ],
        outdir=str(tmp_path),
        void=False,
    )
    
    assert returncode == 0
    assert Path(out.strip()).resolve() == tmp_path.resolve()
    assert err == ""


def test_run_passes_environment():
    env = os.environ.copy()
    env["SPATIALIST_TEST_VALUE"] = "hello"
    
    returncode, out, err = anc.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "print(os.environ['SPATIALIST_TEST_VALUE'])"
            ),
        ],
        env=env,
        void=False,
    )
    
    assert returncode == 0
    assert out == "hello\n"
    assert err == ""


def test_run_writes_stdout_to_logfile(tmp_path):
    logfile = tmp_path / "run.log"
    
    result = anc.run(
        [
            sys.executable,
            "-c",
            "print('logged')",
        ],
        logfile=str(logfile),
        void=False,
    )
    
    returncode, out, err = result
    
    assert returncode == 0
    assert out == ""
    assert err == ""
    
    text = logfile.read_text(encoding="utf-8")
    assert "logged\n" in text
    assert "#" * 70 in text


def test_run_appends_to_existing_logfile(tmp_path):
    logfile = tmp_path / "run.log"
    
    for value in ["first", "second"]:
        anc.run(
            [
                sys.executable,
                "-c",
                f"print('{value}')",
            ],
            logfile=str(logfile),
        )
    
    text = logfile.read_text(encoding="utf-8")
    
    assert "first\n" in text
    assert "second\n" in text
    assert text.count("#" * 70) == 2


def test_run_raises_calledprocesserror_by_default():
    with pytest.raises(sp.CalledProcessError) as exc:
        anc.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "print('failure', file=sys.stderr); "
                    "sys.exit(7)"
                ),
            ],
            void=False,
        )
    
    assert exc.value.returncode == 7
    assert exc.value.stderr == "failure\n"


def test_run_errorpass_returns_nonzero_result():
    returncode, out, err = anc.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('failure', file=sys.stderr); "
                "sys.exit(7)"
            ),
        ],
        void=False,
        errorpass=True,
    )
    
    assert returncode == 7
    assert out == ""
    assert err == "failure\n"


def test_run_raises_oserror_for_missing_executable(tmp_path):
    missing = tmp_path / "definitely-does-not-exist"
    
    with pytest.raises(OSError):
        anc.run(
            [str(missing)],
        )


# ---------------------------------------------------------------------------
# union
# ---------------------------------------------------------------------------

def test_list_intersection_returns_common_values():
    result = anc.list_intersection(
        [1, 2],
        [2, 3],
    )
    
    assert result == [2]


def test_union_eliminates_duplicates():
    result = anc.list_intersection(
        [1, 1],
        [1, 1],
    )
    
    assert result == [1]


# ---------------------------------------------------------------------------
# urlQueryParser
# ---------------------------------------------------------------------------

def test_urlqueryparser_adds_query():
    result = anc.urlQueryParser(
        "https://example.com/path",
        {
            "foo": "bar",
            "page": 1,
        },
    )
    
    parsed = urlparse(result)
    
    assert parsed.scheme == "https"
    assert parsed.netloc == "example.com"
    assert parsed.path == "/path"
    assert parse_qs(parsed.query) == {
        "foo": ["bar"],
        "page": ["1"],
    }


def test_urlqueryparser_replaces_existing_query():
    result = anc.urlQueryParser(
        "https://example.com/path?old=value",
        {
            "new": "value",
        },
    )
    
    parsed = urlparse(result)
    
    assert parse_qs(parsed.query) == {
        "new": ["value"],
    }
    assert "old" not in parsed.query


def test_urlqueryparser_preserves_fragment():
    result = anc.urlQueryParser(
        "https://example.com/path#section",
        {
            "foo": "bar",
        },
    )
    
    assert urlparse(result).fragment == "section"


def test_urlqueryparser_urlencodes_values():
    result = anc.urlQueryParser(
        "https://example.com",
        {
            "value": "a b&c",
        },
    )
    
    assert parse_qs(
        urlparse(result).query,
    ) == {
               "value": ["a b&c"],
           }


# ---------------------------------------------------------------------------
# parallel_apply_along_axis
# ---------------------------------------------------------------------------

def test_parallel_apply_along_axis_rejects_nonpositive_cores():
    with pytest.raises(
            ValueError,
            match="cores must be larger than 0",
    ):
        anc.parallel_apply_along_axis(
            np.sum,
            axis=1,
            arr=np.arange(12).reshape(3, 4),
            cores=0,
        )


@pytest.mark.parametrize("cores", [-1, -10])
def test_parallel_apply_along_axis_rejects_negative_cores(cores):
    with pytest.raises(
            ValueError,
            match="cores must be larger than 0",
    ):
        anc.parallel_apply_along_axis(
            np.sum,
            axis=1,
            arr=np.arange(12).reshape(3, 4),
            cores=cores,
        )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_parallel_apply_along_axis_single_core_matches_numpy(axis):
    arr = np.arange(
        2 * 3 * 4,
        dtype=np.float64,
    ).reshape(2, 3, 4)
    
    expected = np.apply_along_axis(
        np.mean,
        axis,
        arr,
    )
    result = anc.parallel_apply_along_axis(
        np.mean,
        axis=axis,
        arr=arr,
        cores=1,
    )
    
    np.testing.assert_allclose(
        result,
        expected,
    )


def test_parallel_apply_along_axis_forwards_kwargs_single_core():
    arr = np.arange(
        12,
        dtype=np.float64,
    ).reshape(3, 4)
    
    expected = np.apply_along_axis(
        np.quantile,
        1,
        arr,
        q=0.25,
    )
    result = anc.parallel_apply_along_axis(
        np.quantile,
        axis=1,
        arr=arr,
        cores=1,
        q=0.25,
    )
    
    np.testing.assert_allclose(
        result,
        expected,
    )


@pytest.mark.skipif(
    not HAS_PATHOS,
    reason="pathos is required for parallel processing",
)
def test_parallel_apply_along_axis_parallel_axis1_matches_numpy():
    rows = max(CPU_COUNT, 2)
    arr = np.arange(
        rows * 4,
        dtype=np.float64,
    ).reshape(rows, 4)
    
    expected = np.apply_along_axis(
        np.mean,
        1,
        arr,
    )
    result = anc.parallel_apply_along_axis(
        np.mean,
        axis=1,
        arr=arr,
        cores=2,
    )
    
    np.testing.assert_allclose(
        result,
        expected,
    )


@pytest.mark.skipif(
    not HAS_PATHOS,
    reason="pathos is required for parallel processing",
)
def test_parallel_apply_along_axis_parallel_axis0_matches_numpy():
    cols = max(CPU_COUNT, 2)
    arr = np.arange(
        4 * cols,
        dtype=np.float64,
    ).reshape(4, cols)
    
    expected = np.apply_along_axis(
        np.mean,
        0,
        arr,
    )
    result = anc.parallel_apply_along_axis(
        np.mean,
        axis=0,
        arr=arr,
        cores=2,
    )
    
    np.testing.assert_allclose(
        result,
        expected,
    )


@pytest.mark.skipif(
    not HAS_PATHOS,
    reason="pathos is required for parallel processing",
)
def test_parallel_apply_along_axis_parallel_handles_small_array():
    arr = np.arange(
        6,
        dtype=np.float64,
    ).reshape(2, 3)
    
    expected = np.apply_along_axis(
        np.mean,
        1,
        arr,
    )
    result = anc.parallel_apply_along_axis(
        np.mean,
        axis=1,
        arr=arr,
        cores=2,
    )
    
    np.testing.assert_allclose(
        result,
        expected,
    )


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------

def test_sampler_dim1_is_deterministic():
    mask = np.array([
        [False, True, True],
        [True, False, True],
    ])
    
    first = anc.sampler(
        mask,
        samples=3,
        dim=1,
        seed=42,
    )
    second = anc.sampler(
        mask,
        samples=3,
        dim=1,
        seed=42,
    )
    
    np.testing.assert_array_equal(
        first,
        second,
    )


def test_sampler_dim1_returns_only_matching_indices():
    mask = np.array([
        [False, True, True],
        [True, False, True],
    ])
    valid = set(
        np.where(mask.flatten())[0],
    )
    
    sample = anc.sampler(
        mask,
        samples=3,
        dim=1,
        seed=42,
    )
    
    assert set(sample).issubset(valid)


def test_sampler_dim1_without_samples_returns_all_matching_indices():
    mask = np.array([
        [False, True, True],
        [True, False, True],
    ])
    valid = set(
        np.where(mask.flatten())[0],
    )
    
    sample = anc.sampler(
        mask,
        samples=None,
        dim=1,
        seed=42,
    )
    
    assert set(sample) == valid
    assert len(sample) == len(valid)


def test_sampler_dim1_caps_sample_count_at_available_values():
    mask = np.array([
        [True, False],
        [False, True],
    ])
    
    sample = anc.sampler(
        mask,
        samples=10,
        dim=1,
        replace=False,
    )
    
    assert len(sample) == 2
    assert set(sample) == {
        0,
        3,
    }


def test_sampler_without_replacement_returns_unique_indices():
    mask = np.ones(
        (4, 4),
        dtype=bool,
    )
    
    sample = anc.sampler(
        mask,
        samples=10,
        dim=1,
        replace=False,
        seed=1,
    )
    
    assert len(sample) == len(set(sample))


def test_sampler_dim2_corresponds_to_dim1():
    mask = np.array([
        [False, True, True],
        [True, False, True],
        [True, True, False],
    ])
    
    one_dimensional = anc.sampler(
        mask,
        samples=4,
        dim=1,
        seed=42,
    )
    two_dimensional = anc.sampler(
        mask,
        samples=4,
        dim=2,
        seed=42,
    )
    
    assert two_dimensional.shape == (2, 4)
    
    reconstructed = np.ravel_multi_index(
        two_dimensional,
        mask.shape,
    )
    
    np.testing.assert_array_equal(
        reconstructed,
        one_dimensional,
    )


def test_sampler_dim2_positions_match_mask():
    mask = np.array([
        [False, True, True],
        [True, False, True],
        [True, True, False],
    ])
    
    sample = anc.sampler(
        mask,
        samples=4,
        dim=2,
        seed=42,
    )
    
    assert np.all(
        mask[
            sample[0],
            sample[1],
        ]
    )


def test_sampler_empty_mask_dim1_returns_empty_array():
    sample = anc.sampler(
        np.zeros(
            (2, 3),
            dtype=bool,
        ),
        samples=None,
        dim=1,
    )
    
    assert sample.shape == (0,)


def test_sampler_zero_samples_returns_empty_array():
    sample = anc.sampler(
        np.ones(
            (2, 3),
            dtype=bool,
        ),
        samples=0,
        dim=1,
    )
    
    assert sample.shape == (0,)


def test_sampler_rejects_invalid_dimension():
    with pytest.raises(
            ValueError,
            match="dim",
    ):
        anc.sampler(
            np.ones(
                (2, 3),
                dtype=bool,
            ),
            samples=2,
            dim=3,
        )


def test_sampler_dim2_without_samples_returns_all_matching_indices():
    mask = np.array([
        [False, True, True],
        [True, False, True],
    ])
    
    sample = anc.sampler(
        mask,
        samples=None,
        dim=2,
        seed=42,
    )
    
    assert sample.shape == (
        2,
        np.count_nonzero(mask),
    )
    assert np.all(
        mask[
            sample[0],
            sample[1],
        ]
    )


def test_sampler_dim2_caps_sample_count_at_available_values():
    mask = np.array([
        [True, False],
        [False, True],
    ])
    
    sample = anc.sampler(
        mask,
        samples=10,
        dim=2,
        replace=False,
    )
    
    assert sample.shape == (2, 2)
    assert np.all(
        mask[
            sample[0],
            sample[1],
        ]
    )


def test_sampler_replace_true_can_draw_more_than_available():
    mask = np.array([
        [True, False],
        [False, True],
    ])
    
    sample = anc.sampler(
        mask,
        samples=10,
        dim=1,
        replace=True,
        seed=42,
    )
    
    assert sample.shape == (10,)
    assert set(sample).issubset({
        0,
        3,
    })
