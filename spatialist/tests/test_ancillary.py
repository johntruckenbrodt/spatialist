import os
import pytest
import subprocess as sp
import spatialist.ancillary as anc


def test_dissolve_with_lists():
    assert anc.dissolve([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert anc.dissolve([[[1]]]) == [1]
    assert anc.dissolve(((1, 2,), (3, 4))) == [1, 2, 3, 4]
    assert anc.dissolve(((1, 2), (1, 2))) == [1, 2, 1, 2]


def test_union():
    assert anc.union([1], [1]) == [1]


def test_dictmerge():
    assert anc.dictmerge({'a': 1, 'b': 2}, {'c': 3, 'd': 4}) == {'a': 1, 'b': 2, 'c': 3, 'd': 4}


def test_parse_literal():
    assert anc.parse_literal(['1', '2.2', 'a']) == [1, 2.2, 'a']
    with pytest.raises(TypeError):
        anc.parse_literal(1)


def test_run(tmpdir, testdata):
    log = os.path.join(str(tmpdir), 'test_run.log')
    out, err = anc.run(cmd=['gdalinfo', testdata['tif']],
                       logfile=log, void=False)
    with pytest.raises(OSError):
        anc.run(['foobar'])
    with pytest.raises(sp.CalledProcessError):
        anc.run(['gdalinfo', 'foobar'])


def test_which():
    env = os.environ['PATH']
    os.environ['PATH'] = '{}{}{}'.format(os.environ['PATH'], os.path.pathsep, os.path.dirname(os.__file__))
    program = anc.which(os.__file__, os.F_OK)
    assert os.path.isfile(program)
    assert anc.which(program, os.F_OK) == program
    assert anc.which('foobar') is None
    os.environ['PATH'] = env


def test_multicore():
    assert anc.multicore(anc.add, cores=2, multiargs={'x': [1, 2]}, y=5, z=9) == [15, 16]
    assert anc.multicore(anc.add, cores=2, multiargs={'x': [1, 2], 'y': [5, 6]}, z=9) == [15, 17]
    # unknown argument in multiargs
    with pytest.raises(AttributeError):
        anc.multicore(anc.add, cores=2, multiargs={'foobar': [1, 2]}, y=5, z=9)
    # unknown argument in single args
    with pytest.raises(AttributeError):
        anc.multicore(anc.add, cores=2, multiargs={'x': [1, 2]}, y=5, foobar=9)
    # multiarg values of different length
    with pytest.raises(AttributeError):
        anc.multicore(anc.add, cores=2, multiargs={'x': [1, 2], 'y': [5, 6, 7]}, z=9)


def test_finder(tmpdir, testdata):
    dir = str(tmpdir)
    dir_sub1 = os.path.join(dir, 'testdir1')
    dir_sub2 = os.path.join(dir, 'testdir2')
    os.makedirs(dir_sub1)
    os.makedirs(dir_sub2)
    with open(os.path.join(dir_sub1, 'testfile1.txt'), 'w') as t1:
        t1.write('test')
    with open(os.path.join(dir_sub2, 'testfile2.txt'), 'w') as t2:
        t2.write('test')
    assert len(anc.finder(dir, ['test*'], foldermode=0)) == 2
    assert len(anc.finder(dir, ['test*'], foldermode=0, recursive=False)) == 0
    assert len(anc.finder(dir, ['test*'], foldermode=1)) == 4
    assert len(anc.finder(dir, ['test*'], foldermode=2)) == 2
    assert len(anc.finder([dir_sub1, dir_sub2], ['test*'])) == 2

    assert len(anc.finder(testdata['zip'], ['file*'])) == 3
    assert len(anc.finder(testdata['zip'], ['*'], foldermode=1)) == 5
    assert len(anc.finder(testdata['zip'], ['[a-z]{1}'], foldermode=2, regex=True)) == 2
    
    assert len(anc.finder(testdata['tar'], ['file*'])) == 3
    assert len(anc.finder(testdata['tar'], ['*'], foldermode=1)) == 5
    assert len(anc.finder(testdata['tar'], ['[a-z]{1}'], foldermode=2, regex=True)) == 2
    
    with pytest.raises(TypeError):
        anc.finder(1, [])
    
    with pytest.raises(ValueError):
        anc.finder(dir, ['test*'], foldermode=3)
    
    with pytest.raises(TypeError):
        anc.finder('foobar', ['test*'], foldermode=2)


def test_rescale():
    assert anc.rescale([1000, 2000, 3000], [1, 3]) == [1, 2, 3]
    with pytest.raises(RuntimeError):
        anc.rescale([1000, 1000])


def test_Queue():
    st = anc.Queue()
    st.push('a')
    assert st.pop() == 'a'
    assert st.length() == 0


def test_Stack():
    st = anc.Stack()
    assert st.empty() is True
    st = anc.Stack(['a', 'b'])
    assert st.length() == 2
    st = anc.Stack('a')
    st.push('b')
    st.push(['c', 'd'])
    assert st.pop() == 'd'
    st.flush()
    assert st.empty() is True


def test_urlQueryParser():
    assert anc.urlQueryParser('www.somepage.foo', {'foo': 'bar', 'page': 1}) in ['www.somepage.foo?foo=bar&page=1',
                                                                                 'www.somepage.foo?page=1&foo=bar']
