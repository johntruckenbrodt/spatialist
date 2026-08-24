import os
from datetime import datetime

import numpy as np
import pytest
from osgeo import gdal, ogr, osr

import spatialist.raster as raster_module
from spatialist.raster import (
    Dtype,
    Raster,
    apply_along_time,
    png,
    rasterize,
    reproject,
)
from spatialist.vector import Vector, bbox, wkt2vector

GEOTRANSFORM = (100.0, 10.0, 0.0, 200.0, 0.0, -10.0)
EPSG = 32631
NODATA = -9999.0


def _projection(epsg=EPSG):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    if hasattr(srs, "SetAxisMappingStrategy"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs.ExportToWkt()


def _create_raster(
        path,
        arrays,
        *,
        dtype=gdal.GDT_Float32,
        nodata=NODATA,
        geotransform=GEOTRANSFORM,
        epsg=EPSG,
        driver="GTiff",
        metadata=None,
):
    if isinstance(arrays, np.ndarray) and arrays.ndim == 2:
        arrays = [arrays]
    arrays = list(arrays)
    rows, cols = arrays[0].shape
    
    ds = gdal.GetDriverByName(driver).Create(
        str(path),
        cols,
        rows,
        len(arrays),
        dtype,
    )
    ds.SetGeoTransform(geotransform)
    if epsg is not None:
        ds.SetProjection(_projection(epsg))
    if metadata:
        ds.SetMetadata(metadata)
    
    if isinstance(nodata, (list, tuple)):
        nodata_values = list(nodata)
    else:
        nodata_values = [nodata] * len(arrays)
    
    for index, (array, nodata_value) in enumerate(
            zip(arrays, nodata_values),
            start=1,
    ):
        band = ds.GetRasterBand(index)
        if nodata_value is not None:
            band.SetNoDataValue(nodata_value)
        band.WriteArray(array)
        band.FlushCache()
    
    ds = None
    return path


@pytest.fixture
def single_array():
    array = np.arange(24, dtype=np.float32).reshape(4, 6)
    array[0, 0] = NODATA
    return array


@pytest.fixture
def single_raster_path(tmp_path, single_array):
    return _create_raster(
        tmp_path / "single.tif",
        single_array,
        metadata={"TEST_KEY": "TEST_VALUE"},
    )


@pytest.fixture
def byte_raster_path(tmp_path):
    array = np.arange(24, dtype=np.uint8).reshape(4, 6)
    array[0, 0] = 255
    return _create_raster(
        tmp_path / "byte.tif",
        array,
        dtype=gdal.GDT_Byte,
        nodata=255,
    )


@pytest.fixture
def multiband_raster():
    arrays = [
        np.full((4, 6), 1, dtype=np.float32),
        np.full((4, 6), 2, dtype=np.float32),
        np.full((4, 6), 3, dtype=np.float32),
    ]
    arrays[0][0, 0] = -1
    arrays[1][0, 1] = -2
    arrays[2][0, 2] = -3
    
    ds = gdal.GetDriverByName("MEM").Create(
        "",
        6,
        4,
        3,
        gdal.GDT_Float32,
    )
    ds.SetGeoTransform(GEOTRANSFORM)
    ds.SetProjection(_projection())
    
    for index, (array, nodata) in enumerate(
            zip(arrays, [-1, -2, -3]),
            start=1,
    ):
        band = ds.GetRasterBand(index)
        band.SetNoDataValue(nodata)
        band.WriteArray(array)
    
    raster = Raster(ds)
    
    try:
        yield raster
    finally:
        raster.close()
        ds = None


@pytest.fixture
def multiband_raster_path(tmp_path):
    arrays = [
        np.full((4, 6), 1, dtype=np.float32),
        np.full((4, 6), 2, dtype=np.float32),
        np.full((4, 6), 3, dtype=np.float32),
    ]
    arrays[0][0, 0] = -1
    arrays[1][0, 1] = -2
    arrays[2][0, 2] = -3
    return _create_raster(
        tmp_path / "multiband.tif",
        arrays,
        nodata=[-1, -2, -3],
    )


@pytest.fixture
def rgb_raster_path(tmp_path):
    arrays = [
        np.arange(24, dtype=np.uint8).reshape(4, 6),
        np.arange(24, dtype=np.uint8).reshape(4, 6) + 20,
        np.arange(24, dtype=np.uint8).reshape(4, 6) + 40,
    ]
    return _create_raster(
        tmp_path / "rgb.tif",
        arrays,
        dtype=gdal.GDT_Byte,
        nodata=255,
    )


@pytest.fixture
def stack_paths(tmp_path):
    first = _create_raster(
        tmp_path / "first.tif",
        np.full((4, 6), 1, dtype=np.float32),
    )
    second = _create_raster(
        tmp_path / "second.tif",
        np.full((4, 6), 2, dtype=np.float32),
    )
    return [first, second]


# ---------------------------------------------------------------------------
# Raster construction and lifecycle
# ---------------------------------------------------------------------------


def test_raster_rejects_invalid_input_type():
    with pytest.raises(RuntimeError, match="raster input"):
        Raster(1)


def test_raster_rejects_single_element_file_list(single_raster_path):
    with pytest.raises(RuntimeError, match="less than two"):
        Raster([str(single_raster_path)])


def test_raster_rejects_invalid_driver_type(single_raster_path):
    with pytest.raises(RuntimeError, match='"driver" must be of type str or list'):
        Raster(str(single_raster_path), driver=1)


def test_raster_opens_path(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        assert raster.rows == 4
        assert raster.cols == 6
        assert raster.bands == 1


def test_raster_opens_path_with_single_driver(single_raster_path):
    with Raster(str(single_raster_path), driver="GTiff") as raster:
        assert raster.format == "GTiff"


def test_raster_opens_path_with_driver_list(single_raster_path):
    with Raster(
            str(single_raster_path),
            driver=["ENVI", "GTiff"],
    ) as raster:
        assert raster.format == "GTiff"


def test_raster_wraps_gdal_dataset():
    ds = gdal.GetDriverByName("MEM").Create("", 3, 2, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(GEOTRANSFORM)
    ds.SetProjection(_projection())
    
    raster = Raster(ds)
    try:
        assert raster.raster is ds
        assert raster.filename is None
        assert raster.dim == (2, 3, 1)
    finally:
        raster.close()
        ds = None


def test_raster_file_stack_creates_separate_bands(stack_paths):
    with Raster([str(x) for x in stack_paths]) as raster:
        assert raster.bands == 2
        assert raster.bandnames == ["first", "second"]
        assert np.all(raster.matrix(1) == 1)
        assert np.all(raster.matrix(2) == 2)


def test_raster_file_list_can_create_mosaic(stack_paths):
    with Raster(
            [str(x) for x in stack_paths],
            list_separate=False,
    ) as raster:
        assert raster.bands == 1
        assert raster.bandnames == ["mosaic"]


def test_raster_list_stack_temporary_vrt_is_removed_on_close(stack_paths):
    raster = Raster([str(x) for x in stack_paths])
    vrt = raster.filename
    
    assert vrt is not None
    assert os.path.isfile(vrt)
    
    raster.close()
    
    assert not os.path.exists(vrt)


def test_raster_close_does_not_delete_source(single_raster_path):
    raster = Raster(str(single_raster_path))
    raster.close()
    
    assert single_raster_path.exists()


def test_raster_context_manager_returns_self(single_raster_path):
    raster = Raster(str(single_raster_path))
    with raster as entered:
        assert entered is raster


def test_raster_default_bandnames(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        assert raster.bandnames == ["band1"]


def test_raster_bandnames_can_be_changed(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        raster.bandnames = ["amplitude"]
        assert raster.bandnames == ["amplitude"]


def test_raster_bandnames_require_list(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(TypeError, match="must be of type list"):
            raster.bandnames = "amplitude"


def test_raster_bandnames_length_must_match_bands(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(ValueError, match="length mismatch"):
            raster.bandnames = ["a", "b"]


def test_raster_accepts_timestamp_list(multiband_raster_path):
    timestamps = [
        datetime(2026, 1, 1),
        datetime(2026, 1, 2),
        datetime(2026, 1, 3),
    ]
    
    with Raster(
            str(multiband_raster_path),
            timestamps=timestamps,
    ) as raster:
        assert raster.timestamps == timestamps


def test_raster_rejects_timestamp_length_mismatch(multiband_raster_path):
    with pytest.raises(RuntimeError, match="number of time stamps"):
        Raster(
            str(multiband_raster_path),
            timestamps=[datetime(2026, 1, 1)],
        )


def test_raster_accepts_timestamp_function(multiband_raster_path):
    with Raster(
            str(multiband_raster_path),
            timestamps=lambda name: f"time-{name}",
    ) as raster:
        assert raster.timestamps == [
            "time-band1",
            "time-band2",
            "time-band3",
        ]


def test_raster_string_contains_metadata(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        text = str(raster)
    
    assert "spatialist Raster object" in text
    assert "dimensions : 4, 6, 1" in text
    assert "resolution : 10.0, -10.0" in text
    assert "EPSG:32631" in text


def test_raster_string_contains_time_range(multiband_raster_path):
    timestamps = [
        datetime(2026, 1, 3),
        datetime(2026, 1, 1),
        datetime(2026, 1, 2),
    ]
    
    with Raster(
            str(multiband_raster_path),
            timestamps=timestamps,
    ) as raster:
        text = str(raster)
    
    assert "time range : 2026-01-01 00:00:00 .. 2026-01-03 00:00:00" in text


def test_prepend_vsi_directive_zip():
    raster = object.__new__(Raster)
    
    result = raster._Raster__prependVSIdirective("archive.zip/file.tif")
    
    assert result == "/vsizip/archive.zip/file.tif"


def test_prepend_vsi_directive_does_not_duplicate_zip_prefix():
    raster = object.__new__(Raster)
    
    result = raster._Raster__prependVSIdirective(
        "/vsizip/archive.zip/file.tif",
    )
    
    assert result == "/vsizip/archive.zip/file.tif"


def test_prepend_vsi_directive_tar():
    raster = object.__new__(Raster)
    
    result = raster._Raster__prependVSIdirective("archive.tar/file.tif")
    
    assert result == "/vsitar/archive.tar/file.tif"


def test_prepend_vsi_directive_handles_lists():
    raster = object.__new__(Raster)
    
    result = raster._Raster__prependVSIdirective([
        "one.zip/a.tif",
        "two.tar/b.tif",
        "plain.tif",
    ])
    
    assert result == [
        "/vsizip/one.zip/a.tif",
        "/vsitar/two.tar/b.tif",
        "plain.tif",
    ]


def test_create_tmp_name_uses_spatialist_directory():
    path = Raster._Raster__create_tmp_name(".vrt")
    
    assert path.endswith(".vrt")
    assert os.path.basename(os.path.dirname(path)) == "spatialist"


# ---------------------------------------------------------------------------
# Raster metadata and coordinate conversion
# ---------------------------------------------------------------------------


def test_raster_dimensions(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        assert raster.rows == 4
        assert raster.cols == 6
        assert raster.bands == 1
        assert raster.dim == (4, 6, 1)


def test_raster_driver_and_format(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        assert isinstance(raster.driver, gdal.Driver)
        assert raster.driver.ShortName == "GTiff"
        assert raster.format == "GTiff"


def test_raster_dtype(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        assert raster.dtype == "Float32"


def test_raster_geo(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        assert raster.geo == {
            "xmin": 100.0,
            "xres": 10.0,
            "rotation_x": 0.0,
            "ymax": 200.0,
            "rotation_y": 0.0,
            "yres": -10.0,
            "xmax": 160.0,
            "ymin": 160.0,
        }


def test_raster_extent(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        assert raster.extent == {
            "xmin": 100.0,
            "xmax": 160.0,
            "ymin": 160.0,
            "ymax": 200.0,
        }


def test_raster_resolution_is_positive(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        assert raster.res == (10.0, 10.0)


def test_raster_projection_properties(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        assert raster.epsg == EPSG
        assert raster.srs.IsProjected()
        assert raster.projcs is not None
        assert raster.geogcs == "WGS 84"
        assert "+proj=utm" in raster.proj4
        assert raster.proj4args["proj"] == "utm"
        assert raster.proj4args["zone"] == "31"


def test_raster_projection_returns_wkt(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        srs = osr.SpatialReference(wkt=raster.projection)
    
    assert srs.GetAuthorityCode(None) == str(EPSG)


def test_raster_files_are_absolute(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        files = raster.files
    
    assert files is not None
    assert all(os.path.isabs(x) for x in files)
    assert os.path.abspath(single_raster_path) in files


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (120, None, 2),
        (None, 180, 2),
        (120, 180, (2, 2)),
    ],
)
def test_coord_map2img(single_raster_path, x, y, expected):
    with Raster(str(single_raster_path)) as raster:
        assert raster.coord_map2img(x=x, y=y) == expected


def test_coord_map2img_requires_coordinate(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(TypeError, match="cannot be None"):
            raster.coord_map2img()


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (2, None, 120.0),
        (None, 2, 180.0),
        (2, 2, (120.0, 180.0)),
    ],
)
def test_coord_img2map(single_raster_path, x, y, expected):
    with Raster(str(single_raster_path)) as raster:
        assert raster.coord_img2map(x=x, y=y) == expected


def test_coord_img2map_requires_coordinate(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(TypeError, match="cannot be None"):
            raster.coord_img2map()


# ---------------------------------------------------------------------------
# Statistics, arrays, caching, and extraction
# ---------------------------------------------------------------------------


def test_raster_layers(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        layers = raster.layers()
    
    assert len(layers) == 1
    assert isinstance(layers[0], gdal.Band)


def test_raster_allstats(single_raster_path, single_array):
    valid = single_array[single_array != NODATA]
    
    with Raster(str(single_raster_path)) as raster:
        stats = raster.allstats()
    
    assert len(stats) == 1
    assert stats[0]["min"] == pytest.approx(valid.min())
    assert stats[0]["max"] == pytest.approx(valid.max())
    assert stats[0]["mean"] == pytest.approx(valid.mean())
    assert stats[0]["sdev"] == pytest.approx(valid.std())


def test_raster_allstats_forwards_approximate_flag():
    calls = []
    
    class FakeBand:
        def ComputeStatistics(self, approximate):
            calls.append(approximate)
            return [1.0, 2.0, 1.5, 0.5]
    
    class FakeDataset:
        RasterCount = 1
        
        @staticmethod
        def GetRasterBand(index):
            return FakeBand()
    
    raster = object.__new__(Raster)
    raster.raster = FakeDataset()
    
    assert raster.allstats(approximate=True) == [{
        "min": 1.0,
        "max": 2.0,
        "mean": 1.5,
        "sdev": 0.5,
    }]
    assert calls == [True]


def test_raster_matrix_masks_nodata(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        matrix = raster.matrix()
    
    assert matrix.dtype == np.float32
    assert np.isnan(matrix[0, 0])
    assert matrix[0, 1] == 1


def test_raster_matrix_can_keep_nodata(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        matrix = raster.matrix(mask_nan=False)
    
    assert matrix[0, 0] == NODATA


def test_raster_matrix_casts_integer_data_when_masking_nan(
        byte_raster_path,
):
    with Raster(str(byte_raster_path)) as raster:
        matrix = raster.matrix(mask_nan=True)
    
    assert matrix.dtype == np.float32
    assert np.isnan(matrix[0, 0])


def test_raster_array_casts_integer_data_when_masking_nan(
        byte_raster_path,
):
    with Raster(str(byte_raster_path)) as raster:
        array = raster.array(mask_nan=True)
    
    assert array.dtype == np.float32
    assert np.isnan(array[0, 0])


def test_raster_matrix_uses_band_specific_nodata(multiband_raster):
    second = multiband_raster.matrix(band=2)
    
    assert np.isnan(second[0, 1])
    assert second[0, 0] == 2


def test_raster_array_single_band_matches_matrix(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        array = raster.array()
        matrix = raster.matrix()
    
    np.testing.assert_allclose(array, matrix, equal_nan=True)


def test_raster_array_multiband_moves_bands_to_last_axis(
        multiband_raster_path,
):
    with Raster(str(multiband_raster_path)) as raster:
        array = raster.array(mask_nan=False)
    
    assert array.shape == (4, 6, 3)
    assert np.all(array[:, :, 0][1:, :] == 1)
    assert np.all(array[:, :, 1][1:, :] == 2)
    assert np.all(array[:, :, 2][1:, :] == 3)


def test_raster_array_masks_band_specific_nodata(multiband_raster):
    array = multiband_raster.array()
    
    assert np.isnan(array[0, 0, 0])
    assert np.isnan(array[0, 1, 1])
    assert np.isnan(array[0, 2, 2])


def test_raster_array_masks_shared_multiband_nodata(tmp_path):
    arrays = [
        np.array([[255, 1], [2, 3]], dtype=np.uint8),
        np.array([[4, 255], [5, 6]], dtype=np.uint8),
    ]
    path = _create_raster(
        tmp_path / "shared-nodata.tif",
        arrays,
        dtype=gdal.GDT_Byte,
        nodata=255,
    )
    
    with Raster(str(path)) as raster:
        array = raster.array()
    
    assert np.isnan(array[0, 0, 0])
    assert np.isnan(array[0, 1, 1])


def test_raster_nodata_scalar(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        assert raster.nodata == NODATA


def test_raster_nodata_list(multiband_raster):
    assert multiband_raster.nodata == [-1.0, -2.0, -3.0]


def test_raster_assign_replaces_cached_band(single_raster_path):
    replacement = np.full((4, 6), 42, dtype=np.float32)
    
    with Raster(str(single_raster_path)) as raster:
        raster.assign(replacement, band=0)
        result = raster.matrix()
    
    assert result is replacement


def test_raster_load_caches_all_bands(multiband_raster_path):
    with Raster(str(multiband_raster_path)) as raster:
        raster.load()
        
        cached = raster._Raster__data
        
        assert len(cached) == 3
        assert all(isinstance(array, np.ndarray) for array in cached)
        assert cached[0][1, 1] == 1
        assert cached[1][1, 1] == 2
        assert cached[2][1, 1] == 3


def test_raster_rescale_updates_single_band(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        raster.rescale(lambda array: array * 2)
        result = raster.matrix()
    
    assert result[1, 1] == 14
    assert np.isnan(result[0, 0])


def test_raster_rescale_rejects_multiband(multiband_raster_path):
    with Raster(str(multiband_raster_path)) as raster:
        with pytest.raises(ValueError, match="single band"):
            raster.rescale(lambda array: array)


def test_raster_extract_uniform_values(tmp_path):
    path = _create_raster(
        tmp_path / "uniform.tif",
        np.full((5, 5), 7, dtype=np.float32),
        nodata=-9999,
        geotransform=(0, 1, 0, 5, 0, -1),
    )
    
    with Raster(str(path)) as raster:
        value = raster.extract(px=2.25, py=2.25, radius=1)
    
    assert value == pytest.approx(7)


def test_raster_extract_uses_cached_data(tmp_path):
    path = _create_raster(
        tmp_path / "cached.tif",
        np.ones((5, 5), dtype=np.float32),
        nodata=-9999,
        geotransform=(0, 1, 0, 5, 0, -1),
    )
    
    with Raster(str(path)) as raster:
        replacement = np.full((5, 5), 9, dtype=np.float32)
        raster.assign(replacement, band=0)
        value = raster.extract(px=2.25, py=2.25, radius=1)
    
    assert value == pytest.approx(9)


def test_raster_extract_rejects_x_outside_extent(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(RuntimeError, match="px is out of bounds"):
            raster.extract(px=99, py=180)


def test_raster_extract_rejects_y_outside_extent(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(RuntimeError, match="py is out of bounds"):
            raster.extract(px=120, py=201)


def test_raster_extract_returns_nodata_if_no_valid_pixels(tmp_path):
    path = _create_raster(
        tmp_path / "nodata.tif",
        np.full((5, 5), -1, dtype=np.float32),
        nodata=-1,
        geotransform=(0, 1, 0, 5, 0, -1),
    )
    
    with Raster(str(path)) as raster:
        assert raster.extract(2.25, 2.25, radius=1) == -1


def test_raster_extract_accepts_nodata_override(tmp_path):
    array = np.full((5, 5), 7, dtype=np.float32)
    path = _create_raster(
        tmp_path / "override.tif",
        array,
        nodata=-9999,
        geotransform=(0, 1, 0, 5, 0, -1),
    )
    
    with Raster(str(path)) as raster:
        assert raster.extract(
            2.25,
            2.25,
            radius=1,
            nodata=7,
        ) == 7


def test_raster_is_valid(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        assert raster.is_valid()


def test_raster_is_valid_returns_false_on_checksum_error():
    class FakeBand:
        @staticmethod
        def Checksum():
            raise RuntimeError("broken raster")
    
    class FakeDataset:
        RasterCount = 1
        
        @staticmethod
        def GetRasterBand(index):
            return FakeBand()
    
    raster = object.__new__(Raster)
    raster.raster = FakeDataset()
    
    assert raster.is_valid() is False


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------


def test_raster_bbox_from_image(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with raster.bbox() as vector:
            assert vector.extent == raster.extent
            assert vector.getProjection("epsg") == EPSG


def test_raster_bbox_can_be_written(single_raster_path, tmp_path):
    destination = tmp_path / "bbox.geojson"
    
    with Raster(str(single_raster_path)) as raster:
        result = raster.bbox(outname=str(destination))
    
    assert result is None
    assert destination.exists()


def test_raster_bbox_rejects_unknown_source(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(RuntimeError, match="either 'image' or 'gcp'"):
            raster.bbox(source="invalid")


def test_raster_bbox_from_gcps():
    ds = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_Byte)
    gcps = [
        gdal.GCP(10, 20, 0, 0, 0),
        gdal.GCP(30, 20, 0, 1, 0),
        gdal.GCP(30, 40, 0, 1, 1),
        gdal.GCP(10, 40, 0, 0, 1),
    ]
    ds.SetGCPs(gcps, _projection(4326))
    
    raster = Raster(ds)
    try:
        with raster.bbox(source="gcp") as vector:
            assert vector.extent == {
                "xmin": 10.0,
                "xmax": 30.0,
                "ymin": 20.0,
                "ymax": 40.0,
            }
            assert vector.getProjection("epsg") == 4326
    finally:
        raster.close()
        ds = None


# ---------------------------------------------------------------------------
# Raster subsetting
# ---------------------------------------------------------------------------


def test_raster_getitem_rejects_invalid_index(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(TypeError, match="index must be"):
            _ = raster[1]


def test_raster_getitem_rejects_dimension_mismatch(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(IndexError, match="mismatch of index length"):
            _ = raster[:, :, :]


def test_raster_getitem_rejects_row_step(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(IndexError, match="step slicing of rows"):
            _ = raster[::2, :]


def test_raster_getitem_rejects_column_step(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(IndexError, match="step slicing of cols"):
            _ = raster[:, ::2]


def test_raster_getitem_integer_slices(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with raster[1:3, 2:5] as subset:
            assert subset.dim == (2, 3, 1)
            assert subset.extent == {
                "xmin": 120.0,
                "xmax": 150.0,
                "ymin": 170.0,
                "ymax": 190.0,
            }


def test_raster_getitem_map_coordinate_slices(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with raster[170.0:190.0, 120.0:150.0] as subset:
            assert subset.dim == (2, 3, 1)
            assert subset.extent == {
                "xmin": 120.0,
                "xmax": 150.0,
                "ymin": 170.0,
                "ymax": 190.0,
            }


def test_raster_getitem_single_pixel_from_map_coordinates(
        single_raster_path,
):
    with Raster(str(single_raster_path)) as raster:
        with raster[180.0, 120.0] as subset:
            assert subset.dim == (1, 1, 1)


def test_raster_getitem_rejects_empty_subset(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(RuntimeError, match="no suitable subset"):
            _ = raster[1:1, :]


def test_raster_getitem_selects_band_by_integer(multiband_raster_path):
    with Raster(str(multiband_raster_path)) as raster:
        expected = raster.matrix(band=2, mask_nan=False)
        
        with raster[:, :, 1] as subset:
            assert subset.bands == 1
            assert subset.bandnames == ["band2"]
            
            np.testing.assert_array_equal(
                subset.matrix(mask_nan=False),
                expected,
            )


def test_raster_getitem_selects_band_by_name(multiband_raster_path):
    with Raster(str(multiband_raster_path)) as raster:
        raster.bandnames = ["red", "green", "blue"]
        expected = raster.matrix(band=2, mask_nan=False)
        
        with raster[:, :, "green"] as subset:
            assert subset.bands == 1
            assert subset.bandnames == ["green"]
            
            np.testing.assert_array_equal(
                subset.matrix(mask_nan=False),
                expected,
            )


def test_raster_getitem_selects_band_by_datetime(multiband_raster_path):
    timestamps = [
        datetime(2026, 1, 1),
        datetime(2026, 1, 2),
        datetime(2026, 1, 3),
    ]
    
    with Raster(
            str(multiband_raster_path),
            timestamps=timestamps,
    ) as raster:
        with raster[:, :, timestamps[1]] as subset:
            assert subset.bands == 1
            assert subset.timestamps == [timestamps[1]]


def test_raster_getitem_selects_datetime_range(tmp_path):
    timestamps = [
        datetime(2026, 1, 1),
        datetime(2026, 1, 2),
        datetime(2026, 1, 3),
        datetime(2026, 1, 4),
    ]
    path = _create_raster(
        tmp_path / "time-stack.tif",
        [
            np.full((2, 2), value, dtype=np.float32)
            for value in range(1, 5)
        ],
    )
    
    with Raster(str(path), timestamps=timestamps) as raster:
        with raster[
            :,
            :,
            timestamps[0]:timestamps[3],
        ] as subset:
            assert subset.timestamps == timestamps[1:3]
            assert subset.bands == 2


def test_raster_getitem_selects_band_slice(multiband_raster_path):
    with Raster(str(multiband_raster_path)) as raster:
        raster.bandnames = ["red", "green", "blue"]
        
        with raster[:, :, "red":"blue"] as subset:
            assert subset.bands == 2
            assert subset.bandnames == ["red", "green"]


def test_raster_getitem_rejects_invalid_band_index_type(
        multiband_raster_path,
):
    with Raster(str(multiband_raster_path)) as raster:
        with pytest.raises(TypeError, match="band indices"):
            _ = raster[:, :, 1.5]


def test_raster_getitem_rejects_invalid_band_slice_boundary(
        multiband_raster_path,
):
    with Raster(str(multiband_raster_path)) as raster:
        with pytest.raises(TypeError, match="band indices"):
            _ = raster[:, :, 1.5:]


def test_raster_getitem_vector_subset(single_raster_path):
    extent = {
        "xmin": 110,
        "xmax": 140,
        "ymin": 170,
        "ymax": 190,
    }
    
    with Raster(str(single_raster_path)) as raster:
        with bbox(extent, raster.projection) as vector:
            with raster[vector] as subset:
                assert subset.extent == {
                    "xmin": 110.0,
                    "xmax": 140.0,
                    "ymin": 170.0,
                    "ymax": 190.0,
                }


def test_raster_getitem_vector_subset_rejects_no_intersection(
        single_raster_path,
):
    extent = {
        "xmin": 1000,
        "xmax": 1010,
        "ymin": 1000,
        "ymax": 1010,
    }
    
    with Raster(str(single_raster_path)) as raster:
        with bbox(extent, raster.projection) as vector:
            with pytest.raises(RuntimeError, match="no intersection"):
                _ = raster[vector]


def test_raster_getitem_vector_subset_rejects_nonpolygon(
        single_raster_path,
):
    with Raster(str(single_raster_path)) as raster:
        with wkt2vector("POINT (120 180)", raster.projection) as vector:
            with pytest.raises(RuntimeError, match="only supported for POLYGON"):
                _ = raster[vector]


def test_raster_getitem_vector_subset_rejects_mixed_geometry_types(
        single_raster_path,
):
    with Raster(str(single_raster_path)) as raster:
        vector = Vector()
        vector.addlayer("mixed", raster.srs, ogr.wkbUnknown)
        vector.addfeature(ogr.CreateGeometryFromWkt(
            "POLYGON ((110 170, 140 170, 140 190, 110 190, 110 170))"
        ))
        vector.addfeature(ogr.CreateGeometryFromWkt("POINT (120 180)"))
        
        try:
            with pytest.raises(RuntimeError, match="one type of geometry"):
                _ = raster[vector]
        finally:
            vector.close()


def test_raster_getitem_rejects_antimeridian_intersection(
        single_raster_path,
        monkeypatch,
):
    class FakeIntersection:
        extent = {
            "xmin": 179,
            "xmax": -179,
            "ymin": 0,
            "ymax": 1,
        }
        
        @staticmethod
        def close():
            pass
    
    monkeypatch.setattr(
        raster_module,
        "intersect",
        lambda *args, **kwargs: FakeIntersection(),
    )
    
    with Raster(str(single_raster_path)) as raster:
        with bbox(
                {
                    "xmin": 110,
                    "xmax": 140,
                    "ymin": 170,
                    "ymax": 190,
                },
                raster.projection,
        ) as vector:
            with pytest.raises(
                    NotImplementedError,
                    match="across the antimeridian",
            ):
                _ = raster[vector]


def test_extent2slice_rejects_nonoverlap(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(RuntimeError, match="does not overlap"):
            raster._Raster__extent2slice({
                "xmin": 1000,
                "xmax": 1010,
                "ymin": 1000,
                "ymax": 1010,
            })


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_raster_write_creates_geotiff(single_raster_path, tmp_path):
    destination = tmp_path / "written.tif"
    
    with Raster(str(single_raster_path)) as raster:
        raster.write(str(destination))
    
    assert destination.exists()
    
    with Raster(str(destination)) as result:
        assert result.dim == (4, 6, 1)
        assert result.epsg == EPSG


def test_raster_write_appends_tif_extension(single_raster_path, tmp_path):
    destination = tmp_path / "written"
    
    with Raster(str(single_raster_path)) as raster:
        raster.write(str(destination), format="GTiff")
    
    assert (tmp_path / "written.tif").exists()


def test_raster_write_rejects_current_input_file(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(RuntimeError, match="currently opened file"):
            raster.write(str(single_raster_path))


def test_raster_write_rejects_existing_target(
        single_raster_path,
        tmp_path,
):
    destination = tmp_path / "existing.tif"
    destination.write_bytes(b"existing")
    
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(RuntimeError, match="target file already exists"):
            raster.write(str(destination))


def test_raster_write_can_overwrite_existing_target(
        single_raster_path,
        tmp_path,
):
    destination = tmp_path / "existing.tif"
    
    with Raster(str(single_raster_path)) as raster:
        raster.write(str(destination))
        raster.write(str(destination), overwrite=True)
    
    with Raster(str(destination)) as result:
        assert result.dim == (4, 6, 1)


def test_raster_write_custom_dtype(single_raster_path, tmp_path):
    destination = tmp_path / "byte.tif"
    
    with Raster(str(single_raster_path)) as raster:
        with pytest.warns(UserWarning, match="unsafe casting"):
            raster.write(
                str(destination),
                dtype="Byte",
                nodata=255,
            )
    
    with Raster(str(destination)) as result:
        assert result.dtype == "Byte"
        assert result.nodata == 255


def test_raster_write_custom_2d_array(single_raster_path, tmp_path):
    destination = tmp_path / "custom.tif"
    array = np.full((4, 6), 42, dtype=np.float32)
    
    with Raster(str(single_raster_path)) as raster:
        raster.write(str(destination), array=array)
    
    with Raster(str(destination)) as result:
        assert np.all(result.matrix(mask_nan=False) == 42)


def test_raster_write_custom_3d_array(multiband_raster_path, tmp_path):
    destination = tmp_path / "custom3d.tif"
    array = np.stack([
        np.full((4, 6), 10, dtype=np.float32),
        np.full((4, 6), 20, dtype=np.float32),
        np.full((4, 6), 30, dtype=np.float32),
    ], axis=2)
    
    with Raster(str(multiband_raster_path)) as raster:
        raster.write(str(destination), array=array)
    
    with Raster(str(destination)) as result:
        assert np.all(result.matrix(1, mask_nan=False) == 10)
        assert np.all(result.matrix(2, mask_nan=False) == 20)
        assert np.all(result.matrix(3, mask_nan=False) == 30)


def test_raster_write_update_with_offsets(single_raster_path, tmp_path):
    destination = tmp_path / "update.tif"
    patch = np.full((2, 2), 99, dtype=np.float32)
    
    with Raster(str(single_raster_path)) as raster:
        raster.write(str(destination))
        raster.write(
            str(destination),
            update=True,
            xoff=2,
            yoff=1,
            array=patch,
        )
    
    with Raster(str(destination)) as result:
        matrix = result.matrix(mask_nan=False)
    
    np.testing.assert_array_equal(matrix[1:3, 2:4], patch)


def test_raster_write_update_creates_missing_file(
        single_raster_path,
        tmp_path,
):
    destination = tmp_path / "new-update.tif"
    
    with Raster(str(single_raster_path)) as raster:
        raster.write(str(destination), update=True)
    
    assert destination.exists()


def test_raster_write_sets_custom_tiff_tag(single_raster_path, tmp_path):
    destination = tmp_path / "tagged.tif"
    
    with Raster(str(single_raster_path)) as raster:
        raster.write(
            str(destination),
            options=["TIFFTAG_SOFTWARE=spatialist-test"],
        )
    
    ds = gdal.Open(str(destination))
    try:
        assert ds.GetMetadataItem("TIFFTAG_SOFTWARE") == "spatialist-test"
    finally:
        ds = None


def test_raster_write_sets_default_datetime_tag(
        single_raster_path,
        tmp_path,
):
    destination = tmp_path / "datetime.tif"
    
    with Raster(str(single_raster_path)) as raster:
        raster.write(str(destination))
    
    ds = gdal.Open(str(destination))
    try:
        assert ds.GetMetadataItem("TIFFTAG_DATETIME") is not None
    finally:
        ds = None


def test_raster_write_preserves_metadata(single_raster_path, tmp_path):
    destination = tmp_path / "metadata.tif"
    
    with Raster(str(single_raster_path)) as raster:
        raster.write(str(destination))
    
    ds = gdal.Open(str(destination))
    try:
        assert ds.GetMetadataItem("TEST_KEY") == "TEST_VALUE"
    finally:
        ds = None


def test_raster_write_builds_overviews(single_raster_path, tmp_path):
    destination = tmp_path / "overview.tif"
    
    with Raster(str(single_raster_path)) as raster:
        raster.write(
            str(destination),
            overviews=[2],
            overview_resampling="NEAREST",
        )
    
    ds = gdal.Open(str(destination))
    try:
        assert ds.GetRasterBand(1).GetOverviewCount() == 1
    finally:
        ds = None


def test_raster_write_applies_color_table(byte_raster_path, tmp_path):
    destination = tmp_path / "colored.tif"
    cmap = gdal.ColorTable()
    cmap.SetColorEntry(0, (0, 0, 0, 255))
    cmap.SetColorEntry(1, (255, 255, 255, 255))
    
    with Raster(str(byte_raster_path)) as raster:
        raster.write(str(destination), cmap=cmap)
    
    ds = gdal.Open(str(destination))
    try:
        result = ds.GetRasterBand(1).GetRasterColorTable()
        assert result is not None
        assert result.GetColorEntry(1) == (255, 255, 255, 255)
    finally:
        ds = None


def test_raster_write_envi_preserves_bandnames(
        multiband_raster_path,
        tmp_path,
):
    destination = tmp_path / "output.bin"
    
    with Raster(str(multiband_raster_path)) as raster:
        raster.bandnames = ["red", "green", "blue"]
        raster.write(str(destination), format="ENVI")
    
    with Raster(str(destination)) as result:
        assert result.format == "ENVI"
        assert result.bandnames == ["red", "green", "blue"]


def test_raster_write_cog(single_raster_path, tmp_path):
    if gdal.GetDriverByName("COG") is None:
        pytest.skip("COG driver is not available")
    
    destination = tmp_path / "output.tif"
    
    with Raster(str(single_raster_path)) as raster:
        raster.write(str(destination), format="COG")
    
    with Raster(str(destination)) as result:
        assert result.dim == (4, 6, 1)
        assert result.epsg == EPSG


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, gdalint, gdalstr, numpystr",
    [
        ("Byte", gdal.GDT_Byte, "Byte", "uint8"),
        ("Float32", gdal.GDT_Float32, "Float32", "float32"),
        ("uint32", gdal.GDT_UInt32, "UInt32", "uint32"),
        (gdal.GDT_Int16, gdal.GDT_Int16, "Int16", "int16"),
    ],
)
def test_dtype_conversions(value, gdalint, gdalstr, numpystr):
    dtype = Dtype(value)
    
    assert dtype.gdalint == gdalint
    assert dtype.gdalstr == gdalstr
    assert dtype.numpystr == numpystr


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Byte", 1),
        ("UInt16", 2),
        ("Float32", 4),
        ("Float64", 8),
    ],
)
def test_dtype_bytes(value, expected):
    assert Dtype(value).bytes == expected


def test_dtype_rejects_unknown_string():
    with pytest.raises(ValueError, match="unknown data type"):
        Dtype("foobar")


def test_dtype_rejects_unknown_integer():
    with pytest.raises(ValueError, match="unknown data type"):
        Dtype(999)


def test_dtype_rejects_invalid_identifier_type():
    with pytest.raises(TypeError, match="must be of type int or str"):
        Dtype(None)


def test_dtype_numpy2gdalint_map():
    mapping = Dtype("Byte").numpy2gdalint
    
    assert mapping["uint8"] == gdal.GDT_Byte
    assert mapping["float32"] == gdal.GDT_Float32


def test_dtype_gdalstr2gdalint_map():
    mapping = Dtype("Byte").gdalstr2gdalint
    
    assert mapping["Byte"] == gdal.GDT_Byte
    assert mapping["Float32"] == gdal.GDT_Float32


def test_dtype_gdalint2numpystr_map():
    mapping = Dtype("Byte").gdalint2numpystr
    
    assert mapping[gdal.GDT_Byte] == "uint8"
    assert mapping[gdal.GDT_Float32] == "float32"


def test_dtype_gdalint2gdalstr_map():
    mapping = Dtype("Byte").gdalint2gdalstr
    
    assert mapping[gdal.GDT_Byte] == "Byte"
    assert mapping[gdal.GDT_Float32] == "Float32"


# ---------------------------------------------------------------------------
# PNG creation
# ---------------------------------------------------------------------------


def test_png_rejects_non_raster(tmp_path):
    with pytest.raises(TypeError, match="'src' must be of type Raster"):
        png("input.tif", str(tmp_path / "output.png"))


def test_png_rejects_two_band_raster(tmp_path):
    path = _create_raster(
        tmp_path / "two-band.tif",
        [
            np.ones((4, 6), dtype=np.uint8),
            np.ones((4, 6), dtype=np.uint8),
        ],
        dtype=gdal.GDT_Byte,
        nodata=255,
    )
    
    with Raster(str(path)) as raster:
        with pytest.raises(ValueError, match="either 1 or 3 bands"):
            png(raster, str(tmp_path / "output.png"))


def test_png_appends_extension(byte_raster_path, tmp_path):
    destination = tmp_path / "output"
    
    with Raster(str(byte_raster_path)) as raster:
        png(raster, str(destination), percent=100)
    
    assert (tmp_path / "output.png").exists()


def test_png_single_band_percentile_scaling(byte_raster_path, tmp_path):
    destination = tmp_path / "scaled.png"
    
    with Raster(str(byte_raster_path)) as raster:
        png(
            raster,
            str(destination),
            percent=100,
            scale=(10, 90),
        )
    
    ds = gdal.Open(str(destination))
    try:
        assert ds.RasterCount == 1
        assert ds.RasterXSize == 6
        assert ds.RasterYSize == 4
        assert ds.GetRasterBand(1).DataType == gdal.GDT_Byte
    finally:
        ds = None


def test_png_explicit_min_max_scaling(byte_raster_path, tmp_path):
    destination = tmp_path / "minmax.png"
    
    with Raster(str(byte_raster_path)) as raster:
        png(
            raster,
            str(destination),
            percent=100,
            vmin=0,
            vmax=23,
        )
    
    assert destination.exists()


def test_png_without_scaling(byte_raster_path, tmp_path):
    destination = tmp_path / "unscaled.png"
    
    with Raster(str(byte_raster_path)) as raster:
        png(
            raster,
            str(destination),
            percent=100,
            scale=None,
        )
    
    assert destination.exists()


def test_png_rgb(rgb_raster_path, tmp_path):
    destination = tmp_path / "rgb.png"
    
    with Raster(str(rgb_raster_path)) as raster:
        png(raster, str(destination), percent=100)
    
    ds = gdal.Open(str(destination))
    try:
        assert ds.RasterCount == 3
    finally:
        ds = None


def test_png_worldfile(byte_raster_path, tmp_path):
    destination = tmp_path / "world.png"
    
    with Raster(str(byte_raster_path)) as raster:
        png(
            raster,
            str(destination),
            percent=100,
            worldfile=True,
        )
    
    candidates = [
        tmp_path / "world.wld",
        tmp_path / "world.pgw",
    ]
    assert any(path.exists() for path in candidates)


def test_png_forwards_nodata(byte_raster_path, tmp_path, monkeypatch):
    captured = {}
    
    def fake_translate(src, dst, **kwargs):
        captured.update(kwargs)
    
    monkeypatch.setattr(raster_module, "gdal_translate", fake_translate)
    
    with Raster(str(byte_raster_path)) as raster:
        png(
            raster,
            str(tmp_path / "nodata.png"),
            percent=100,
            nodata=255,
        )
    
    assert captured["noData"] == 255


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


def test_rasterize_rejects_expression_burn_length_mismatch(
        single_raster_path,
):
    with Raster(str(single_raster_path)) as reference:
        with reference.bbox() as vector:
            with pytest.raises(RuntimeError, match="different length"):
                rasterize(
                    vector,
                    reference,
                    burn_values=[1],
                    expressions=["area=1", "area=2"],
                )


def test_rasterize_rejects_invalid_expression(single_raster_path):
    with Raster(str(single_raster_path)) as reference:
        with reference.bbox() as vector:
            with pytest.raises(RuntimeError, match="failed to set"):
                rasterize(
                    vector,
                    reference,
                    burn_values=[1],
                    expressions=["missing_field=1"],
                )


def test_rasterize_requires_raster_reference(single_raster_path):
    with Raster(str(single_raster_path)) as raster:
        with raster.bbox() as vector:
            with pytest.raises(RuntimeError, match="reference.*Raster"):
                rasterize(vector, reference="not-a-raster")


def test_rasterize_returns_memory_raster(single_raster_path):
    with Raster(str(single_raster_path)) as reference:
        with reference.bbox() as vector:
            with rasterize(vector, reference, burn_values=3) as result:
                assert isinstance(result, Raster)
                assert result.dim == (4, 6, 1)
                assert result.nodata == 0
                assert np.all(result.matrix(mask_nan=False) == 3)


def test_rasterize_resets_attribute_filter(single_raster_path):
    with Raster(str(single_raster_path)) as reference:
        with reference.bbox() as vector:
            vector.addfield("class", ogr.OFTInteger, values=[1])
            
            with rasterize(
                    vector,
                    reference,
                    burn_values=3,
                    expressions=["class=1"],
            ):
                pass
            
            assert vector.layer.GetFeatureCount() == 1


def test_rasterize_supports_no_nodata(single_raster_path):
    with Raster(str(single_raster_path)) as reference:
        with reference.bbox() as vector:
            with rasterize(
                    vector,
                    reference,
                    burn_values=3,
                    nodata=None,
            ) as result:
                assert result.nodata is None


def test_rasterize_multiple_expressions(single_raster_path):
    with Raster(str(single_raster_path)) as reference:
        extent_left = {
            "xmin": 100,
            "xmax": 130,
            "ymin": 160,
            "ymax": 200,
        }
        extent_right = {
            "xmin": 130,
            "xmax": 160,
            "ymin": 160,
            "ymax": 200,
        }
        with bbox(extent_left, reference.projection) as left:
            left.addfield("class", ogr.OFTInteger, values=[1])
            with bbox(extent_right, reference.projection) as right:
                right.addfield("class", ogr.OFTInteger, values=[2])
                left.addvector(right)
            
            with rasterize(
                    left,
                    reference,
                    burn_values=[10, 20],
                    expressions=["class=1", "class=2"],
            ) as result:
                matrix = result.matrix(mask_nan=False)
    
    assert np.all(matrix[:, :3] == 10)
    assert np.all(matrix[:, 3:] == 20)


def test_rasterize_writes_file(single_raster_path, tmp_path):
    destination = tmp_path / "mask.tif"
    
    with Raster(str(single_raster_path)) as reference:
        with reference.bbox() as vector:
            result = rasterize(
                vector,
                reference,
                outname=str(destination),
                burn_values=5,
            )
    
    assert result is None
    assert destination.exists()
    
    with Raster(str(destination)) as raster:
        assert np.all(raster.matrix(mask_nan=False) == 5)


def test_rasterize_append_updates_existing_file(
        single_raster_path,
        tmp_path,
):
    destination = tmp_path / "append.tif"
    
    with Raster(str(single_raster_path)) as reference:
        left_extent = {
            "xmin": 100,
            "xmax": 130,
            "ymin": 160,
            "ymax": 200,
        }
        right_extent = {
            "xmin": 130,
            "xmax": 160,
            "ymin": 160,
            "ymax": 200,
        }
        
        with bbox(left_extent, reference.projection) as left:
            rasterize(
                left,
                reference,
                outname=str(destination),
                burn_values=1,
            )
        
        with bbox(right_extent, reference.projection) as right:
            rasterize(
                right,
                reference=None,
                outname=str(destination),
                burn_values=2,
                append=True,
            )
    
    with Raster(str(destination)) as result:
        matrix = result.matrix(mask_nan=False)
    
    assert np.all(matrix[:, :3] == 1)
    assert np.all(matrix[:, 3:] == 2)


# ---------------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------------


def test_reproject_rejects_invalid_rasterobject(tmp_path):
    with pytest.raises(RuntimeError, match="Raster or str"):
        reproject(
            rasterobject=1,
            reference=4326,
            outname=str(tmp_path / "output.tif"),
            targetres=(0.1, 0.1),
        )


def test_reproject_rejects_invalid_reference(
        single_raster_path,
        tmp_path,
):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(TypeError, match="reference must be"):
            reproject(
                raster,
                reference=object(),
                outname=str(tmp_path / "output.tif"),
            )


def test_reproject_requires_resolution_for_crs_reference(
        single_raster_path,
        tmp_path,
):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(RuntimeError, match="targetres is missing"):
            reproject(
                raster,
                reference=4326,
                outname=str(tmp_path / "output.tif"),
            )


def test_reproject_rejects_unreadable_reference_crs(
        single_raster_path,
        tmp_path,
):
    with Raster(str(single_raster_path)) as raster:
        with pytest.raises(RuntimeError, match="reference projection cannot be read"):
            reproject(
                raster,
                reference="not-a-crs",
                outname=str(tmp_path / "output.tif"),
                targetres=(1, 1),
            )


def test_reproject_uses_raster_reference_resolution(
        single_raster_path,
        tmp_path,
):
    destination = tmp_path / "same-grid.tif"
    
    with Raster(str(single_raster_path)) as source:
        with Raster(str(single_raster_path)) as reference:
            reproject(
                source,
                reference,
                outname=str(destination),
                resampling="nearest",
            )
    
    with Raster(str(destination)) as result:
        assert result.epsg == EPSG
        assert result.res == (10.0, 10.0)


def test_reproject_accepts_raster_path(
        single_raster_path,
        tmp_path,
):
    destination = tmp_path / "from-path.tif"
    
    reproject(
        str(single_raster_path),
        EPSG,
        outname=str(destination),
        targetres=(10, 10),
        resampling="nearest",
    )
    
    assert destination.exists()


def test_reproject_to_epsg4326(single_raster_path, tmp_path):
    destination = tmp_path / "4326.tif"
    
    with Raster(str(single_raster_path)) as raster:
        reproject(
            raster,
            4326,
            outname=str(destination),
            targetres=(0.0001, 0.0001),
            resampling="nearest",
        )
    
    with Raster(str(destination)) as result:
        assert result.epsg == 4326
        assert result.res == pytest.approx((0.0001, 0.0001))


def test_reproject_accepts_osr_reference(single_raster_path, tmp_path):
    destination = tmp_path / "osr.tif"
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    
    with Raster(str(single_raster_path)) as raster:
        reproject(
            raster,
            srs,
            outname=str(destination),
            targetres=(10, 10),
        )
    
    assert destination.exists()


def test_reproject_accepts_vector_reference(
        single_raster_path,
        tmp_path,
):
    destination = tmp_path / "vector-reference.tif"
    
    with Raster(str(single_raster_path)) as raster:
        with raster.bbox() as vector:
            reproject(
                raster,
                vector,
                outname=str(destination),
                targetres=(10, 10),
            )
    
    assert destination.exists()


def test_reproject_vector_reference_requires_targetres(
        single_raster_path,
        tmp_path,
):
    destination = tmp_path / "vector-reference.tif"
    
    with Raster(str(single_raster_path)) as raster:
        with raster.bbox() as vector:
            with pytest.raises(
                    RuntimeError,
                    match="parameter targetres is missing",
            ):
                reproject(
                    raster,
                    vector,
                    outname=str(destination),
                )


# ---------------------------------------------------------------------------
# apply_along_time
# ---------------------------------------------------------------------------


def test_apply_along_time_processes_stack_in_chunks(
        multiband_raster_path,
        tmp_path,
        monkeypatch,
):
    destination = tmp_path / "mean.tif"
    
    def apply(func1d, axis, arr, cores, *args, **kwargs):
        return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
    
    monkeypatch.setattr(
        raster_module,
        "parallel_apply_along_axis",
        apply,
    )
    
    with Raster(str(multiband_raster_path)) as raster:
        apply_along_time(
            src=raster,
            dst=str(destination),
            func1d=np.nanmean,
            nodata=-9999,
            format="GTiff",
            maxlines=2,
            cores=1,
        )
    
    with Raster(str(destination)) as result:
        matrix = result.matrix(mask_nan=False)
    
    assert matrix.shape == (4, 6)
    assert matrix[1, 1] == pytest.approx(2)


def test_apply_along_time_caps_maxlines_to_rows(
        multiband_raster_path,
        tmp_path,
        monkeypatch,
        capsys,
):
    destination = tmp_path / "mean.tif"
    
    def apply(func1d, axis, arr, cores, *args, **kwargs):
        return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
    
    monkeypatch.setattr(
        raster_module,
        "parallel_apply_along_axis",
        apply,
    )
    
    with Raster(str(multiband_raster_path)) as raster:
        apply_along_time(
            src=raster,
            dst=str(destination),
            func1d=np.nanmean,
            nodata=-9999,
            format="GTiff",
            maxlines=100,
            cores=1,
        )
    
    assert capsys.readouterr().out.count("processing lines") == 1
    assert destination.exists()


def test_apply_along_time_defaults_to_all_rows(
        multiband_raster_path,
        tmp_path,
        monkeypatch,
        capsys,
):
    destination = tmp_path / "mean.tif"
    
    def apply(func1d, axis, arr, cores, *args, **kwargs):
        return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
    
    monkeypatch.setattr(
        raster_module,
        "parallel_apply_along_axis",
        apply,
    )
    
    with Raster(str(multiband_raster_path)) as raster:
        apply_along_time(
            src=raster,
            dst=str(destination),
            func1d=np.nanmean,
            nodata=-9999,
            format="GTiff",
            maxlines=None,
            cores=1,
        )
    
    assert capsys.readouterr().out.count("processing lines") == 1


# ---------------------------------------------------------------------------
# ENVI initialization
# ---------------------------------------------------------------------------


def test_raster_reads_envi_bandnames(multiband_raster_path, tmp_path):
    destination = tmp_path / "bands.bin"
    
    with Raster(str(multiband_raster_path)) as raster:
        raster.bandnames = ["alpha", "beta", "gamma"]
        raster.write(str(destination), format="ENVI")
    
    with Raster(str(destination)) as result:
        assert result.bandnames == ["alpha", "beta", "gamma"]
