import os

import numpy as np
import pytest
from osgeo import gdal, ogr, osr
from packaging.version import Version

import spatialist.auxil as auxil
from spatialist.auxil import (
    cmap_mpl2gdal,
    coordinate_reproject,
    crsConvert,
    gdal_rasterize,
    gdal_translate,
    gdalbuildvrt,
    gdalwarp,
    haversine,
    iter_geometries,
    iter_points,
    latlon_clamp,
    latlon_extent_center,
    latlon_normalize,
    longitude_shortest_interval,
    ogr2ogr,
    utm_autodetect,
)
from spatialist.vector import bbox

REGULAR_EXTENT = {
    "xmin": 11.5,
    "xmax": 11.7,
    "ymin": 50.8,
    "ymax": 51.0,
}


def _make_srs(epsg):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    if Version(gdal.__version__) >= Version("3.0"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


@pytest.fixture
def raster_file(tmp_path):
    path = tmp_path / "source.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(path),
        4,
        3,
        1,
        gdal.GDT_Byte,
    )
    dataset.SetGeoTransform((10.0, 0.5, 0.0, 52.0, 0.0, -0.5))
    dataset.SetProjection(_make_srs(4326).ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(
        np.arange(12, dtype=np.uint8).reshape(3, 4)
    )
    dataset.FlushCache()
    dataset = None
    driver = None
    return path


@pytest.fixture
def vector_file(tmp_path):
    path = tmp_path / "source.gpkg"
    with bbox(REGULAR_EXTENT, crs=4326) as vector:
        vector.write(str(path), driver="GPKG")
    return path


# ---------------------------------------------------------------------------
# crsConvert
# ---------------------------------------------------------------------------

def test_crs_convert_epsg_integer_to_epsg():
    assert crsConvert(4326, "epsg") == 4326


def test_crs_convert_epsg_integer_to_osr():
    result = crsConvert(4326, "osr")
    
    assert isinstance(result, osr.SpatialReference)
    assert crsConvert(result, "epsg") == 4326


def test_crs_convert_spatial_reference_is_cloned():
    source = _make_srs(4326)
    result = crsConvert(source, "osr")
    
    assert result is not source
    assert result.IsSame(source) == 1


def test_crs_convert_wkt_roundtrip():
    wkt = crsConvert(4326, "wkt")
    assert crsConvert(wkt, "epsg") == 4326


def test_crs_convert_pretty_wkt_roundtrip():
    pretty_wkt = crsConvert(4326, "prettyWkt")
    assert crsConvert(pretty_wkt, "epsg") == 4326


def test_crs_convert_proj4_roundtrip():
    proj4 = crsConvert(4326, "proj4")
    assert crsConvert(proj4, "epsg") == 4326


def test_crs_convert_opengis():
    assert (
            crsConvert(4326, "opengis")
            == "https://www.opengis.net/def/crs/EPSG/0/4326"
    )


def test_crs_convert_opengis_roundtrip():
    uri = "https://www.opengis.net/def/crs/EPSG/0/4326"
    assert crsConvert(uri, "epsg") == 4326


@pytest.mark.skipif(
    Version(gdal.__version__) < Version("3.0"),
    reason="WKT2 export requires GDAL 3",
)
def test_crs_convert_explicit_wkt_format():
    wkt = crsConvert(4326, "wkt", wkt_format="WKT2_2019")
    
    assert wkt.startswith("GEOGCRS[")
    assert crsConvert(wkt, "epsg") == 4326


def test_crs_convert_compound_crs_to_proj4():
    proj4 = crsConvert("EPSG:4326+5773", "proj4")
    
    assert "+proj=longlat" in proj4
    assert "+vunits=m" in proj4


def test_crs_convert_invalid_crs_string():
    with pytest.raises(TypeError, match="crsIn not recognized"):
        crsConvert("xyz", "epsg")


def test_crs_convert_invalid_input_type():
    with pytest.raises(TypeError, match="crsIn must be of type"):
        crsConvert([], "epsg")


def test_crs_convert_invalid_output_type():
    with pytest.raises(ValueError, match="crsOut not recognized"):
        crsConvert(4326, "xyz")


# ---------------------------------------------------------------------------
# haversine
# ---------------------------------------------------------------------------

def test_haversine_zero_distance():
    assert haversine(50, 10, 50, 10) == 0.0


def test_haversine_one_degree_latitude():
    assert haversine(50, 10, 51, 10) == pytest.approx(
        111194.92664455889
    )


def test_haversine_is_symmetric():
    forward = haversine(50, 10, 51, 12)
    reverse = haversine(51, 12, 50, 10)
    
    assert forward == pytest.approx(reverse)


def test_haversine_antimeridian():
    result = haversine(0, 179, 0, -179)
    
    assert result == pytest.approx(222389.85328911748)


# ---------------------------------------------------------------------------
# GDAL wrappers
# ---------------------------------------------------------------------------

def test_gdalwarp_writes_reprojected_raster(raster_file, tmp_path):
    destination = tmp_path / "warped.tif"
    
    result = gdalwarp(
        src=str(raster_file),
        dst=str(destination),
        format="GTiff",
        dstSRS="EPSG:3857",
    )
    
    assert result is None
    assert destination.exists()
    
    dataset = gdal.Open(str(destination))
    assert crsConvert(dataset.GetProjection(), "epsg") == 3857
    dataset = None


def test_gdalwarp_progress_bar(monkeypatch):
    state = {
        "updates": [],
        "finished": False,
    }
    
    class FakeProgressBar:
        def __init__(self, max_value, widgets):
            assert max_value == 100
            assert widgets
        
        def start(self):
            return self
        
        def update(self, value):
            state["updates"].append(value)
        
        def finish(self):
            state["finished"] = True
    
    def fake_warp_options(**kwargs):
        return kwargs
    
    def fake_warp(dst, src, options):
        assert dst == "dst"
        assert src == "src"
        options["callback"](0.42, "", options["callback_data"])
        return object()
    
    monkeypatch.setattr(auxil.pb, "ProgressBar", FakeProgressBar)
    monkeypatch.setattr(auxil.gdal, "WarpOptions", fake_warp_options)
    monkeypatch.setattr(auxil.gdal, "Warp", fake_warp)
    
    gdalwarp(src="src", dst="dst", pbar=True)
    
    assert state["updates"] == [42]
    assert state["finished"] is True


def test_gdalwarp_runtime_error_adds_context(monkeypatch):
    def fake_warp(dst, src, options):
        raise RuntimeError("warp failed")
    
    monkeypatch.setattr(auxil.gdal, "Warp", fake_warp)
    
    with pytest.raises(RuntimeError) as exc:
        gdalwarp(
            src="source.tif",
            dst="destination.tif",
            dstSRS="EPSG:4326",
        )
    
    message = str(exc.value)
    assert "warp failed" in message
    assert "src: source.tif" in message
    assert "dst: destination.tif" in message
    assert "options:" in message


def test_gdalbuildvrt_returns_dataset_when_void_false(
        raster_file,
        tmp_path,
):
    destination = tmp_path / "result.vrt"
    
    dataset = gdalbuildvrt(
        src=str(raster_file),
        dst=str(destination),
        void=False,
    )
    
    try:
        assert isinstance(dataset, gdal.Dataset)
        assert dataset.RasterXSize == 4
        assert dataset.RasterYSize == 3
        assert destination.exists()
    finally:
        dataset = None


def test_gdalbuildvrt_returns_none_when_void_true(
        raster_file,
        tmp_path,
):
    destination = tmp_path / "result.vrt"
    
    result = gdalbuildvrt(
        src=str(raster_file),
        dst=str(destination),
        void=True,
    )
    
    assert result is None
    assert destination.exists()


def test_gdalbuildvrt_warns_for_old_gdal_with_output_bounds(
        monkeypatch,
):
    class FakeDataset:
        def FlushCache(self):
            pass
    
    monkeypatch.setattr(auxil.gdal, "__version__", "2.3.3")
    monkeypatch.setattr(
        auxil.gdal,
        "BuildVRTOptions",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        auxil.gdal,
        "BuildVRT",
        lambda dst, src, options: FakeDataset(),
    )
    
    with pytest.warns(UserWarning, match="subsetted extent"):
        result = gdalbuildvrt(
            src="source.tif",
            dst="result.vrt",
            outputBounds=[0, 0, 1, 1],
        )
    
    assert result is None


def test_gdal_translate_returns_dataset_when_void_false(
        raster_file,
        tmp_path,
):
    destination = tmp_path / "translated.tif"
    
    dataset = gdal_translate(
        src=str(raster_file),
        dst=str(destination),
        void=False,
        format="GTiff",
    )
    
    try:
        assert isinstance(dataset, gdal.Dataset)
        assert dataset.RasterXSize == 4
        assert dataset.RasterYSize == 3
        assert destination.exists()
    finally:
        dataset = None


def test_gdal_translate_returns_none_when_void_true(
        raster_file,
        tmp_path,
):
    destination = tmp_path / "translated.tif"
    
    result = gdal_translate(
        src=str(raster_file),
        dst=str(destination),
        void=True,
        format="GTiff",
    )
    
    assert result is None
    assert destination.exists()


def test_ogr2ogr_returns_dataset_when_void_false(vector_file):
    dataset = ogr2ogr(
        src=str(vector_file),
        dst="",
        void=False,
        format="MEM",
    )

    try:
        assert isinstance(dataset, gdal.Dataset)
        assert dataset.GetLayerCount() == 1
        assert dataset.GetLayer(0).GetFeatureCount() == 1
    finally:
        dataset = None


def test_ogr2ogr_returns_none_when_void_true(
        vector_file,
        tmp_path,
):
    destination = tmp_path / "translated.geojson"

    result = ogr2ogr(
        src=str(vector_file),
        dst=str(destination),
        void=True,
        format="GeoJSON",
    )

    assert result is None
    assert destination.exists()

    dataset = gdal.OpenEx(
        str(destination),
        gdal.OF_VECTOR,
    )
    try:
        assert dataset.GetLayerCount() == 1
        assert dataset.GetLayer(0).GetFeatureCount() == 1
    finally:
        dataset = None


def test_gdal_rasterize_creates_raster(vector_file, tmp_path):
    destination = tmp_path / "rasterized.tif"
    
    result = gdal_rasterize(
        src=str(vector_file),
        dst=str(destination),
        format="GTiff",
        outputBounds=[
            REGULAR_EXTENT["xmin"],
            REGULAR_EXTENT["ymin"],
            REGULAR_EXTENT["xmax"],
            REGULAR_EXTENT["ymax"],
        ],
        width=20,
        height=20,
        burnValues=[1],
        outputType=gdal.GDT_Byte,
    )
    
    assert result is None
    assert destination.exists()
    
    dataset = gdal.Open(str(destination))
    try:
        array = dataset.GetRasterBand(1).ReadAsArray()
        assert array.shape == (20, 20)
        assert np.all(array == 1)
    finally:
        dataset = None


# ---------------------------------------------------------------------------
# coordinate_reproject
# ---------------------------------------------------------------------------

def test_coordinate_reproject():
    point = coordinate_reproject(
        x=11,
        y=51,
        s_crs=4326,
        t_crs=32632,
    )
    
    assert point == pytest.approx(
        (640333.296, 5651728.683),
        abs=0.001,
    )


def test_coordinate_reproject_identity():
    point = coordinate_reproject(
        x=11,
        y=51,
        s_crs=4326,
        t_crs=4326,
    )
    
    assert point == pytest.approx((11, 51))


def test_coordinate_reproject_roundtrip():
    projected = coordinate_reproject(
        x=11,
        y=51,
        s_crs=4326,
        t_crs=32632,
    )
    geographic = coordinate_reproject(
        x=projected[0],
        y=projected[1],
        s_crs=32632,
        t_crs=4326,
    )
    
    assert geographic == pytest.approx((11, 51), abs=1e-8)


# ---------------------------------------------------------------------------
# utm_autodetect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "extent, expected_epsg",
    [
        (
                {"xmin": 11.5, "xmax": 11.7, "ymin": 50.8, "ymax": 51.0},
                32632,
        ),
        (
                {"xmin": 11.5, "xmax": 11.7, "ymin": -51.0, "ymax": -50.8},
                32732,
        ),
        (
                {"xmin": 178.0, "xmax": -178.0, "ymin": 50.0, "ymax": 51.0},
                32660,
        ),
        (
                {"xmin": 178.0, "xmax": -178.0, "ymin": -51.0, "ymax": -50.0},
                32760,
        ),
        (
                {"xmin": -124.0, "xmax": -123.0, "ymin": 45.0, "ymax": 46.0},
                32610,
        ),
        (
                {"xmin": 176.0, "xmax": 179.0, "ymin": 10.0, "ymax": 11.0},
                32660,
        ),
        (
                {"xmin": -179.0, "xmax": -176.0, "ymin": 10.0, "ymax": 11.0},
                32601,
        ),
    ],
    ids=[
        "north",
        "south",
        "antimeridian-north",
        "antimeridian-south",
        "western-hemisphere",
        "zone-60",
        "zone-1",
    ],
)
def test_utm_autodetect_epsg(extent, expected_epsg):
    with bbox(extent, crs=4326) as vector:
        result = utm_autodetect(vector, crsOut="epsg")
    
    assert result == expected_epsg


def test_utm_autodetect_output_osr():
    with bbox(REGULAR_EXTENT, crs=4326) as vector:
        result = utm_autodetect(vector, crsOut="osr")
    
    assert isinstance(result, osr.SpatialReference)
    assert crsConvert(result, "epsg") == 32632


def test_utm_autodetect_projected_input():
    with bbox(REGULAR_EXTENT, crs=4326) as vector:
        vector.reproject(32632)
        result = utm_autodetect(vector, crsOut="epsg")
    
    assert result == 32632


# ---------------------------------------------------------------------------
# cmap_mpl2gdal
# ---------------------------------------------------------------------------

def test_cmap_mpl2gdal_returns_color_table():
    cmap = cmap_mpl2gdal(
        mplcolor="YlGnBu",
        values=range(0, 100),
    )
    
    assert isinstance(cmap, gdal.ColorTable)


def test_cmap_mpl2gdal_sets_requested_entries():
    cmap = cmap_mpl2gdal(
        mplcolor="viridis",
        values=range(3),
    )
    
    assert cmap.GetCount() == 3
    for index in range(3):
        color = cmap.GetColorEntry(index)
        assert len(color) == 4
        assert all(0 <= value <= 255 for value in color)


def test_cmap_mpl2gdal_invalid_colormap():
    with pytest.raises(ValueError):
        cmap_mpl2gdal(
            mplcolor="not-a-colormap",
            values=range(3),
        )


# ---------------------------------------------------------------------------
# latlon_clamp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (-200, -180.0),
        (-180, -180.0),
        (0, 0.0),
        (180, 180.0),
        (200, 180.0),
    ],
)
def test_latlon_clamp_longitude(value, expected):
    assert latlon_clamp(lon=value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (-100, -90.0),
        (-90, -90.0),
        (0, 0.0),
        (90, 90.0),
        (100, 90.0),
    ],
)
def test_latlon_clamp_latitude(value, expected):
    assert latlon_clamp(lat=value) == expected


def test_latlon_clamp_rejects_both_coordinates():
    with pytest.raises(
            ValueError,
            match="only one of lat and lon can be specified",
    ):
        latlon_clamp(lat=0, lon=0)


def test_latlon_clamp_rejects_missing_coordinate():
    with pytest.raises(
            ValueError,
            match="either lat or lon must be specified",
    ):
        latlon_clamp()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lat": "50"},
        {"lon": "10"},
    ],
)
def test_latlon_clamp_rejects_non_numeric_values(kwargs):
    with pytest.raises(
            ValueError,
            match="lat and lon must be numeric or None",
    ):
        latlon_clamp(**kwargs)


# ---------------------------------------------------------------------------
# latlon_normalize
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (-540, 180.0),
        (-181, 179.0),
        (-180, 180.0),
        (-179, -179.0),
        (0, 0.0),
        (179, 179.0),
        (180, 180.0),
        (181, -179.0),
        (540, 180.0),
    ],
)
def test_latlon_normalize_longitude(value, expected):
    assert latlon_normalize(lon=value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (-270, 90.0),
        (-190, 10.0),
        (-100, -80.0),
        (-90, -90.0),
        (0, 0.0),
        (90, 90.0),
        (100, 80.0),
        (190, -10.0),
        (270, -90.0),
    ],
)
def test_latlon_normalize_latitude(value, expected):
    assert latlon_normalize(lat=value) == expected


def test_latlon_normalize_rejects_both_coordinates():
    with pytest.raises(
            ValueError,
            match="only one of lat and lon can be specified",
    ):
        latlon_normalize(lat=0, lon=0)


def test_latlon_normalize_rejects_missing_coordinate():
    with pytest.raises(
            ValueError,
            match="either lat or lon must be specified",
    ):
        latlon_normalize()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lat": "50"},
        {"lon": "10"},
    ],
)
def test_latlon_normalize_rejects_non_numeric_values(kwargs):
    with pytest.raises(
            ValueError,
            match="lat and lon must be numeric or None",
    ):
        latlon_normalize(**kwargs)


# ---------------------------------------------------------------------------
# latlon_extent_center
# ---------------------------------------------------------------------------

def test_latlon_extent_center_regular():
    extent = {
        "xmin": 10,
        "xmax": 14,
        "ymin": 50,
        "ymax": 54,
    }
    
    assert latlon_extent_center(extent) == (12.0, 52.0)


def test_latlon_extent_center_antimeridian():
    extent = {
        "xmin": 170,
        "xmax": -160,
        "ymin": 50,
        "ymax": 54,
    }
    
    assert latlon_extent_center(extent) == (-175.0, 52.0)


def test_latlon_extent_center_antimeridian_center_at_180():
    extent = {
        "xmin": 170,
        "xmax": -170,
        "ymin": -10,
        "ymax": 10,
    }
    
    assert latlon_extent_center(extent) == (180.0, 0.0)


# ---------------------------------------------------------------------------
# longitude_shortest_interval
# ---------------------------------------------------------------------------

def test_longitude_shortest_interval_empty():
    with pytest.raises(ValueError, match="longitudes must be non-empty"):
        longitude_shortest_interval([])


def test_longitude_shortest_interval_single_value():
    assert longitude_shortest_interval([190]) == (-170.0, -170.0)


def test_longitude_shortest_interval_regular():
    assert longitude_shortest_interval([-10, 0, 10]) == (-10.0, 10.0)


def test_longitude_shortest_interval_antimeridian():
    assert longitude_shortest_interval([170, 175, -170]) == (
        170.0,
        -170.0,
    )


def test_longitude_shortest_interval_normalizes_values():
    assert longitude_shortest_interval([350, 10]) == (-10.0, 10.0)


def test_longitude_shortest_interval_duplicate_values():
    assert longitude_shortest_interval([10, 10, 10]) == (10.0, 10.0)


def test_longitude_shortest_interval_tie_is_deterministic():
    assert longitude_shortest_interval([0, 180]) == (180.0, 0.0)


# ---------------------------------------------------------------------------
# iter_geometries
# ---------------------------------------------------------------------------

def test_iter_geometries_none():
    assert list(iter_geometries(None)) == []


def test_iter_geometries_empty():
    geometry = ogr.CreateGeometryFromWkt("POINT EMPTY")
    
    assert list(iter_geometries(geometry)) == []


def test_iter_geometries_simple_geometry():
    geometry = ogr.CreateGeometryFromWkt("LINESTRING (0 0, 1 1)")
    
    parts = list(iter_geometries(geometry))
    
    assert len(parts) == 1
    assert parts[0].GetGeometryName() == "LINESTRING"


def test_iter_geometries_treats_polygon_as_atomic():
    geometry = ogr.CreateGeometryFromWkt(
        "POLYGON ("
        "(0 0, 0 3, 3 3, 3 0, 0 0), "
        "(1 1, 2 1, 2 2, 1 2, 1 1)"
        ")"
    )
    
    parts = list(iter_geometries(geometry))
    
    assert len(parts) == 1
    assert parts[0].GetGeometryName() == "POLYGON"


def test_iter_geometries_unpacks_multi_geometry():
    geometry = ogr.CreateGeometryFromWkt(
        "MULTIPOLYGON ("
        "((0 0, 0 1, 1 1, 1 0, 0 0)), "
        "((2 0, 2 1, 3 1, 3 0, 2 0))"
        ")"
    )
    
    parts = list(iter_geometries(geometry))
    
    assert [part.GetGeometryName() for part in parts] == [
        "POLYGON",
        "POLYGON",
    ]


def test_iter_geometries_recursively_unpacks_collection():
    geometry = ogr.CreateGeometryFromWkt(
        "GEOMETRYCOLLECTION ("
        "POINT (0 0), "
        "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))"
        ")"
    )
    
    parts = list(iter_geometries(geometry))
    
    assert [part.GetGeometryName() for part in parts] == [
        "POINT",
        "LINESTRING",
        "LINESTRING",
    ]


# ---------------------------------------------------------------------------
# iter_points
# ---------------------------------------------------------------------------

def test_iter_points_none():
    assert list(iter_points(None)) == []


def test_iter_points_empty():
    geometry = ogr.CreateGeometryFromWkt("POINT EMPTY")
    
    assert list(iter_points(geometry)) == []


def test_iter_points_linestring():
    geometry = ogr.CreateGeometryFromWkt(
        "LINESTRING (0 0, 1 2, 3 4)"
    )
    
    points = list(iter_points(geometry))
    
    assert [point[:2] for point in points] == [
        (0.0, 0.0),
        (1.0, 2.0),
        (3.0, 4.0),
    ]


def test_iter_points_polygon_includes_ring_vertices():
    geometry = ogr.CreateGeometryFromWkt(
        "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"
    )
    
    points = list(iter_points(geometry))
    
    assert [point[:2] for point in points] == [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 0.0),
    ]


def test_iter_points_recursively_unpacks_collection():
    geometry = ogr.CreateGeometryFromWkt(
        "GEOMETRYCOLLECTION ("
        "POINT (5 6), "
        "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))"
        ")"
    )
    
    points = list(iter_points(geometry))
    
    assert [point[:2] for point in points] == [
        (5.0, 6.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
    ]


def test_iter_points_preserves_z_coordinate():
    geometry = ogr.CreateGeometryFromWkt("POINT Z (1 2 3)")
    
    point = list(iter_points(geometry))[0]
    
    assert point[:3] == (1.0, 2.0, 3.0)
