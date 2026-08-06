import pytest
import os
from osgeo import gdal
from spatialist.auxil import (crsConvert, haversine, ogr2ogr, gdal_translate, gdal_rasterize,
                              utm_autodetect, coordinate_reproject, cmap_mpl2gdal)
from spatialist.vector import bbox
from spatialist.raster import Raster

def test_auxil_cmap_mpl2gdal():
    cmap = cmap_mpl2gdal(mplcolor='YlGnBu', values=range(0, 100))
    assert type(cmap) == gdal.ColorTable

def test_coordinate_reproject():
    point = coordinate_reproject(x=11, y=51, s_crs=4326, t_crs=32632)
    assert round(point[0], 3) == 640333.296
    assert round(point[1], 3) == 5651728.683

def test_crsConvert():
    assert crsConvert(crsConvert(4326, 'wkt'), 'proj4').strip() == '+proj=longlat +datum=WGS84 +no_defs'
    assert crsConvert(crsConvert(4326, 'prettyWkt'), 'opengis') == 'https://www.opengis.net/def/crs/EPSG/0/4326'
    assert crsConvert('https://www.opengis.net/def/crs/EPSG/0/4326', 'epsg') == 4326
    assert crsConvert(crsConvert('https://www.opengis.net/def/crs/EPSG/0/4326', 'osr'), 'epsg') == 4326
    assert crsConvert('EPSG:4326+5773', 'proj4').strip() \
           == '+proj=longlat +datum=WGS84 +geoidgrids=egm96_15.gtx +vunits=m +no_defs' \
           or '+proj=longlat +datum=WGS84 +vunits=m +no_defs'
    with pytest.raises(TypeError):
        crsConvert('xyz', 'epsg')
    with pytest.raises(ValueError):
        crsConvert(4326, 'xyz')

def test_haversine():
    assert haversine(50, 10, 51, 10) == 111194.92664455889

def test_translate_rasterize(tmpdir, testdata):
    dir = str(tmpdir)
    with Raster(testdata['tif']) as ras:
        bbox = os.path.join(dir, 'bbox.shp')
        ras.bbox(bbox)
        ogr2ogr(src=bbox, dst=os.path.join(dir, 'bbox.gml'), format='GML')
        gdal_translate(src=ras.raster, dst=os.path.join(dir, 'test'), format='ENVI')
    gdal_rasterize(src=bbox, dst=os.path.join(dir, 'test2'), format='GTiff', xRes=20, yRes=20)

@pytest.mark.parametrize(
    "extent, expected_epsg",
    [
        # normal Northern Hemisphere case, Germany -> UTM zone 32N
        (
                {"xmin": 11.5, "xmax": 11.7, "ymin": 50.8, "ymax": 51.0},
                32632,
        ),
        
        # normal Southern Hemisphere case -> UTM zone 32S
        (
                {"xmin": 11.5, "xmax": 11.7, "ymin": -51.0, "ymax": -50.8},
                32732,
        ),
        
        # antimeridian crossing, Northern Hemisphere -> UTM zone 60N
        (
                {"xmin": 178.0, "xmax": -178.0, "ymin": 50.0, "ymax": 51.0},
                32660,
        ),
        
        # antimeridian crossing, Southern Hemisphere -> UTM zone 60S
        (
                {"xmin": 178.0, "xmax": -178.0, "ymin": -51.0, "ymax": -50.0},
                32760,
        ),
        
        # western hemisphere case -> UTM zone 10N
        (
                {"xmin": -124.0, "xmax": -123.0, "ymin": 45.0, "ymax": 46.0},
                32610,
        ),
        
        # near 180 but not crossing -> still UTM zone 60N
        (
                {"xmin": 176.0, "xmax": 179.0, "ymin": 10.0, "ymax": 11.0},
                32660,
        ),
        
        # just east of -180, not crossing -> UTM zone 1N
        (
                {"xmin": -179.0, "xmax": -176.0, "ymin": 10.0, "ymax": 11.0},
                32601,
        ),
    ],
)
def test_utm_autodetect(extent, expected_epsg):
    with bbox(extent, crs=4326) as vec:
        assert utm_autodetect(vec, crsOut="epsg") == expected_epsg
