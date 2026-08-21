import os
import pytest
import platform
import numpy as np
from datetime import datetime, timezone
from osgeo import ogr, osr, gdal
from spatialist.raster import Raster
from spatialist.vector import (feature2vector, dissolve, Vector, intersect,
                               bbox, wkt2vector, set_field, vectorize,
                               outer_hull, combine_polygons)
from spatialist.envi import hdr, HDRobject
from spatialist.sqlite_util import sqlite_setup, __Handler
from spatialist.auxil import utm_autodetect

import logging

logging.basicConfig(level=logging.DEBUG)


def test_Vector(tmpdir, testdata):
    scene = Raster(testdata['tif'])
    bbox1 = scene.bbox()
    assert bbox1.getArea() == 23262400.0
    assert bbox1.extent == {'ymax': 4830114.70107, 'ymin': 4825774.70107,
                            'xmin': 620048.241204, 'xmax': 625408.241204}
    assert bbox1.nlayers == 1
    assert bbox1.getProjection('epsg') == 32631
    assert bbox1.proj4.strip() == '+proj=utm +zone=31 +datum=WGS84 +units=m +no_defs'
    assert isinstance(bbox1.getFeatureByIndex(0), ogr.Feature)
    with pytest.raises(IndexError):
        bbox1.getFeatureByIndex(1)
    bbox1.reproject(4326)
    assert bbox1.proj4.strip() == '+proj=longlat +datum=WGS84 +no_defs'
    ext = {key: round(val, 3) for key, val in bbox1.extent.items()}
    assert ext == {'xmax': 4.554, 'xmin': 4.487, 'ymax': 43.614, 'ymin': 43.574}
    assert utm_autodetect(bbox1, 'epsg') == 32631
    assert isinstance(bbox1['fid=0'], Vector)
    with pytest.raises(RuntimeError):
        test = bbox1[0.1]
    assert bbox1.fieldnames == ['area']
    assert bbox1.getUniqueAttributes('area') == [23262400.0]
    feat = bbox1.getFeatureByAttribute('area', 23262400.0)
    assert isinstance(feat, ogr.Feature)
    bbox2 = feature2vector(feat, ref=bbox1)
    bbox2.close()
    feat.Destroy()
    with pytest.raises(KeyError):
        select = bbox1.getFeatureByAttribute('foo', 'bar')
    with pytest.raises(OSError):
        vec = Vector(filename='foobar')
    bbox1.close()
    with scene.bbox() as bbox3:
        bbox3.addfield(name='datetime', type=ogr.OFTDateTime,
                       values=[datetime.now()])
        gdf_out = tmpdir / "test.gpkg"
        gdf = bbox3.to_geopandas()
        gdf.to_file(str(gdf_out))
        assert gdf_out.exists()
    coords = {'xmin': 10, 'ymin': 20, 'xmax': 50, 'ymax': 51}
    with bbox(coordinates=coords, crs=4326, buffer=None) as bbox4:
        assert bbox4.getArea() == 1240.0
    with bbox(coordinates=coords, crs=4326, buffer=1) as bbox4:
        assert bbox4.getArea() == 1386.0
    with bbox(coordinates=coords, crs=4326, buffer=(1, 2)) as bbox4:
        assert bbox4.getArea() == 1470.0


@pytest.mark.parametrize(
    "extent, crs, geometry_type",
    [
        (
                {'xmin': 10, 'ymin': 11, 'xmax': 50, 'ymax': 51},
                4326,
                'Polygon'
        ),
        (
                {'xmin': 179, 'ymin': -179, 'xmax': 50, 'ymax': 51},
                4326,
                'MultiPolygon'
        ),
        (
                {'xmin': 600000, 'xmax': 709800, 'ymin': 5790240, 'ymax': 5900040},
                32632,
                'Polygon'
        ),
        (
                {'xmin': 600000, 'xmax': 709800, 'ymin': 5790240, 'ymax': 5900040},
                32660,
                'MultiPolygon'
        )
    ],
    ids=['regular', 'antimeridian', 'regular_utm', 'antimeridian_utm']
)
def test_vector_geo_interface(extent, crs, geometry_type):
    with bbox(coordinates=extent, crs=crs) as box:
        geom = box.__geo_interface__
    print(geom['features'][0]['geometry'])
    assert geom['type'] == 'FeatureCollection'
    assert len(geom['features']) == 1
    assert geom['features'][0]['type'] == 'Feature'
    assert geom['features'][0]['geometry']['type'] == geometry_type


def test_dissolve(tmpdir, travis, testdata):
    scene = Raster(testdata['tif'])
    bbox1 = scene.bbox()
    # retrieve extent and shift its coordinates by one unit
    ext = bbox1.extent
    for key in ext.keys():
        ext[key] += 1
    # create new bbox shapefile with modified extent
    bbox2_name = os.path.join(str(tmpdir), 'bbox2.shp')
    bbox(ext, bbox1.srs, bbox2_name)
    # assert intersection between the two bboxes and combine them into one
    with Vector(bbox2_name) as bbox2:
        assert intersect(bbox1, bbox2) is not None
        bbox1.addvector(bbox2)
        # write combined bbox into new shapefile
        bbox3_name = os.path.join(str(tmpdir), 'bbox3.shp')
        bbox1.write(bbox3_name)
    bbox1.close()
    
    if not travis and platform.system() != 'Windows':
        # dissolve the geometries in bbox3 and write the result to new bbox4
        # this test is currently disabled for Travis as the current sqlite3 version on Travis seems to not support
        # loading gdal as extension; Travis CI setup: Ubuntu 14.04 (Trusty), sqlite3 version 3.8.2 (2018-06-04)
        bbox4_name = os.path.join(str(tmpdir), 'bbox4.shp')
        dissolve(bbox3_name, bbox4_name, field='area')
        assert os.path.isfile(bbox4_name)


def test_envi(tmpdir):
    with pytest.raises(RuntimeError):
        obj = HDRobject(1)
    with pytest.raises(RuntimeError):
        obj = HDRobject('foobar')
    outname = os.path.join(str(tmpdir), 'test')
    with HDRobject() as header:
        header.band_names = ['band1']
        header.write(outname)
    outname += '.hdr'
    with HDRobject(outname) as header:
        assert header.band_names == ['band1']
        vals = vars(header)
    with HDRobject(vals) as header:
        assert header.byte_order == 0
    hdr(vals, outname + '2')


def test_sqlite():
    with pytest.raises(RuntimeError):
        con = sqlite_setup(extensions='spatialite')
    con = sqlite_setup(extensions=['spatialite'])
    con.close()
    con = __Handler()
    assert sorted(con.version.keys()) == ['sqlite']
    
    con = __Handler(extensions=['spatialite'])
    assert sorted(con.version.keys()) == ['spatialite', 'sqlite']
    assert 'spatial_ref_sys' in con.get_tablenames()


def test_addfield():
    extent = {'xmin': 10, 'xmax': 11, 'ymin': 50, 'ymax': 51}
    with bbox(coordinates=extent, crs=4326) as box:
        box.addfield(name='test1', type=ogr.OFTString, values=['a'])
        box.addfield(name='test2', type=ogr.OFTStringList, values=[['a', 'b']])
        box.addfield(name='test3', type=ogr.OFTInteger, values=[1])
        box.addfield(name='test4', type=ogr.OFTIntegerList, values=[[1, 2]])
        box.addfield(name='test5', type=ogr.OFTInteger64, values=[1])
        box.addfield(name='test6', type=ogr.OFTInteger64List, values=[[1, 2]])
        box.addfield(name='test7', type=ogr.OFTReal, values=[1])
        box.addfield(name='test8', type=ogr.OFTRealList, values=[[1., 2.]])
        box.addfield(name='test9', type=ogr.OFTBinary, values=[b'1'])
        now = datetime.now()  # timezone unaware
        box.addfield(name='test10', type=ogr.OFTDateTime, values=[now])
        now = now.astimezone()  # local timezone
        box.addfield(name='test11', type=ogr.OFTDateTime, values=[now])
        now = now.astimezone(timezone.utc)  # UTC timezone
        box.addfield(name='test12', type=ogr.OFTDateTime, values=[now])
        with pytest.raises(ValueError):
            # Date type is not supported
            box.addfield(name='test13', type=ogr.OFTDate, values=[now])
        with pytest.raises(ValueError):
            # Time type is not supported
            box.addfield(name='test14', type=ogr.OFTTime, values=[now])
        with pytest.raises(TypeError):
            # value must be a datetime object
            box.addfield(name='test15', type=ogr.OFTDateTime, values=[1])
        with pytest.raises(RuntimeError):
            # one feature, two values
            box.addfield(name='test16', type=ogr.OFTString, values=['a', 'b'])
        with pytest.raises(TypeError):
            # target must be Vector or ogr.Feature
            set_field(target='x', name='test17', type=ogr.OFTString, values=['a'])


def test_wkt2vector():
    wkt1 = 'POLYGON ((0. 0., 0. 1., 1. 1., 1. 0., 0. 0.))'
    wkt2 = 'POLYGON ((1. 1., 1. 2., 2. 2., 2. 1., 1. 1.))'
    with wkt2vector(wkt1, srs=4326) as vec:
        assert vec.getArea() == 1.
    with wkt2vector([wkt1, wkt2], srs=4326) as vec:
        assert vec.getArea() == 2.


def test_bbox_antimeridian():
    crs = 4326
    extent = {'xmin': 178, 'xmax': -178, 'ymin': 50, 'ymax': 51}
    
    # 4326 crossing the antimeridian, not wrapped
    with bbox(coordinates=extent, crs=crs, split_antimeridian=False) as vec:
        assert vec.geomType == ogr.wkbPolygon
        assert vec.get_extent() == {'xmin': -178, 'xmax': 178, 'ymin': 50, 'ymax': 51}
        
        # wrap separately
        vec.wrap_antimeridian()
        assert vec.geomType == ogr.wkbMultiPolygon
        assert vec.get_extent() == {'xmin': 178, 'xmax': -178, 'ymin': 50, 'ymax': 51}
    
    # 4326 crossing the antimeridian, not wrapped, buffered
    with bbox(coordinates=extent, crs=crs, split_antimeridian=False, buffer=3) as vec:
        assert vec.geomType == ogr.wkbPolygon
        assert vec.get_extent() == {'xmin': -180., 'xmax': 180., 'ymin': 47., 'ymax': 54.}
        
        # wrap separately: the polygon is now truly world-spanning,
        # and wrapping thus does not have any effect
        vec.wrap_antimeridian()
        assert vec.geomType == ogr.wkbPolygon
        assert vec.get_extent() == {'xmin': -180., 'xmax': 180., 'ymin': 47., 'ymax': 54.}
    
    # 4326 crossing the antimeridian, wrapped
    with bbox(coordinates=extent, crs=crs, split_antimeridian=True) as vec:
        assert vec.geomType == ogr.wkbMultiPolygon
        assert vec.get_extent(split_antimeridian=False) == {'xmin': -180, 'xmax': 180, 'ymin': 50, 'ymax': 51}
        assert vec.get_extent(split_antimeridian=True) == extent
    
    # 4326 crossing the antimeridian, wrapped, buffered
    with bbox(coordinates=extent, crs=crs, split_antimeridian=True, buffer=3) as vec:
        assert vec.geomType == ogr.wkbMultiPolygon
        assert vec.get_extent(split_antimeridian=False) == {'xmin': -180, 'xmax': 180, 'ymin': 47, 'ymax': 54}
        assert vec.get_extent(split_antimeridian=True) == {'xmin': 175, 'xmax': -175, 'ymin': 47, 'ymax': 54}
    
    # UTM not crossing the antimeridian
    crs = 32632
    extent_utm = {'xmin': 600000, 'xmax': 709800, 'ymin': 5790240, 'ymax': 5900040}
    
    with bbox(coordinates=extent_utm, crs=crs) as vec:
        vec.reproject(4326)
        assert vec.geomType == ogr.wkbPolygon
        assert vec.geomTypes == ['POLYGON']
        extent_4326 = vec.get_extent(split_antimeridian=True)
        expected = {'xmin': 10.5, 'xmax': 12.1, 'ymin': 52.2, 'ymax': 53.2}
        assert extent_4326 == pytest.approx(expected, rel=1e-1)
    
    # UTM crossing the antimeridian, wrapped
    crs = 32660
    
    with bbox(coordinates=extent_utm, crs=crs) as vec:
        vec.reproject(4326)
        assert vec.geomType == ogr.wkbMultiPolygon
        assert vec.geomTypes == ['MULTIPOLYGON']
        extent_4326 = vec.get_extent(split_antimeridian=True)
        expected = {'xmin': 178.5, 'xmax': -179.9, 'ymin': 52.2, 'ymax': 53.2}
        assert extent_4326 == pytest.approx(expected, rel=1e-1)


def test_outer_hull():
    array = np.array(
        [
            [0, 0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create(
        "",
        array.shape[1],
        array.shape[0],
        1,
        gdal.GDT_Byte,
    )
    
    dataset.SetGeoTransform((0, 1, 0, 6, 0, -1))
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dataset.SetProjection(srs.ExportToWkt())
    
    dataset.GetRasterBand(1).WriteArray(array)
    
    with Raster(dataset) as raster:
        with vectorize(array, raster) as vec:
            with vec.filter(expression="value = 1") as filt:
                with outer_hull(filt) as result:
                    assert result.nfeatures == 1
                    assert result.geomTypes == ["MULTIPOLYGON"]
                    assert result.getArea() == 13
                    assert result.extent == {
                        "xmin": 1.0,
                        "xmax": 6.0,
                        "ymin": 1.0,
                        "ymax": 6.0,
                    }
                    
                    feature = result.getFeatureByIndex(0)
                    geometry = feature.GetGeometryRef()
                    
                    assert geometry.GetGeometryCount() == 2
    
    dataset = None
    driver = None


@pytest.mark.parametrize(
    'input_extents, expected_extent',
    [
        (
                (
                        {'xmin': 10, 'xmax': 11, 'ymin': 50, 'ymax': 51},
                        {'xmin': 11, 'xmax': 12, 'ymin': 50, 'ymax': 51},
                ),
                {'xmin': 10, 'xmax': 12, 'ymin': 50, 'ymax': 51},
        ),
    ],
)
def test_combine_polygons_preserves_polygons(
        input_extents,
        expected_extent,
):
    vectors = [bbox(extent, 4326) for extent in input_extents]
    
    try:
        with combine_polygons(vectors) as combined:
            assert combined.extent == expected_extent
            assert combined.nfeatures == 2
            assert combined.geomType == ogr.wkbPolygon
    finally:
        for vector in vectors:
            vector.close()


def test_combine_polygons_multipolygon_roundtrip():
    ext1 = {'xmin': 10, 'xmax': 11, 'ymin': 50, 'ymax': 51}
    ext2 = {'xmin': 11, 'xmax': 12, 'ymin': 50, 'ymax': 51}
    ext3 = {'xmin': 21, 'xmax': 22, 'ymin': 50, 'ymax': 51}
    
    with bbox(ext1, 4326) as vec1:
        with bbox(ext2, 4326) as vec2:
            # Two Polygon -> one MultiPolygon.
            with combine_polygons(
                    [vec1, vec2],
                    multipolygon=True,
            ) as multipolygon:
                assert multipolygon.nfeatures == 1
                assert multipolygon.geomType == ogr.wkbMultiPolygon
                
                # One MultiPolygon -> two Polygon.
                with combine_polygons(
                        multipolygon,
                        explode=True,
                ) as exploded:
                    assert exploded.nfeatures == 2
                    assert exploded.geomType == ogr.wkbPolygon
                
                # One MultiPolygon -> one MultiPolygon.
                with combine_polygons(
                        multipolygon,
                        multipolygon=True,
                ) as multipolygon_again:
                    assert multipolygon_again.nfeatures == 1
                    assert multipolygon_again.geomType == ogr.wkbMultiPolygon
                
                # MultiPolygon + Polygon -> three Polygon.
                with bbox(ext3, 4326) as vec3:
                    with combine_polygons(
                            [multipolygon, vec3],
                            explode=True,
                    ) as combined:
                        assert combined.nfeatures == 3
                        assert combined.geomType == ogr.wkbPolygon


@pytest.mark.parametrize(
    'explode, multipolygon, expected_features, expected_geom_type',
    [
        (True, False, 2, ogr.wkbPolygon),
        (False, True, 1, ogr.wkbMultiPolygon),
    ],
    ids=[
        'split-into-polygons',
        'retain-as-multipolygon',
    ],
)
def test_combine_polygons_antimeridian(
        explode,
        multipolygon,
        expected_features,
        expected_geom_type,
):
    antimeridian_extent = {
        'xmin': 179,
        'xmax': -179,
        'ymin': 50,
        'ymax': 51,
    }
    
    with bbox(antimeridian_extent, 4326) as vector:
        with combine_polygons(
                vector,
                explode=explode,
                multipolygon=multipolygon,
        ) as combined:
            assert combined.nfeatures == expected_features
            assert combined.geomType == expected_geom_type
            assert combined.getArea() == pytest.approx(2.0)


def test_combine_polygons_mixed_antimeridian():
    ordinary_extent = {
        'xmin': 10,
        'xmax': 11,
        'ymin': 50,
        'ymax': 51,
    }
    antimeridian_extent = {
        'xmin': 179,
        'xmax': -179,
        'ymin': 50,
        'ymax': 51,
    }
    
    with bbox(ordinary_extent, 4326) as ordinary:
        with bbox(antimeridian_extent, 4326) as antimeridian:
            # One ordinary polygon plus two polygons created by splitting
            # the antimeridian-crossing polygon.
            with combine_polygons(
                    [ordinary, antimeridian],
                    explode=True,
            ) as combined:
                assert combined.nfeatures == 3
                assert combined.geomType == ogr.wkbPolygon
                assert combined.getArea() == pytest.approx(3.0)
