import platform
from datetime import datetime, timedelta, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, ogr, osr
from shapely.geometry import Point, Polygon

from spatialist.raster import Raster
from spatialist.vector import (
    Vector,
    bbox,
    combine_polygons,
    dissolve,
    feature2vector,
    from_geopandas,
    hull,
    intersect,
    set_field,
    vectorize,
    wkt2vector,
)

REGULAR_EXTENT = {
    'xmin': 10,
    'xmax': 11,
    'ymin': 50,
    'ymax': 51,
}
ANTIMERIDIAN_EXTENT = {
    'xmin': 179,
    'xmax': -179,
    'ymin': 50,
    'ymax': 51,
}
UTM_EXTENT = {
    'xmin': 600000,
    'xmax': 709800,
    'ymin': 5790240,
    'ymax': 5900040,
}


def _assert_polygon_orientation(geometry):
    geom_type = ogr.GT_Flatten(geometry.GetGeometryType())
    
    if geom_type == ogr.wkbPolygon:
        exterior = geometry.GetGeometryRef(0)
        assert exterior is not None
        assert not exterior.IsClockwise()
        
        for i in range(1, geometry.GetGeometryCount()):
            interior = geometry.GetGeometryRef(i)
            assert interior.IsClockwise()
    
    elif geom_type == ogr.wkbMultiPolygon:
        for i in range(geometry.GetGeometryCount()):
            _assert_polygon_orientation(
                geometry.GetGeometryRef(i)
            )
    
    else:
        raise AssertionError(
            f'expected polygonal geometry, got {geometry.GetGeometryName()}'
        )


def _vector_from_wkts(
        wkts,
        srs=4326,
        geom_type=None,
        layer_name='layer',
):
    """Create a small in-memory Vector for unit tests."""
    if isinstance(wkts, str):
        wkts = [wkts]
    
    geometries = [ogr.CreateGeometryFromWkt(item) for item in wkts]
    if geom_type is None:
        geom_type = geometries[0].GetGeometryType()
    
    vector = Vector()
    vector.addlayer(layer_name, srs, geom_type)
    for geometry in geometries:
        vector.addfeature(geometry)
    return vector


def _memory_raster(
        array,
        epsg=4326,
        geotransform=None,
):
    """Create a georeferenced in-memory Raster for vectorize tests."""
    driver = gdal.GetDriverByName('MEM')
    dataset = driver.Create(
        '',
        array.shape[1],
        array.shape[0],
        1,
        gdal.GDT_Byte,
    )
    
    if geotransform is None:
        geotransform = (
            0,
            1,
            0,
            array.shape[0],
            0,
            -1,
        )
    
    dataset.SetGeoTransform(geotransform)
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(array)
    
    return Raster(dataset)


# -----------------------------------------------------------------------------
# Vector construction, lifecycle and metadata
# -----------------------------------------------------------------------------


def test_vector_memory_constructor_has_no_layer():
    vector = Vector()
    try:
        assert vector.filename is None
        assert vector.nlayers == 0
    finally:
        # Vector.close() currently assumes that a layer has already been created.
        vector.vector = None


def test_vector_constructor_rejects_invalid_filename_type():
    with pytest.raises(TypeError, match='filename must either be str or None'):
        Vector(filename=1)


def test_vector_constructor_rejects_missing_file(tmp_path):
    with pytest.raises(OSError, match='file does not exist'):
        Vector(filename=str(tmp_path / 'missing.shp'))


def test_vector_driver_autodetect_rejects_unknown_extension(tmp_path):
    filename = tmp_path / 'vector.unsupported'
    filename.touch()
    
    with pytest.raises(RuntimeError, match='file extension'):
        Vector(filename=str(filename))


def test_vector_context_manager_closes_datasource():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        assert vector.__enter__() is vector
    assert vector.vector is None


def test_vector_layer_metadata():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        assert vector.nlayers == 1
        assert vector.nfeatures == 1
        assert vector.nfields == 1
        assert vector.layername == 'bbox'
        assert vector.geomType == ogr.wkbPolygon
        assert vector.geomTypes == ['POLYGON']
        assert vector.fieldnames == ['area']
        assert len(vector.fieldDefs) == 1
        assert vector.layerdef.GetFieldCount() == 1
        assert vector.srs.IsGeographic()


def test_vector_string_representation():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        text = str(vector)
    
    assert 'spatialist Vector object' in text
    assert 'geometry type : POLYGON' in text
    assert 'extent        : 10.000, 11.000, 50.000, 51.000' in text
    assert 'data source   : memory' in text


def test_vector_projection_properties():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        assert vector.getProjection('epsg') == 4326
        assert '+proj=longlat' in vector.proj4


def test_vector_getfeatures_returns_clones():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        feature = vector.getfeatures()[0]
        feature.SetField('area', 999)
        
        assert vector.getfeatures()[0].GetField('area') == 1.0


def test_vector_load_populates_feature_cache():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        assert vector._Vector__features == [None]
        vector.load()
        assert isinstance(vector._Vector__features[0], ogr.Feature)


def test_vector_init_features_resets_feature_cache():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.load()
        vector.init_features()
        assert vector._Vector__features == [None]


def test_vector_init_layer_refreshes_layer_reference():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.layer = None
        vector.init_layer()
        assert vector.layer.GetName() == 'bbox'
        assert vector.nfeatures == 1


# -----------------------------------------------------------------------------
# Feature access, selection and filtering
# -----------------------------------------------------------------------------


def test_vector_get_feature_by_index():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        feature = vector.getFeatureByIndex(0)
        assert isinstance(feature, ogr.Feature)
        assert feature.GetField('area') == 1.0


def test_vector_get_feature_by_index_out_of_range():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with pytest.raises(IndexError):
            vector.getFeatureByIndex(1)


def test_vector_get_feature_by_attribute_returns_single_feature():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        feature = vector.getFeatureByAttribute('area', 1.0)
        assert isinstance(feature, ogr.Feature)


def test_vector_get_feature_by_attribute_returns_multiple_features():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        second = bbox(
            {'xmin': 11, 'xmax': 12, 'ymin': 50, 'ymax': 51},
            4326,
        )
        try:
            vector.addvector(second)
            features = vector.getFeatureByAttribute('area', 1.0)
            assert isinstance(features, list)
            assert len(features) == 2
        finally:
            second.close()


def test_vector_get_feature_by_attribute_strips_strings():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.addfield('name', ogr.OFTString, values=['  example  '])
        feature = vector.getFeatureByAttribute('name', ' example ')
        assert isinstance(feature, ogr.Feature)


def test_vector_get_feature_by_attribute_returns_none_for_no_match():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        assert vector.getFeatureByAttribute('area', 999) is None


def test_vector_get_feature_by_attribute_rejects_invalid_field():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with pytest.raises(KeyError, match='invalid field name'):
            vector.getFeatureByAttribute('missing', 1)


def test_vector_getitem_by_index_returns_vector():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with vector[0] as selected:
            assert isinstance(selected, Vector)
            assert selected.nfeatures == 1
            assert selected.extent == REGULAR_EXTENT


def test_vector_getitem_parses_numeric_string_as_index():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with vector['0'] as selected:
            assert isinstance(selected, Vector)
            assert selected.nfeatures == 1


def test_vector_getitem_by_attribute_filter():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with vector['area = 1'] as selected:
            assert isinstance(selected, Vector)
            assert selected.nfeatures == 1


def test_vector_getitem_returns_none_for_empty_filter():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        assert vector['area = 999'] is None


def test_vector_getitem_rejects_invalid_expression_type():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with pytest.raises(RuntimeError, match='expression must be of type int or str'):
            vector[0.1]


def test_vector_filter_returns_matching_features_only():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        other = bbox(
            {'xmin': 12, 'xmax': 14, 'ymin': 50, 'ymax': 51},
            4326,
        )
        try:
            vector.addvector(other)
            with vector.filter('area = 2') as filtered:
                assert filtered.nfeatures == 1
                assert filtered.getArea() == 2.0
                assert filtered.fieldnames == ['area']
        finally:
            other.close()


def test_vector_get_unique_attributes():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        other = bbox(
            {'xmin': 12, 'xmax': 14, 'ymin': 50, 'ymax': 51},
            4326,
        )
        try:
            vector.addvector(other)
            assert vector.getUniqueAttributes('area') == [1.0, 2.0]
        finally:
            other.close()


# -----------------------------------------------------------------------------
# Layer, field and feature creation
# -----------------------------------------------------------------------------


def test_vector_addlayer():
    with Vector() as vector:
        vector.addlayer('test', 4326, ogr.wkbPoint)
        
        assert vector.layername == 'test'
        assert vector.geomType == ogr.wkbPoint
        assert vector.getProjection('epsg') == 4326


def test_vector_addfeature():
    with Vector() as vector:
        vector.addlayer('test', 4326, ogr.wkbPoint)
        vector.addfield('name', ogr.OFTString)
        vector.addfeature(
            ogr.CreateGeometryFromWkt('POINT (1 2)'),
            fields={'name': 'one'},
        )
        
        feature = vector.getFeatureByIndex(0)
        assert vector.nfeatures == 1
        assert feature.GetField('name') == 'one'
        assert feature.GetGeometryRef().ExportToWkt() == 'POINT (1 2)'


def test_vector_addfeature_rejects_missing_field():
    with Vector() as vector:
        vector.addlayer('test', 4326, ogr.wkbPoint)
        with pytest.raises(IOError, match='field "missing" is missing'):
            vector.addfeature(
                ogr.CreateGeometryFromWkt('POINT (1 2)'),
                fields={'missing': 'value'},
            )


def test_vector_addfeature_wraps_field_conversion_error():
    with Vector() as vector:
        vector.addlayer('test', 4326, ogr.wkbPoint)
        vector.addfield('when', ogr.OFTDateTime)
        
        with pytest.raises(RuntimeError, match='trying to set field when'):
            vector.addfeature(
                ogr.CreateGeometryFromWkt('POINT (1 2)'),
                fields={'when': 1},
            )


def test_vector_addfeature_orients_polygon():
    geometry = ogr.CreateGeometryFromWkt(
        'POLYGON ('
        '(0 0, 0 4, 4 4, 4 0, 0 0),'
        '(1 1, 3 1, 3 3, 1 3, 1 1)'
        ')'
    )
    
    # Deliberately wrong:
    # exterior is clockwise, interior is counter-clockwise.
    assert geometry.GetGeometryRef(0).IsClockwise()
    assert not geometry.GetGeometryRef(1).IsClockwise()
    
    with Vector() as vector:
        vector.addlayer('test', 4326, ogr.wkbPolygon)
        vector.addfeature(geometry)
        
        feature = vector.getFeatureByIndex(0)
        stored = feature.GetGeometryRef()
        
        _assert_polygon_orientation(stored)
    
    # addfeature() must not modify the geometry supplied by the caller.
    assert geometry.GetGeometryRef(0).IsClockwise()
    assert not geometry.GetGeometryRef(1).IsClockwise()


def test_vector_addvector_appends_features():
    with bbox(REGULAR_EXTENT, 4326) as first:
        with bbox(
                {'xmin': 12, 'xmax': 13, 'ymin': 50, 'ymax': 51},
                4326,
        ) as second:
            first.addvector(second)
            assert first.nfeatures == 2
            assert second.nfeatures == 1


def test_vector_addfield_string_width():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.addfield('name', ogr.OFTString, width=25, values=['example'])
        definition = vector.layerdef.GetFieldDefn(vector.layerdef.GetFieldIndex('name'))
        
        assert definition.GetWidth() == 25
        assert vector.getFeatureByIndex(0).GetField('name') == 'example'


@pytest.mark.parametrize(
    'field_type,value',
    [
        (ogr.OFTString, 'a'),
        (ogr.OFTStringList, ['a', 'b']),
        (ogr.OFTInteger, 1),
        (ogr.OFTIntegerList, [1, 2]),
        (ogr.OFTInteger64, 2 ** 40),
        (ogr.OFTInteger64List, [2 ** 40, 2 ** 40 + 1]),
        (ogr.OFTReal, 1.5),
        (ogr.OFTRealList, [1.5, 2.5]),
    ],
    ids=[
        'string',
        'string-list',
        'integer',
        'integer-list',
        'integer64',
        'integer64-list',
        'real',
        'real-list',
    ],
)
def test_vector_addfield_supported_types(field_type, value):
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.addfield('value', field_type, values=[value])
        assert vector.getFeatureByIndex(0).GetField('value') == value


def test_vector_addfield_binary():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.addfield('value', ogr.OFTBinary, values=[b'abc'])
        feature = vector.getFeatureByIndex(0)
        assert bytes(feature.GetFieldAsBinary('value')) == b'abc'


@pytest.mark.parametrize(
    'field_type',
    [ogr.OFTDate, ogr.OFTTime],
    ids=['date', 'time'],
)
def test_vector_addfield_rejects_unsupported_types(field_type):
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with pytest.raises(ValueError, match='Unsupported field type'):
            vector.addfield('value', field_type, values=[datetime.now()])


def test_vector_addfield_rejects_wrong_number_of_values():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with pytest.raises(RuntimeError, match='number of values does not match'):
            vector.addfield('value', ogr.OFTString, values=['a', 'b'])


def test_set_field_rejects_invalid_target():
    with pytest.raises(TypeError, match="'target' must be of type"):
        set_field('invalid', 'value', ogr.OFTString, values='a')


def test_set_field_on_feature():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.addfield('name', ogr.OFTString)
        feature = vector.getFeatureByIndex(0)
        
        set_field(feature, 'name', ogr.OFTString, values='example')
        
        assert feature.GetField('name') == 'example'


@pytest.mark.parametrize(
    'dt,expected_tz_flag',
    [
        (datetime(2024, 1, 2, 3, 4, 5, 123600), 0),
        (datetime(2024, 1, 2, 3, 4, 5, 123600, tzinfo=timezone.utc), 100),
        (
                datetime(
                    2024,
                    1,
                    2,
                    3,
                    4,
                    5,
                    123600,
                    tzinfo=timezone(timedelta(hours=2)),
                ),
                1,
        ),
    ],
    ids=['naive', 'utc', 'known-non-utc'],
)
def test_set_field_datetime_timezone_flag(dt, expected_tz_flag):
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.addfield('when', ogr.OFTDateTime, values=[dt])
        feature = vector.getFeatureByIndex(0)
        index = feature.GetFieldIndex('when')
        
        assert feature.GetFieldAsDateTime(index)[6] == expected_tz_flag


def test_set_field_datetime_rounds_to_milliseconds():
    dt = datetime(2024, 1, 2, 3, 4, 5, 123600, tzinfo=timezone.utc)
    
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.addfield('when', ogr.OFTDateTime, values=[dt])
        gdf = vector.to_geopandas()
    
    assert gdf.loc[0, 'when'].microsecond == 124000


def test_set_field_datetime_rejects_non_datetime():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with pytest.raises(TypeError, match='datetime.datetime object'):
            vector.addfield('when', ogr.OFTDateTime, values=[1])


# -----------------------------------------------------------------------------
# Geometry conversion and extents
# -----------------------------------------------------------------------------


def test_vector_get_area_sums_features():
    with wkt2vector(
            [
                'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))',
                'POLYGON ((2 0, 2 2, 3 2, 3 0, 2 0))',
            ],
            srs=4326,
    ) as vector:
        assert vector.getArea() == 3.0


@pytest.mark.parametrize(
    'wkt,expected_type',
    [
        ('POINT (1 2)', ogr.wkbMultiPoint),
        ('LINESTRING (0 0, 1 1)', ogr.wkbMultiLineString),
        (
                'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))',
                ogr.wkbMultiPolygon,
        ),
    ],
    ids=['point', 'linestring', 'polygon'],
)
def test_vector_convert2wkt_promotes_to_multi(wkt, expected_type):
    with _vector_from_wkts(wkt) as vector:
        output = vector.convert2wkt(multi=True)
    
    geometry = ogr.CreateGeometryFromWkt(output[0])
    assert ogr.GT_Flatten(geometry.GetGeometryType()) == expected_type


def test_vector_convert2wkt_preserves_geometry_type_by_default():
    with _vector_from_wkts('POINT (1 2)') as vector:
        output = vector.convert2wkt()
    
    geometry = ogr.CreateGeometryFromWkt(output[0])
    assert ogr.GT_Flatten(geometry.GetGeometryType()) == ogr.wkbPoint


def test_vector_convert2wkt_can_remove_z_dimension():
    with _vector_from_wkts('POINT Z (1 2 3)') as vector:
        output = vector.convert2wkt(set3D=False)
    
    geometry = ogr.CreateGeometryFromWkt(output[0])
    assert geometry.GetCoordinateDimension() == 2


def test_vector_convert2wkt_can_preserve_z_dimension():
    with _vector_from_wkts('POINT Z (1 2 3)') as vector:
        output = vector.convert2wkt(set3D=True)
    
    geometry = ogr.CreateGeometryFromWkt(output[0])
    assert geometry.GetCoordinateDimension() == 3


def test_vector_extent_regular_polygon():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        assert vector.extent == REGULAR_EXTENT
        assert vector.get_extent(split_antimeridian=False) == REGULAR_EXTENT


def test_vector_extent_projected_ignores_antimeridian_logic():
    extent = {
        'xmin': 600000,
        'xmax': 700000,
        'ymin': 5700000,
        'ymax': 5800000,
    }
    with bbox(extent, 32632) as vector:
        assert vector.get_extent(split_antimeridian=True) == extent


def test_vector_extent_points_uses_shortest_longitude_interval():
    with _vector_from_wkts(
            ['POINT (179 0)', 'POINT (-179 1)'],
            geom_type=ogr.wkbPoint,
    ) as vector:
        assert vector.get_extent(split_antimeridian=False) == {
            'xmin': -179.0,
            'xmax': 179.0,
            'ymin': 0.0,
            'ymax': 1.0,
        }
        assert vector.get_extent(split_antimeridian=True) == {
            'xmin': 179.0,
            'xmax': -179.0,
            'ymin': 0.0,
            'ymax': 1.0,
        }


def test_vector_extent_multipoint_uses_shortest_longitude_interval():
    with _vector_from_wkts(
            'MULTIPOINT ((179 0), (-179 1))',
            geom_type=ogr.wkbMultiPoint,
    ) as vector:
        assert vector.extent == {
            'xmin': 179.0,
            'xmax': -179.0,
            'ymin': 0.0,
            'ymax': 1.0,
        }


def test_vector_get_extent_parts_splits_multipolygon_parts():
    with bbox(ANTIMERIDIAN_EXTENT, 4326) as vector:
        parts = sorted(vector.get_extent_parts(), key=lambda item: item['xmin'])
    
    assert parts == [
        {'xmin': -180.0, 'xmax': -179.0, 'ymin': 50.0, 'ymax': 51.0},
        {'xmin': 179.0, 'xmax': 180.0, 'ymin': 50.0, 'ymax': 51.0},
    ]


def test_vector_bbox_method():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with vector.bbox(buffer=1) as box:
            assert box.extent == {
                'xmin': 9.0,
                'xmax': 12.0,
                'ymin': 49.0,
                'ymax': 52.0,
            }


# -----------------------------------------------------------------------------
# bbox()
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    'buffer,expected_area,expected_extent',
    [
        (
                None,
                1240.0,
                {'xmin': 10, 'xmax': 50, 'ymin': 20, 'ymax': 51},
        ),
        (
                1,
                1386.0,
                {'xmin': 9, 'xmax': 51, 'ymin': 19, 'ymax': 52},
        ),
        (
                (1, 2),
                1470.0,
                {'xmin': 9, 'xmax': 51, 'ymin': 18, 'ymax': 53},
        ),
    ],
    ids=['none', 'scalar', 'xy'],
)
def test_bbox_buffer(buffer, expected_area, expected_extent):
    coordinates = {'xmin': 10, 'xmax': 50, 'ymin': 20, 'ymax': 51}
    
    with bbox(coordinates, 4326, buffer=buffer) as vector:
        assert vector.getArea() == expected_area
        assert vector.extent == expected_extent


def test_bbox_geographic_buffer_is_clamped():
    coordinates = {'xmin': -179, 'xmax': 179, 'ymin': -89, 'ymax': 89}
    
    with bbox(coordinates, 4326, buffer=5) as vector:
        assert vector.extent == {
            'xmin': -180.0,
            'xmax': 180.0,
            'ymin': -90.0,
            'ymax': 90.0,
        }


def test_bbox_antimeridian_unsplit():
    with bbox(
            ANTIMERIDIAN_EXTENT,
            4326,
            split_antimeridian=False,
    ) as vector:
        assert vector.geomType == ogr.wkbPolygon
        assert vector.get_extent() == {
            'xmin': -179.0,
            'xmax': 179.0,
            'ymin': 50.0,
            'ymax': 51.0,
        }


def test_bbox_antimeridian_split():
    with bbox(
            ANTIMERIDIAN_EXTENT,
            4326,
            split_antimeridian=True,
    ) as vector:
        assert vector.geomType == ogr.wkbMultiPolygon
        assert vector.get_extent(split_antimeridian=False) == {
            'xmin': -180.0,
            'xmax': 180.0,
            'ymin': 50.0,
            'ymax': 51.0,
        }
        assert vector.extent == ANTIMERIDIAN_EXTENT


def test_bbox_antimeridian_split_with_buffer():
    with bbox(
            ANTIMERIDIAN_EXTENT,
            4326,
            split_antimeridian=True,
            buffer=3,
    ) as vector:
        assert vector.extent == {
            'xmin': 176.0,
            'xmax': -176.0,
            'ymin': 47.0,
            'ymax': 54.0,
        }


def test_bbox_antimeridian_logic_is_not_applied_to_projected_crs():
    coordinates = {
        'xmin': 709800,
        'xmax': 600000,
        'ymin': 5790240,
        'ymax': 5900040,
    }
    
    with bbox(coordinates, 32632) as vector:
        assert vector.geomType == ogr.wkbPolygon


def test_bbox_can_write_file(tmp_path):
    filename = tmp_path / 'bbox.shp'
    
    result = bbox(REGULAR_EXTENT, 4326, outname=str(filename))
    
    assert result is None
    assert filename.exists()
    with Vector(str(filename)) as vector:
        assert vector.extent == REGULAR_EXTENT


# -----------------------------------------------------------------------------
# Cloning, CRS handling, reprojection and antimeridian wrapping
# -----------------------------------------------------------------------------


def test_vector_clone_is_independent():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with vector.clone() as cloned:
            cloned.addfield('name', ogr.OFTString, values=['clone'])
            
            assert cloned.fieldnames == ['area', 'name']
            assert vector.fieldnames == ['area']


def test_vector_reproject_changes_crs():
    with bbox(UTM_EXTENT, 32632) as vector:
        vector.reproject(4326)
        
        assert vector.getProjection('epsg') == 4326
        assert vector.geomType == ogr.wkbPolygon
        assert vector.geomTypes == ['POLYGON']
        assert vector.extent == pytest.approx(
            {'xmin': 10.5, 'xmax': 12.1, 'ymin': 52.2, 'ymax': 53.2},
            rel=1e-1,
        )


def test_vector_reproject_can_return_new_vector():
    with bbox(UTM_EXTENT, 32632) as vector:
        with vector.reproject(4326, inplace=False) as reprojected:
            assert reprojected.getProjection('epsg') == 4326
            assert vector.getProjection('epsg') == 32632


def test_vector_reproject_noop_inplace_returns_none():
    with bbox(UTM_EXTENT, 32632) as vector:
        result = vector.reproject(
            32632,
            split_antimeridian=False,
            inplace=True,
        )
        
        assert result is None
        assert vector.getProjection('epsg') == 32632


def test_vector_reproject_noop_returns_clone_when_not_inplace():
    with bbox(UTM_EXTENT, 32632) as vector:
        with vector.reproject(
                32632,
                split_antimeridian=False,
                inplace=False,
        ) as cloned:
            assert cloned is not vector
            assert cloned.extent == vector.extent


def test_vector_reproject_rejects_layer_feature_geometry_type_mismatch():
    polygon = ogr.CreateGeometryFromWkt(
        'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    )
    multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
    multipolygon.AddGeometry(polygon)
    
    with Vector() as vector:
        vector.addlayer('test', 4326, ogr.wkbPolygon)
        vector.addfeature(multipolygon)
        
        assert vector.geomType == ogr.wkbPolygon
        assert vector.geomTypes == ['MULTIPOLYGON']
        with pytest.raises(ValueError, match='geometry types of the layer'):
            vector.reproject(3857)


def test_vector_reproject_can_disable_antimeridian_splitting():
    with bbox(
            ANTIMERIDIAN_EXTENT,
            4326,
            split_antimeridian=False,
    ) as vector:
        with vector.reproject(
                4326,
                split_antimeridian=False,
                inplace=False,
        ) as cloned:
            assert cloned.geomType == ogr.wkbPolygon
            assert cloned.geomTypes == ['POLYGON']


def test_vector_reproject_promotes_split_polygon_to_multipolygon():
    with bbox(UTM_EXTENT, 32660) as vector:
        vector.reproject(4326)
        
        assert vector.geomType == ogr.wkbMultiPolygon
        assert vector.geomTypes == ['MULTIPOLYGON']
        assert vector.extent == pytest.approx(
            {'xmin': 178.5, 'xmax': -179.9, 'ymin': 52.2, 'ymax': 53.2},
            rel=1e-1,
        )


def test_vector_reproject_orients_polygon():
    geometry = ogr.CreateGeometryFromWkt(
        'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    )
    assert geometry.GetGeometryRef(0).IsClockwise()
    
    with Vector() as vector:
        vector.addlayer('test', 4326, ogr.wkbPolygon)
        
        feature = ogr.Feature(vector.layerdef)
        feature.SetGeometry(geometry)
        vector.layer.CreateFeature(feature)
        vector.init_features()
        
        # Confirm that the test really starts with wrong winding.
        input_feature = vector.getFeatureByIndex(0)
        input_geometry = input_feature.GetGeometryRef()
        
        assert input_geometry.GetGeometryRef(0).IsClockwise()
        
        vector.reproject(3857)
        
        output_feature = vector.getFeatureByIndex(0)
        output_geometry = output_feature.GetGeometryRef()
        
        _assert_polygon_orientation(output_geometry)


def test_vector_reproject_returned_vector_orients_polygon():
    geometry = ogr.CreateGeometryFromWkt(
        'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    )
    
    with Vector() as vector:
        vector.addlayer('test', 4326, ogr.wkbPolygon)
        
        feature = ogr.Feature(vector.layerdef)
        feature.SetGeometry(geometry)
        vector.layer.CreateFeature(feature)
        vector.init_features()
        
        with vector.reproject(3857, inplace=False) as result:
            output_feature = result.getFeatureByIndex(0)
            output_geometry = output_feature.GetGeometryRef()
            
            _assert_polygon_orientation(output_geometry)


def test_vector_wrap_antimeridian_inplace():
    with bbox(
            ANTIMERIDIAN_EXTENT,
            4326,
            split_antimeridian=False,
    ) as vector:
        result = vector.wrap_antimeridian(inplace=True)
        
        assert result is None
        assert vector.geomType == ogr.wkbMultiPolygon
        assert vector.extent == ANTIMERIDIAN_EXTENT
        
        feature = vector.getFeatureByIndex(0)
        geometry = feature.GetGeometryRef()
        
        _assert_polygon_orientation(geometry)


def test_vector_wrap_antimeridian_can_return_new_vector():
    with bbox(
            ANTIMERIDIAN_EXTENT,
            4326,
            split_antimeridian=False,
    ) as vector:
        with vector.wrap_antimeridian(inplace=False) as wrapped:
            assert wrapped.geomType == ogr.wkbMultiPolygon
            assert vector.geomType == ogr.wkbPolygon
            
            feature = wrapped.getFeatureByIndex(0)
            geometry = feature.GetGeometryRef()
            
            _assert_polygon_orientation(geometry)


def test_vector_wrap_antimeridian_does_not_change_world_spanning_polygon():
    extent = {'xmin': 178, 'xmax': -178, 'ymin': 50, 'ymax': 51}
    
    with bbox(
            extent,
            4326,
            split_antimeridian=False,
            buffer=3,
    ) as vector:
        vector.wrap_antimeridian()
        
        assert vector.geomType == ogr.wkbPolygon
        assert vector.extent == {
            'xmin': -180.0,
            'xmax': 180.0,
            'ymin': 47.0,
            'ymax': 54.0,
        }


def test_vector_set_crs_changes_crs_without_moving_coordinates():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        before = vector.get_extent(split_antimeridian=False)
        vector.setCRS(3857)
        
        assert vector.getProjection('epsg') == 3857
        assert vector.get_extent(split_antimeridian=False) == before


# -----------------------------------------------------------------------------
# Geo interface and GeoPandas conversion
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    'extent,crs,geometry_type',
    [
        (
                {'xmin': 10, 'ymin': 11, 'xmax': 50, 'ymax': 51},
                4326,
                'Polygon',
        ),
        (
                {'xmin': 179, 'ymin': -50, 'xmax': -179, 'ymax': 51},
                4326,
                'MultiPolygon',
        ),
        (
                UTM_EXTENT,
                32632,
                'Polygon',
        ),
        (
                UTM_EXTENT,
                32660,
                'MultiPolygon',
        ),
    ],
    ids=['regular', 'antimeridian', 'regular-utm', 'antimeridian-utm'],
)
def test_vector_geo_interface(extent, crs, geometry_type):
    with bbox(extent, crs) as vector:
        geojson = vector.__geo_interface__
    
    assert geojson['type'] == 'FeatureCollection'
    assert len(geojson['features']) == 1
    assert geojson['features'][0]['type'] == 'Feature'
    assert geojson['features'][0]['geometry']['type'] == geometry_type


def test_vector_to_geopandas_preserves_geometry_and_fields():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.addfield('name', ogr.OFTString, values=['example'])
        gdf = vector.to_geopandas()
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert list(gdf.columns) == ['area', 'name', 'geometry']
    assert gdf.loc[0, 'area'] == 1.0
    assert gdf.loc[0, 'name'] == 'example'
    assert gdf.geometry.iloc[0].geom_type == 'Polygon'
    assert gdf.crs.to_epsg() == 4326


def test_vector_to_geopandas_converts_datetime_fields():
    value = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.addfield('when', ogr.OFTDateTime, values=[value])
        gdf = vector.to_geopandas()
    
    assert pd.api.types.is_datetime64_any_dtype(gdf['when'])
    assert gdf.loc[0, 'when'].year == 2024


def test_from_geopandas_polygon():
    gdf = gpd.GeoDataFrame(
        {
            'integer': [1],
            'real': [1.5],
            'text': ['example'],
            'geometry': [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
        },
        crs=4326,
    )
    
    with from_geopandas(gdf, layer_name='example') as vector:
        feature = vector.getFeatureByIndex(0)
        
        assert vector.layername == 'example'
        assert vector.geomType == ogr.wkbPolygon
        assert vector.getProjection('epsg') == 4326
        assert feature.GetField('integer') == 1
        assert feature.GetField('real') == 1.5
        assert feature.GetField('text') == 'example'


def test_from_geopandas_maps_integer_to_integer64():
    gdf = gpd.GeoDataFrame(
        {'value': [2 ** 40], 'geometry': [Point(0, 0)]},
        crs=4326,
    )
    
    with from_geopandas(gdf) as vector:
        definition = vector.layerdef.GetFieldDefn(
            vector.layerdef.GetFieldIndex('value')
        )
        assert definition.GetType() == ogr.OFTInteger64


def test_from_geopandas_rejects_multiple_geometry_types():
    gdf = gpd.GeoDataFrame(
        {
            'geometry': [
                Point(0, 0),
                Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            ]
        },
        crs=4326,
    )
    
    with pytest.raises(RuntimeError, match='Multiple geometry types are not supported'):
        from_geopandas(gdf)


def test_from_geopandas_orients_polygon():
    polygon = Polygon(
        shell=[
            (0, 0),
            (0, 4),
            (4, 4),
            (4, 0),
            (0, 0),
        ],
        holes=[[
            (1, 1),
            (3, 1),
            (3, 3),
            (1, 3),
            (1, 1),
        ]],
    )
    
    gdf = gpd.GeoDataFrame(
        {'geometry': [polygon]},
        crs=4326,
    )
    
    with from_geopandas(gdf) as vector:
        feature = vector.getFeatureByIndex(0)
        geometry = feature.GetGeometryRef()
        
        _assert_polygon_orientation(geometry)


# -----------------------------------------------------------------------------
# File writing
# -----------------------------------------------------------------------------


def test_vector_write_and_reopen(tmp_path):
    filename = tmp_path / 'vector.shp'
    
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.write(str(filename))
    
    with Vector(str(filename)) as reopened:
        assert reopened.extent == REGULAR_EXTENT
        assert reopened.fieldnames == ['area']


def test_vector_write_overwrites_by_default(tmp_path):
    filename = tmp_path / 'vector.shp'
    
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.write(str(filename))
        vector.write(str(filename))
    
    assert filename.exists()


def test_vector_write_rejects_existing_target_without_overwrite(tmp_path):
    filename = tmp_path / 'vector.shp'
    
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.write(str(filename))
        with pytest.raises(RuntimeError, match='target file already exists'):
            vector.write(str(filename), overwrite=False)


def test_vector_write_with_explicit_driver(tmp_path):
    filename = tmp_path / 'vector.data'
    
    with bbox(REGULAR_EXTENT, 4326) as vector:
        vector.write(str(filename), driver='GeoJSON')
    
    assert filename.exists()
    with Vector(str(filename), driver='GeoJSON') as reopened:
        assert reopened.extent == REGULAR_EXTENT


# -----------------------------------------------------------------------------
# feature2vector() and wkt2vector()
# -----------------------------------------------------------------------------


def test_feature2vector_single_feature():
    with bbox(REGULAR_EXTENT, 4326) as reference:
        feature = reference.getFeatureByIndex(0)
        with feature2vector(feature, ref=reference) as vector:
            assert vector.nfeatures == 1
            assert vector.fieldnames == ['area']
            assert vector.extent == REGULAR_EXTENT


def test_feature2vector_feature_list_and_custom_layer_name():
    with bbox(REGULAR_EXTENT, 4326) as reference:
        other = bbox(
            {'xmin': 12, 'xmax': 13, 'ymin': 50, 'ymax': 51},
            4326,
        )
        try:
            reference.addvector(other)
            features = reference.getfeatures()
            with feature2vector(
                    features,
                    ref=reference,
                    layername='selected',
            ) as vector:
                assert vector.layername == 'selected'
                assert vector.nfeatures == 2
        finally:
            other.close()


def test_wkt2vector_single_polygon():
    wkt = 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
    
    with wkt2vector(wkt, srs=4326) as vector:
        assert vector.nfeatures == 1
        assert vector.getArea() == 1.0
        assert vector.getFeatureByIndex(0).GetField('area') == 1.0


def test_wkt2vector_multiple_polygons():
    wkts = [
        'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))',
        'POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))',
    ]
    
    with wkt2vector(wkts, srs=4326, layername='polygons') as vector:
        assert vector.layername == 'polygons'
        assert vector.nfeatures == 2
        assert vector.getArea() == 2.0


def test_wkt2vector_point_has_null_area():
    with wkt2vector('POINT (1 2)', srs=4326) as vector:
        assert vector.geomType == ogr.wkbPoint
        assert vector.getFeatureByIndex(0).GetField('area') is None


def test_wkt2vector_flattens_to_2d():
    with wkt2vector('POINT Z (1 2 3)', srs=4326) as vector:
        feature = vector.getFeatureByIndex(0)
        geometry = feature.GetGeometryRef()
        assert not geometry.Is3D()


def test_wkt2vector_orients_polygon():
    wkt = (
        'POLYGON ('
        '(0 0, 0 4, 4 4, 4 0, 0 0),'
        '(1 1, 3 1, 3 3, 1 3, 1 1)'
        ')'
    )
    
    with wkt2vector(wkt, srs=4326) as vector:
        feature = vector.getFeatureByIndex(0)
        geometry = feature.GetGeometryRef()
        
        _assert_polygon_orientation(geometry)


# -----------------------------------------------------------------------------
# intersect()
# -----------------------------------------------------------------------------


def test_intersect_overlapping_polygons():
    with bbox(
            {'xmin': 0, 'xmax': 2, 'ymin': 0, 'ymax': 2},
            4326,
    ) as first:
        with bbox(
                {'xmin': 1, 'xmax': 3, 'ymin': 1, 'ymax': 3},
                4326,
        ) as second:
            with intersect(first, second) as result:
                assert result.nfeatures == 1
                assert result.geomType == ogr.wkbMultiPolygon
                assert result.geomTypes == ['MULTIPOLYGON']
                assert result.getArea() == pytest.approx(1.0)
                
                feature = result.getFeatureByIndex(0)
                geometry = feature.GetGeometryRef()
                
                _assert_polygon_orientation(geometry)


def test_intersect_disjoint_polygons_returns_none():
    with bbox(
            {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1},
            4326,
    ) as first:
        with bbox(
                {'xmin': 2, 'xmax': 3, 'ymin': 2, 'ymax': 3},
                4326,
        ) as second:
            assert intersect(first, second) is None


def test_intersect_reprojects_first_object_to_second_crs():
    with bbox(REGULAR_EXTENT, 4326) as first:
        with first.reproject(3857, inplace=False) as second:
            with intersect(first, second) as result:
                assert result.getProjection('epsg') == 3857
                assert result.nfeatures == 1
                assert result.getArea() > 0


def test_intersect_handles_antimeridian():
    with bbox(
            {'xmin': 178, 'xmax': -178, 'ymin': 50, 'ymax': 51},
            4326,
    ) as first:
        with bbox(ANTIMERIDIAN_EXTENT, 4326) as second:
            with intersect(first, second) as result:
                assert result.getArea() == pytest.approx(2.0)
                assert result.extent == ANTIMERIDIAN_EXTENT


def test_intersect_rejects_non_vector_input():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with pytest.raises(RuntimeError, match='both objects must be of type Vector'):
            intersect(vector, 'invalid')


def test_intersect_rejects_non_polygon_geometry():
    with _vector_from_wkts('POINT (0 0)') as point:
        with bbox(REGULAR_EXTENT, 4326) as polygon:
            with pytest.raises(RuntimeError, match='only supports polygon'):
                intersect(point, polygon)


def test_intersect_contained_polygon():
    with bbox(
            {'xmin': 0, 'xmax': 4, 'ymin': 0, 'ymax': 4},
            4326,
    ) as first:
        with bbox(
                {'xmin': 1, 'xmax': 2, 'ymin': 1, 'ymax': 2},
                4326,
        ) as second:
            with intersect(first, second) as result:
                assert result.nfeatures == 1
                assert result.extent == {
                    'xmin': 1,
                    'xmax': 2,
                    'ymin': 1,
                    'ymax': 2,
                }
                assert result.getArea() == pytest.approx(1.0)


def test_intersect_touching_edge_returns_none():
    with bbox(
            {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1},
            4326,
    ) as first:
        with bbox(
                {'xmin': 1, 'xmax': 2, 'ymin': 0, 'ymax': 1},
                4326,
        ) as second:
            assert intersect(first, second) is None


def test_intersect_preserves_fields():
    with bbox(
            {'xmin': 0, 'xmax': 2, 'ymin': 0, 'ymax': 2},
            4326,
    ) as first:
        with bbox(
                {'xmin': 1, 'xmax': 3, 'ymin': 1, 'ymax': 3},
                4326,
        ) as second:
            first.addfield('name', ogr.OFTString, values=['a'])
            second.addfield('name', ogr.OFTString, values=['b'])
            
            with intersect(first, second) as result:
                assert result.nfeatures == 1
                assert 'input_name' in result.fieldnames
                assert 'method_name' in result.fieldnames
                
                feature = result.getFeatureByIndex(0)
                assert feature.GetField('input_name') == 'a'
                assert feature.GetField('method_name') == 'b'


# -----------------------------------------------------------------------------
# vectorize()
# -----------------------------------------------------------------------------


def test_vectorize_returns_vector():
    array = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [2, 2, 2],
        ],
        dtype=np.uint8,
    )
    
    with _memory_raster(array) as raster:
        with vectorize(array, raster) as vector:
            assert vector.getProjection('epsg') == 4326
            assert vector.geomType == ogr.wkbPolygon
            assert vector.fieldnames == ['value']
            assert vector.getUniqueAttributes('value') == [0, 1, 2]


def test_vectorize_preserves_geotransform_extent():
    array = np.ones((2, 3), dtype=np.uint8)
    
    with _memory_raster(array) as raster:
        with vectorize(array, raster) as vector:
            assert vector.extent == {
                'xmin': 0.0,
                'xmax': 3.0,
                'ymin': 0.0,
                'ymax': 2.0,
            }


def test_vectorize_can_write_file(tmp_path):
    array = np.ones((2, 3), dtype=np.uint8)
    filename = tmp_path / 'polygonized.shp'
    
    with _memory_raster(array) as raster:
        result = vectorize(array, raster, outname=str(filename))
    
    assert result is None
    assert filename.exists()
    with Vector(str(filename)) as vector:
        assert vector.nfeatures == 1
        assert vector.getFeatureByIndex(0).GetField('value') == 1


def test_vectorize_orients_polygon():
    array = np.ones((2, 2), dtype=np.uint8)
    
    with _memory_raster(
            array,
            # positive Y pixel size flips the winding in Polygonize()
            geotransform=(0, 1, 0, 0, 0, 1),
    ) as raster:
        with vectorize(array, raster) as vector:
            assert vector.nfeatures == 1
            
            feature = vector.getFeatureByIndex(0)
            geometry = feature.GetGeometryRef()
            
            _assert_polygon_orientation(geometry)


# -----------------------------------------------------------------------------
# hull()
# -----------------------------------------------------------------------------


def test_hull_rejects_non_vector_input():
    with pytest.raises(TypeError, match="'vectorobject' must be of type Vector"):
        hull('invalid')


@pytest.mark.parametrize('ratio', ['0.5', None, True])
def test_hull_rejects_non_numeric_ratio(ratio):
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with pytest.raises(TypeError, match="'ratio' must be numeric"):
            hull(vector, ratio=ratio)


@pytest.mark.parametrize('ratio', [-0.1, 1.1])
def test_hull_rejects_ratio_outside_range(ratio):
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with pytest.raises(ValueError, match=r'\[0, 1\]'):
            hull(vector, ratio=ratio)


def test_hull_rejects_empty_geometry():
    with _vector_from_wkts('POINT EMPTY', geom_type=ogr.wkbPoint) as vector:
        with pytest.raises(RuntimeError, match='no valid geometry found'):
            hull(vector)


def test_hull_rejects_mixed_geometry_types():
    with _vector_from_wkts(
            ['POINT (0 0)', 'LINESTRING (0 0, 1 1)'],
            geom_type=ogr.wkbUnknown,
    ) as vector:
        with pytest.raises(RuntimeError, match='exactly one geometry type'):
            hull(vector)


def test_hull_rejects_unsupported_geometry_type():
    with _vector_from_wkts(
            'GEOMETRYCOLLECTION (POINT (0 0), POINT (1 1))',
            geom_type=ogr.wkbGeometryCollection,
    ) as vector:
        with pytest.raises(RuntimeError, match='only supports Point'):
            hull(vector)


def test_hull_polygon_ratio_one_returns_convex_hull():
    wkts = [
        'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))',
        'POLYGON ((2 0, 2 1, 3 1, 3 0, 2 0))',
    ]
    
    with _vector_from_wkts(wkts, geom_type=ogr.wkbPolygon) as vector:
        with hull(vector, ratio=1) as result:
            assert result.geomType == ogr.wkbPolygon
            assert result.getArea() == pytest.approx(3.0)


def test_hull_points_ratio_one_returns_convex_hull():
    with _vector_from_wkts(
            [
                'POINT (0 0)',
                'POINT (0 1)',
                'POINT (1 1)',
                'POINT (1 0)',
            ],
            geom_type=ogr.wkbPoint,
    ) as vector:
        with hull(vector, ratio=1) as result:
            assert result.geomType == ogr.wkbPolygon
            assert result.getArea() == pytest.approx(1.0)


@pytest.mark.parametrize(
    'wkt,expected_type',
    [
        ('POINT (0 0)', ogr.wkbPoint),
        ('LINESTRING (0 0, 1 1)', ogr.wkbLineString),
    ],
    ids=['point', 'line'],
)
def test_hull_degenerate_input_preserves_lower_dimension(wkt, expected_type):
    with _vector_from_wkts(wkt) as vector:
        with hull(vector, ratio=1) as result:
            assert ogr.GT_Flatten(result.geomType) == expected_type
            assert result.nfeatures == 1


@pytest.mark.parametrize(
    'wkt',
    [
        'MULTIPOINT ((0 0), (0 1), (1 1), (1 0))',
        (
                'MULTILINESTRING ('
                '(0 0, 0 1), (0 1, 1 1), (1 1, 1 0), (1 0, 0 0)'
                ')'
        ),
    ],
    ids=['multipoint', 'multilinestring'],
)
def test_hull_supports_multi_geometry_types(wkt):
    with _vector_from_wkts(wkt) as vector:
        with hull(vector, ratio=1) as result:
            assert result.nfeatures == 1
            feature = result.getFeatureByIndex(0)
            geometry = feature.GetGeometryRef()
            assert not geometry.IsEmpty()


def test_hull_points_concave_when_supported():
    if not hasattr(ogr.Geometry, 'ConcaveHull'):
        pytest.skip('OGRGeometry::ConcaveHull is unavailable')
    
    with _vector_from_wkts(
            [
                'POINT (0 0)',
                'POINT (0 2)',
                'POINT (1 1)',
                'POINT (2 2)',
                'POINT (2 0)',
            ],
            geom_type=ogr.wkbPoint,
    ) as vector:
        with hull(vector, ratio=0.5) as result:
            assert result.nfeatures == 1
            assert result.getArea() > 0


def test_hull_lines_concave_when_supported():
    if not hasattr(ogr.Geometry, 'ConcaveHull'):
        pytest.skip('OGRGeometry::ConcaveHull is unavailable')
    
    with _vector_from_wkts(
            [
                'LINESTRING (0 0, 0 2)',
                'LINESTRING (0 2, 2 2)',
                'LINESTRING (2 2, 2 0)',
            ],
            geom_type=ogr.wkbLineString,
    ) as vector:
        with hull(vector, ratio=0.5) as result:
            assert result.nfeatures == 1
            assert result.getArea() > 0


def test_hull_connects_disconnected_polygons_when_supported():
    wkts = [
        'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))',
        'POLYGON ((3 0, 3 1, 4 1, 4 0, 3 0))',
    ]
    
    with _vector_from_wkts(wkts, geom_type=ogr.wkbPolygon) as vector:
        if not hasattr(ogr.Geometry, 'ConcaveHullOfPolygons'):
            with pytest.raises(RuntimeError, match='ConcaveHullOfPolygons'):
                hull(vector, ratio=0.5, connect=True)
            return
        
        with hull(vector, ratio=0.5, connect=True) as result:
            assert ogr.GT_Flatten(result.geomType) in {
                ogr.wkbPolygon,
                ogr.wkbMultiPolygon,
            }
            assert result.getArea() >= 2.0


def test_hull_antimeridian():
    with bbox(ANTIMERIDIAN_EXTENT, 4326) as vector:
        with hull(vector, ratio=0.5) as result:
            assert result.extent == ANTIMERIDIAN_EXTENT
            assert result.getArea() == pytest.approx(2.0)
            feature = result.getFeatureByIndex(0)
            geometry = feature.GetGeometryRef()
            _assert_polygon_orientation(geometry)


@pytest.mark.parametrize(
    'wkt',
    [
        (
                'POLYGON (('
                '0 0, 1 0, 1 1, 0 1, 0 0'
                '))'
        ),
        (
                'POLYGON (('
                '0 0, 0 1, 1 1, 1 0, 0 0'
                '))'
        ),
    ],
    ids=['counter-clockwise', 'clockwise'],
)
def test_hull_polygon_orientation(wkt):
    with _vector_from_wkts(
            wkt,
            geom_type=ogr.wkbPolygon,
    ) as vector:
        with hull(vector) as result:
            feature = result.getFeatureByIndex(0)
            geometry = feature.GetGeometryRef()
            
            _assert_polygon_orientation(geometry)


def test_hull_convex_hull_is_counterclockwise():
    wkt = (
        'POLYGON (('
        '8.505644 50.295261, '
        '12.0268 50.688881, '
        '11.653832 52.183979, '
        '8.017178 51.788181, '
        '8.505644 50.295261'
        '))'
    )
    
    with _vector_from_wkts(
            wkt,
            geom_type=ogr.wkbPolygon,
    ) as vector:
        with hull(vector) as result:
            feature = result.getFeatureByIndex(0)
            geometry = feature.GetGeometryRef()
            
            _assert_polygon_orientation(geometry)


def test_hull_unary_union_orientation():
    wkts = [
        'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))',
        'POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))',
    ]
    
    with _vector_from_wkts(
            wkts,
            geom_type=ogr.wkbPolygon,
    ) as vector:
        with hull(vector, ratio=0.5) as result:
            feature = result.getFeatureByIndex(0)
            geometry = feature.GetGeometryRef()
            
            _assert_polygon_orientation(geometry)


def test_hull_multipolygon_orientation():
    wkts = [
        'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))',
        'POLYGON ((2 0, 3 0, 3 1, 2 1, 2 0))',
    ]
    
    with _vector_from_wkts(
            wkts,
            geom_type=ogr.wkbPolygon,
    ) as vector:
        with hull(vector, ratio=0.5) as result:
            feature = result.getFeatureByIndex(0)
            geometry = feature.GetGeometryRef()
            
            assert ogr.GT_Flatten(
                geometry.GetGeometryType()
            ) == ogr.wkbMultiPolygon
            
            _assert_polygon_orientation(geometry)


# -----------------------------------------------------------------------------
# dissolve()
# -----------------------------------------------------------------------------


def test_dissolve(tmp_path, travis):
    if travis:
        pytest.skip('requires loadable GDAL and SpatiaLite SQLite extensions')
    
    source = tmp_path / 'source.shp'
    output = tmp_path / 'dissolved.gpkg'
    
    with bbox(
            {'xmin': 0, 'xmax': 2, 'ymin': 0, 'ymax': 1},
            4326,
    ) as vector:
        with bbox(
                {'xmin': 1, 'xmax': 3, 'ymin': 0, 'ymax': 1},
                4326,
        ) as other:
            vector.addvector(other)
        vector.write(str(source))
    
    dissolve(
        infile=str(source),
        outfile=str(output),
        field='area',
        layername='merged',
    )
    
    with Vector(str(output)) as result:
        assert result.layername == 'merged'
        assert result.nfeatures == 1
        assert result.getArea() == pytest.approx(3.0)
        assert result.getFeatureByIndex(0).GetField('area') == 2.0
        
        feature = result.getFeatureByIndex(0)
        geometry = feature.GetGeometryRef()
        
        _assert_polygon_orientation(geometry)


# -----------------------------------------------------------------------------
# combine_polygons()
# -----------------------------------------------------------------------------


def test_combine_polygons_preserves_polygons():
    extents = [
        {'xmin': 10, 'xmax': 11, 'ymin': 50, 'ymax': 51},
        {'xmin': 11, 'xmax': 12, 'ymin': 50, 'ymax': 51},
    ]
    vectors = [bbox(extent, 4326) for extent in extents]
    
    try:
        with combine_polygons(vectors) as combined:
            assert combined.extent == {
                'xmin': 10,
                'xmax': 12,
                'ymin': 50,
                'ymax': 51,
            }
            assert combined.nfeatures == 2
            assert combined.geomType == ogr.wkbPolygon
    finally:
        for vector in vectors:
            vector.close()


def test_combine_polygons_accepts_single_vector():
    with bbox(REGULAR_EXTENT, 4326) as vector:
        with combine_polygons(vector) as combined:
            assert combined.nfeatures == 1
            assert combined.geomType == ogr.wkbPolygon
            assert combined.extent == REGULAR_EXTENT


def test_combine_polygons_explodes_multipolygon():
    with bbox(ANTIMERIDIAN_EXTENT, 4326) as vector:
        with combine_polygons(vector, explode=True) as combined:
            assert combined.nfeatures == 2
            assert combined.geomType == ogr.wkbPolygon
            assert combined.getArea() == pytest.approx(2.0)


def test_combine_polygons_combines_into_single_multipolygon():
    extents = [
        {'xmin': 10, 'xmax': 11, 'ymin': 50, 'ymax': 51},
        {'xmin': 12, 'xmax': 13, 'ymin': 50, 'ymax': 51},
    ]
    vectors = [bbox(extent, 4326) for extent in extents]
    
    try:
        with combine_polygons(vectors, multipolygon=True) as combined:
            feature = combined.getFeatureByIndex(0)
            geometry = feature.GetGeometryRef()
            assert combined.nfeatures == 1
            assert combined.geomType == ogr.wkbMultiPolygon
            assert geometry.GetGeometryCount() == 2
    finally:
        for vector in vectors:
            vector.close()


def test_combine_polygons_mixed_polygon_and_multipolygon_promotes_polygon():
    with bbox(REGULAR_EXTENT, 4326) as polygon:
        with bbox(ANTIMERIDIAN_EXTENT, 4326) as multipolygon:
            with combine_polygons([polygon, multipolygon]) as combined:
                assert combined.nfeatures == 2
                assert combined.geomType == ogr.wkbMultiPolygon
                assert combined.geomTypes == ['MULTIPOLYGON', 'MULTIPOLYGON']


def test_combine_polygons_mixed_polygon_and_multipolygon_can_explode():
    with bbox(REGULAR_EXTENT, 4326) as polygon:
        with bbox(ANTIMERIDIAN_EXTENT, 4326) as multipolygon:
            with combine_polygons(
                    [polygon, multipolygon],
                    explode=True,
            ) as combined:
                assert combined.nfeatures == 3
                assert combined.geomType == ogr.wkbPolygon
                assert combined.getArea() == pytest.approx(3.0)


def test_combine_polygons_multipolygon_option_takes_precedence_over_explode():
    with bbox(ANTIMERIDIAN_EXTENT, 4326) as vector:
        with combine_polygons(
                vector,
                explode=True,
                multipolygon=True,
        ) as combined:
            assert combined.nfeatures == 1
            assert combined.geomType == ogr.wkbMultiPolygon
            assert combined.getArea() == pytest.approx(2.0)


def test_combine_polygons_reprojects_to_requested_crs():
    with bbox(REGULAR_EXTENT, 4326) as geographic:
        with geographic.reproject(3857, inplace=False) as projected:
            with combine_polygons(
                    [geographic, projected],
                    crs=4326,
            ) as combined:
                assert combined.getProjection('epsg') == 4326
                assert combined.nfeatures == 2
                assert combined.extent == pytest.approx(REGULAR_EXTENT)
