import os
import shutil
import pytest
import platform
import numpy as np
from osgeo import ogr, gdal
from spatialist import crsConvert, haversine, Raster, stack, ogr2ogr, gdal_translate, gdal_rasterize, bbox, rasterize, \
    gdalwarp, utm_autodetect, coordinate_reproject, cmap_mpl2gdal
from spatialist.raster import Dtype, png
from spatialist.vector import feature2vector, dissolve, Vector, intersect
from spatialist.envi import hdr, HDRobject
from spatialist.sqlite_util import sqlite_setup, __Handler
from spatialist.ancillary import parallel_apply_along_axis

import logging

logging.basicConfig(level=logging.DEBUG)


def test_crsConvert():
    assert crsConvert(crsConvert(4326, 'wkt'), 'proj4').strip() == '+proj=longlat +datum=WGS84 +no_defs'
    assert crsConvert(crsConvert(4326, 'prettyWkt'), 'opengis') == 'http://www.opengis.net/def/crs/EPSG/0/4326'
    assert crsConvert('+proj=longlat +datum=WGS84 +no_defs ', 'epsg') == 4326
    assert crsConvert('http://www.opengis.net/def/crs/EPSG/0/4326', 'epsg') == 4326
    assert crsConvert(crsConvert('http://www.opengis.net/def/crs/EPSG/0/4326', 'osr'), 'epsg') == 4326
    assert crsConvert('EPSG:4326+5773', 'proj4').strip() \
           == '+proj=longlat +datum=WGS84 +geoidgrids=egm96_15.gtx +vunits=m +no_defs' \
           or '+proj=longlat +datum=WGS84 +vunits=m +no_defs'
    with pytest.raises(TypeError):
        crsConvert('xyz', 'epsg')
    with pytest.raises(ValueError):
        crsConvert(4326, 'xyz')


def test_haversine():
    assert haversine(50, 10, 51, 10) == 111194.92664455889


def test_Vector(testdata):
    scene = Raster(testdata['tif'])
    bbox1 = scene.bbox()
    assert bbox1.getArea() == 23262400.0
    assert bbox1.extent == {'ymax': 4830114.70107, 'ymin': 4825774.70107, 'xmin': 620048.241204, 'xmax': 625408.241204}
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


def test_Raster(tmpdir, testdata):
    with pytest.raises(RuntimeError):
        with Raster(1) as ras:
            print(ras)
    with Raster(testdata['tif']) as ras:
        print(ras)
        assert ras.bands == 1
        assert ras.proj4.strip() == '+proj=utm +zone=31 +datum=WGS84 +units=m +no_defs'
        assert ras.cols == 268
        assert ras.rows == 217
        assert ras.dim == (217, 268, 1)
        assert ras.dtype == 'Float32'
        assert ras.epsg == 32631
        assert ras.format == 'GTiff'
        assert ras.geo == {'ymax': 4830114.70107, 'rotation_y': 0.0, 'rotation_x': 0.0, 'xmax': 625408.241204,
                           'xres': 20.0, 'xmin': 620048.241204, 'ymin': 4825774.70107, 'yres': -20.0}
        assert ras.geogcs == 'WGS 84'
        assert ras.is_valid() is True
        assert ras.proj4args == {'units': 'm', 'no_defs': None, 'datum': 'WGS84', 'proj': 'utm', 'zone': '31'}
        assert ras.allstats() == [{'min': -26.65471076965332, 'max': 1.4325850009918213,
                                   'mean': -12.124929534450377, 'sdev': 4.738273594738293}]
        assert ras.bbox().getArea() == 23262400.0
        assert len(ras.layers()) == 1
        assert ras.projcs == 'WGS 84 / UTM zone 31N'
        assert ras.res == (20.0, 20.0)
        
        # test writing a subset with no original data in memory
        outname = os.path.join(str(tmpdir), 'test_sub.tif')
        with ras[0:200, 0:100] as sub:
            sub.write(outname, format='GTiff')
        with Raster(outname) as ras2:
            assert ras2.cols == 100
            assert ras2.rows == 200
        
        ras.load()
        mat = ras.matrix()
        assert isinstance(mat, np.ndarray)
        ras.assign(mat, band=0)
        # ras.reduce()
        ras.rescale(lambda x: 10 * x)
        
        # test writing data with original data in memory
        ras.write(os.path.join(str(tmpdir), 'test'), format='GTiff')
        with pytest.raises(RuntimeError):
            ras.write(os.path.join(str(tmpdir), 'test.tif'), format='GTiff')


def test_Raster_subset(testdata):
    with Raster(testdata['tif']) as ras:
        ext = ras.bbox().extent
        xres, yres = ras.res
        ext['xmin'] += xres
        ext['xmax'] -= xres
        ext['ymin'] += yres
        ext['ymax'] -= yres
        with bbox(ext, ras.proj4) as vec:
            with ras[vec] as sub:
                xres, yres = ras.res
                assert sub.geo['xmin'] - ras.geo['xmin'] == xres
                assert ras.geo['xmax'] - sub.geo['xmax'] == xres
                assert sub.geo['ymin'] - ras.geo['ymin'] == xres
                assert ras.geo['ymax'] - sub.geo['ymax'] == xres


def test_Raster_extract(testdata):
    with Raster(testdata['tif']) as ras:
        assert ras.extract(px=624000, py=4830000, radius=5) == -10.48837461270875
        with pytest.raises(RuntimeError):
            ras.extract(1, 4830000)
        with pytest.raises(RuntimeError):
            ras.extract(624000, 1)
        
        # ensure corner extraction capability
        assert ras.extract(px=ras.geo['xmin'], py=ras.geo['ymax']) == -10.147890090942383
        assert ras.extract(px=ras.geo['xmin'], py=ras.geo['ymin']) == -14.640368461608887
        assert ras.extract(px=ras.geo['xmax'], py=ras.geo['ymax']) == -9.599242210388182
        assert ras.extract(px=ras.geo['xmax'], py=ras.geo['ymin']) == -9.406558990478516
        
        # test nodata handling capability and correct indexing
        mat = ras.matrix()
        mat[0:10, 0:10] = ras.nodata
        mat[207:217, 258:268] = ras.nodata
        ras.assign(mat, band=0)
        assert ras.extract(px=ras.geo['xmin'], py=ras.geo['ymax'], radius=5) == ras.nodata
        assert ras.extract(px=ras.geo['xmax'], py=ras.geo['ymin'], radius=5) == ras.nodata


def test_Raster_filestack(testdata):
    with pytest.raises(RuntimeError):
        with Raster([testdata['tif']]) as ras:
            print(ras)
    with Raster([testdata['tif'], testdata['tif2']]) as ras:
        assert ras.bands == 2
        arr = ras.array()
    mean = parallel_apply_along_axis(np.nanmean, axis=2, arr=arr, cores=4)
    assert mean.shape == (217, 268)


def test_dtypes():
    assert Dtype('Float32').gdalint == 6
    assert Dtype(6).gdalstr == 'Float32'
    assert Dtype('uint32').gdalstr == 'UInt32'
    with pytest.raises(ValueError):
        Dtype('foobar')
    with pytest.raises(ValueError):
        Dtype(999)
    with pytest.raises(TypeError):
        Dtype(None)


def test_stack(tmpdir, testdata):
    name = testdata['tif']
    outname = os.path.join(str(tmpdir), 'test')
    tr = (30, 30)
    # no input files provided
    with pytest.raises(RuntimeError):
        stack(srcfiles=[], resampling='near', targetres=tr,
              srcnodata=-99, dstnodata=-99, dstfile=outname)
    
    # two files, but only one layer name
    with pytest.raises(RuntimeError):
        stack(srcfiles=[name, name], resampling='near', targetres=tr,
              srcnodata=-99, dstnodata=-99, dstfile=outname, layernames=['a'])
    
    # targetres must be a two-entry tuple/list
    with pytest.raises(RuntimeError):
        stack(srcfiles=[name, name], resampling='near', targetres=30,
              srcnodata=-99, dstnodata=-99, dstfile=outname)
    
    # only one file specified
    with pytest.raises(RuntimeError):
        stack(srcfiles=[name], resampling='near', targetres=tr, overwrite=True,
              srcnodata=-99, dstnodata=-99, dstfile=outname)
    
    # targetres must contain two values
    with pytest.raises(RuntimeError):
        stack(srcfiles=[name, name], resampling='near', targetres=(30, 30, 30),
              srcnodata=-99, dstnodata=-99, dstfile=outname)
    
    # unknown resampling method
    with pytest.raises(RuntimeError):
        stack(srcfiles=[name, name], resampling='foobar', targetres=tr,
              srcnodata=-99, dstnodata=-99, dstfile=outname)
    
    # non-existing files
    with pytest.raises(RuntimeError):
        stack(srcfiles=['foo', 'bar'], resampling='near', targetres=tr,
              srcnodata=-99, dstnodata=-99, dstfile=outname)
    
    # create a multi-band stack
    stack(srcfiles=[name, name], resampling='near', targetres=tr, overwrite=True,
          srcnodata=-99, dstnodata=-99, dstfile=outname, layernames=['test1', 'test2'])
    with Raster(outname) as ras:
        assert ras.bands == 2
        # Raster.rescale currently only supports one band
        with pytest.raises(ValueError):
            ras.rescale(lambda x: x * 10)
    
    # outname exists and overwrite is False
    with pytest.raises(RuntimeError):
        stack(srcfiles=[name, name], resampling='near', targetres=tr, overwrite=False,
              srcnodata=-99, dstnodata=-99, dstfile=outname, layernames=['test1', 'test2'])
    
    # pass shapefile
    outname = os.path.join(str(tmpdir), 'test2')
    with Raster(name).bbox() as box:
        stack(srcfiles=[name, name], resampling='near', targetres=tr, overwrite=True,
              srcnodata=-99, dstnodata=-99, dstfile=outname, shapefile=box, layernames=['test1', 'test2'])
    with Raster(outname) as ras:
        assert ras.bands == 2
    
    # pass shapefile and do mosaicing
    outname = os.path.join(str(tmpdir), 'test3')
    with Raster(name).bbox() as box:
        stack(srcfiles=[[name, name]], resampling='near', targetres=tr, overwrite=True,
              srcnodata=-99, dstnodata=-99, dstfile=outname, shapefile=box)
    with Raster(outname + '.tif') as ras:
        assert ras.bands == 1
        assert ras.format == 'GTiff'
    
    # projection mismatch
    name2 = os.path.join(str(tmpdir), os.path.basename(name))
    outname = os.path.join(str(tmpdir), 'test4')
    gdalwarp(name, name2, options={'dstSRS': crsConvert(4326, 'wkt')})
    with pytest.raises(RuntimeError):
        stack(srcfiles=[name, name2], resampling='near', targetres=tr, overwrite=True,
              srcnodata=-99, dstnodata=-99, dstfile=outname)
    
    # no projection found
    outname = os.path.join(str(tmpdir), 'test5')
    gdal_translate(name, name2, {'options': ['-co', 'PROFILE=BASELINE']})
    with Raster(name2) as ras:
        print(ras.projection)
    with pytest.raises(RuntimeError):
        stack(srcfiles=[name2, name2], resampling='near', targetres=tr, overwrite=True,
              srcnodata=-99, dstnodata=-99, dstfile=outname)
    
    # create separate GeoTiffs
    outdir = os.path.join(str(tmpdir), 'subdir')
    stack(srcfiles=[name, name], resampling='near', targetres=tr, overwrite=True, layernames=['test1', 'test2'],
          srcnodata=-99, dstnodata=-99, dstfile=outdir, separate=True, compress=True)
    
    # repeat with overwrite disabled (no error raised, just a print message)
    stack(srcfiles=[name, name], resampling='near', targetres=tr, overwrite=False, layernames=['test1', 'test2'],
          srcnodata=-99, dstnodata=-99, dstfile=outdir, separate=True, compress=True)
    
    # repeat without layernames but sortfun
    # bandnames not unique
    outdir = os.path.join(str(tmpdir), 'subdir2')
    with pytest.raises(RuntimeError):
        stack(srcfiles=[name, name], resampling='near', targetres=tr, overwrite=True, sortfun=os.path.basename,
              srcnodata=-99, dstnodata=-99, dstfile=outdir, separate=True, compress=True)
    
    # repeat without layernames but sortfun
    name2 = os.path.join(str(tmpdir), os.path.basename(name).replace('VV', 'XX'))
    shutil.copyfile(name, name2)
    outdir = os.path.join(str(tmpdir), 'subdir2')
    stack(srcfiles=[name, name2], resampling='near', targetres=tr, overwrite=True, sortfun=os.path.basename,
          srcnodata=-99, dstnodata=-99, dstfile=outdir, separate=True, compress=True)
    
    # shapefile filtering
    outdir = os.path.join(str(tmpdir), 'subdir3')
    files = [testdata['tif'], testdata['tif2'], testdata['tif3']]
    with Raster(files[0]).bbox() as box:
        stack(srcfiles=files, resampling='near', targetres=(30, 30),
              overwrite=False, layernames=['test1', 'test2', 'test3'],
              srcnodata=-99, dstnodata=-99, dstfile=outdir,
              separate=True, compress=True, shapefile=box)
        # repeated run with different scene selection and only one scene after spatial filtering
        stack(srcfiles=files[1:], resampling='near', targetres=(30, 30),
              overwrite=True, layernames=['test2', 'test3'],
              srcnodata=-99, dstnodata=-99, dstfile=outdir,
              separate=True, compress=True, shapefile=box)


def test_auxil(tmpdir, testdata):
    dir = str(tmpdir)
    with Raster(testdata['tif']) as ras:
        bbox = os.path.join(dir, 'bbox.shp')
        ras.bbox(bbox)
        ogr2ogr(bbox, os.path.join(dir, 'bbox.gml'), {'format': 'GML'})
        gdal_translate(ras.raster, os.path.join(dir, 'test'), {'format': 'ENVI'})
    gdal_rasterize(bbox, os.path.join(dir, 'test2'), {'format': 'GTiff', 'xRes': 20, 'yRes': 20})


def test_auxil_coordinate_reproject():
    point = coordinate_reproject(x=11, y=51, s_crs=4326, t_crs=32632)
    assert round(point[0], 3) == 640333.296
    assert round(point[1], 3) == 5651728.683


def test_auxil_cmap_mpl2gdal():
    cmap = cmap_mpl2gdal(mplcolor='YlGnBu', values=range(0, 100))
    assert type(cmap) == gdal.ColorTable


def test_rasterize(tmpdir, testdata):
    outname = os.path.join(str(tmpdir), 'test.shp')
    with Raster(testdata['tif']) as ras:
        vec = ras.bbox()
        
        # test length mismatch between burn_values and expressions
        with pytest.raises(RuntimeError):
            rasterize(vec, reference=ras, outname=outname, burn_values=[1], expressions=['foo', 'bar'])
        
        # test a faulty expression
        with pytest.raises(RuntimeError):
            rasterize(vec, reference=ras, outname=outname, burn_values=[1], expressions=['foo'])
        
        # test default parametrization
        rasterize(vec, reference=ras, outname=outname)
        assert os.path.isfile(outname)
        
        # test appending to existing file with valid expression
        rasterize(vec, reference=ras, outname=outname, append=True, burn_values=[1], expressions=['area=23262400.0'])
        
        # test wrong input type for reference
        with pytest.raises(RuntimeError):
            rasterize(vec, reference='foobar', outname=outname)


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


def test_png(tmpdir, testdata):
    outname = os.path.join(str(tmpdir), 'test')
    with Raster(testdata['tif']) as ras:
        png(src=ras, dst=outname, percent=100, scale=(2, 98), worldfile=True)
    assert os.path.isfile(outname + '.png')
    
    with pytest.raises(TypeError):
        png(src=testdata['tif'], dst=outname, percent=100, scale=(2, 98), worldfile=True)

    src = [testdata['tif'], testdata['tif2']]
    with pytest.raises(ValueError):
        with Raster(src) as ras:
            png(src=ras, dst=outname, percent=100, scale=(2, 98), worldfile=True)
    
    src.append(testdata['tif3'])
    outname = os.path.join(str(tmpdir), 'test_rgb.png')
    with Raster(src) as ras:
        png(src=ras, dst=outname, percent=100, scale=(2, 98), worldfile=True)
