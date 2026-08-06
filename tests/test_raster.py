import os
import pytest
import numpy as np
from spatialist.raster import Dtype, Raster, png, rasterize
from spatialist.vector import bbox
from spatialist.ancillary import parallel_apply_along_axis


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

def test_Raster(tmpdir, testdata):
    with pytest.raises(RuntimeError):
        ras = Raster(1)
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
    with Raster(testdata['tif'], driver='GTiff') as ras:
        print(ras, " with GTiff driver")
        assert ras.bands == 1
    
    with Raster(testdata['tif'], driver=['ENVI', 'GTiff']) as ras:
        print(ras, " with ['ENVI','GTiff'] driver list")
        assert ras.bands == 1


def test_Raster_subset(testdata):
    with Raster(testdata['tif']) as ras:
        ext = ras.bbox().extent
        xres, yres = ras.res
        ext['xmin'] += xres
        ext['xmax'] -= xres
        ext['ymin'] += yres
        ext['ymax'] -= yres
        with bbox(ext, ras.projection) as vec:
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
        ras = Raster([testdata['tif']])
    with Raster([testdata['tif'], testdata['tif2']]) as ras:
        assert ras.bands == 2
        arr = ras.array()
    mean = parallel_apply_along_axis(np.nanmean, axis=2, arr=arr, cores=4)
    assert mean.shape == (217, 268)


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
