#################################################################
# GDAL wrapper for convenient raster data handling and processing
# John Truckenbrodt 2015-2019
#################################################################


# todo: function to write data with the same metadata as a given file

from __future__ import division
import os
import re
import platform
import warnings
import tempfile
from math import sqrt, floor, ceil
from time import gmtime, strftime
import numpy as np

from . import envi
from .auxil import gdalwarp, gdalbuildvrt
from .vector import Vector, bbox, crsConvert, intersect
from .ancillary import dissolve, multicore
from .envi import HDRobject

from osgeo import gdal, gdal_array, osr
from osgeo.gdalconst import GA_ReadOnly, GA_Update

os.environ['GDAL_PAM_PROXY_DIR'] = tempfile.gettempdir()

gdal.UseExceptions()

subset_tolerance = 0  # percent
"""
this parameter can be set to increase the pixel tolerance in percent when subsetting
:class:`Raster` objects with the extent of other spatial objects.

Examples
--------
| Coordinates are in EPSG:32632, pixel resolution of the image to be subsetted is 90 m:
| (subsetting extent)
| {'xmin': 534093.341, 'xmax': 830103.341, 'ymin': 5030609.645, 'ymax': 5250929.645}
| subset_tolerance = 0
| {'xmin': 534003.341, 'xmax': 830103.341, 'ymin': 5030519.645, 'ymax': 5250929.645}
| subset_tolerance = 0.02
| {'xmin': 534093.341, 'xmax': 830103.341, 'ymin': 5030609.645, 'ymax': 5250929.645}
"""


class Raster(object):
    """
    This is intended as a raster meta information handler with options for reading and writing raster data in a
    convenient manner by simplifying the numerous options provided by the `GDAL <http://www.gdal.org/>`_ python binding.
    Several methods are provided along with this class to directly modify the raster object in memory or directly
    write a newly created file to disk (without modifying the raster object itself).
    Upon initializing a Raster object, only metadata is loaded. The actual data can be, for example,
    loaded to memory by calling methods :meth:`matrix` or :meth:`load`.

    Parameters
    ----------
    filename: str, list or :osgeo:class:`gdal.Dataset`
        the raster file(s)/object to read
    list_separate: bool
        treat a list of files as separate layers or otherwise as a single layer? The former is intended for single
        layers of a stack, the latter for tiles of a mosaic.
    """
    
    # todo: init a Raster object from array data not only from a filename
    def __init__(self, filename, list_separate=True):
        if isinstance(filename, gdal.Dataset):
            self.raster = filename
            self.filename = self.files[0] if self.files is not None else None
        elif isinstance(filename, str):
            self.filename = filename if os.path.isabs(filename) else os.path.join(os.getcwd(), filename)
            self.raster = gdal.Open(filename, GA_ReadOnly)
        elif isinstance(filename, list):
            self.raster = gdalbuildvrt(src=filename,
                                       dst=tempfile.NamedTemporaryFile(suffix='.vrt').name,
                                       options={'separate': list_separate},
                                       void=False)
        else:
            raise RuntimeError('raster input must be of type str, list or gdal.Dataset')
        
        # a list to contain arrays
        self.__data = [None] * self.bands
        
        if self.format == 'ENVI':
            with HDRobject(self.filename + '.hdr') as hdr:
                if hasattr(hdr, 'band_names'):
                    self.bandnames = hdr.band_names
                else:
                    self.bandnames = ['band{}'.format(x) for x in range(1, self.bands + 1)]
        elif self.format == 'VRT':
            vrt_tilenames = [os.path.splitext(os.path.basename(x))[0] for x in self.files]
            if len(vrt_tilenames) == self.bands:
                self.bandnames = vrt_tilenames
            elif self.bands == 1:
                self.bandnames = ['mosaic']
        else:
            self.bandnames = ['band{}'.format(x) for x in range(1, self.bands + 1)]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __str__(self):
        vals = dict()
        vals['rows'], vals['cols'], vals['bands'] = self.dim
        vals.update(self.geo)
        vals['proj4'] = self.proj4
        vals['filename'] = self.filename if self.filename is not None else 'memory'
        
        info = 'class      : spatialist Raster object\n' \
               'dimensions : {rows}, {cols}, {bands} (rows, cols, bands)\n' \
               'resolution : {xres}, {yres} (x, y)\n' \
               'extent     : {xmin}, {xmax}, {ymin}, {ymax} (xmin, xmax, ymin, ymax)\n' \
               'coord. ref.: {proj4}\n' \
               'data source: {filename}'.format(**vals)
        return info
    
    def __getitem__(self, index):
        """
        subset the object by slices or vector geometry. If slices are provided, one slice for each raster dimension
        needs to be defined. I.e., if the raster object contains several image bands, three slices are necessary. If a
        :class:`~spatialist.vector.Vector` geometry is defined, it is internally projected to the raster CRS if necessary, its extent
        derived and the extent converted to raster pixel slices, which are then used for subsetting.

        Parameters
        ----------
        index: :obj:`tuple` of :obj:`slice` or :obj:`~spatialist.vector.Vector`
            the subsetting indices to be used

        Returns
        -------
        Raster
            a new raster object referenced through an in-memory GDAL VRT file
        """
        # subsetting via Vector object
        if isinstance(index, Vector):
            # reproject vector object on the fly
            index.reproject(self.proj4)
            # intersect vector object with raster bounding box
            inter = intersect(self.bbox(), index)
            if inter is None:
                raise RuntimeError('no intersection between Raster and Vector object')
            # get raster indexing slices from intersect bounding box extent
            sl = self.__extent2slice(inter.extent)
            # subset raster object with slices
            with self[sl] as sub:
                # mask subsetted raster object with vector geometries
                masked = sub.__maskbyvector(inter)
            inter = None
            return masked
        #####################################################################################################
        # subsetting via slices
        if isinstance(index, tuple):
            ras_dim = 2 if self.raster.RasterCount == 1 else 3
            if ras_dim != len(index):
                raise IndexError(
                    'mismatch of index length ({0}) and raster dimensions ({1})'.format(len(index), ras_dim))
            for i in [0, 1]:
                if index[i].step is not None:
                    raise IndexError('step slicing of {} is not allowed'.format(['rows', 'bands'][i]))
        
        # create index lists from subset slices
        subset = dict()
        subset['rows'] = list(range(0, self.rows))[index[0]]
        subset['cols'] = list(range(0, self.cols))[index[1]]
        if len(index) > 2:
            subset['bands'] = list(range(0, self.bands))[index[2]]
            if not isinstance(subset['bands'], list):
                subset['bands'] = [subset['bands']]
        else:
            subset['bands'] = [0]
        
        if len(subset['rows']) == 0 or len(subset['cols']) == 0 or len(subset['bands']) == 0:
            raise RuntimeError('no suitable subset for defined slice:\n  {}'.format(index))
        
        # update geo dimensions from subset list indices
        geo = self.geo
        geo['xmin'] = geo['xmin'] + min(subset['cols']) * geo['xres']
        geo['ymax'] = geo['ymax'] - min(subset['rows']) * abs(geo['yres'])
        
        # note: yres is negative!
        geo['xmax'] = geo['xmin'] + geo['xres'] * len(subset['cols'])
        geo['ymin'] = geo['ymax'] + geo['yres'] * len(subset['rows'])
        
        # create options for creating a GDAL VRT dataset
        opts = dict()
        opts['xRes'], opts['yRes'] = self.res
        opts['outputSRS'] = self.projection
        opts['srcNodata'] = self.nodata
        opts['VRTNodata'] = self.nodata
        opts['bandList'] = [x + 1 for x in subset['bands']]
        opts['outputBounds'] = (geo['xmin'], geo['ymin'], geo['xmax'], geo['ymax'])
        
        # create an in-memory VRT file and return the output raster dataset as new Raster object
        outname = os.path.join('/vsimem/', os.path.basename(tempfile.mktemp()))
        out_ds = gdalbuildvrt(src=self.filename, dst=outname, options=opts, void=False)
        out = Raster(out_ds)
        if len(index) > 2:
            bandnames = self.bandnames[index[2]]
        else:
            bandnames = self.bandnames
        if not isinstance(bandnames, list):
            bandnames = [bandnames]
        out.bandnames = bandnames
        return out
    
    def __extent2slice(self, extent):
        extent_bbox = bbox(extent, self.proj4)
        inter = intersect(self.bbox(), extent_bbox)
        extent_bbox.close()
        if inter:
            ext_inter = inter.extent
            ext_ras = self.geo
            xres, yres = self.res
            tolerance_x = xres * subset_tolerance / 100
            tolerance_y = yres * subset_tolerance / 100
            colmin = int(floor((ext_inter['xmin'] - ext_ras['xmin'] + tolerance_x) / xres))
            colmax = int(ceil((ext_inter['xmax'] - ext_ras['xmin'] - tolerance_x) / xres))
            rowmin = int(floor((ext_ras['ymax'] - ext_inter['ymax'] + tolerance_y) / yres))
            rowmax = int(ceil((ext_ras['ymax'] - ext_inter['ymin'] - tolerance_y) / yres))
            inter.close()
            if self.bands == 1:
                return slice(rowmin, rowmax), slice(colmin, colmax)
            else:
                return slice(rowmin, rowmax), slice(colmin, colmax), slice(0, self.bands)
        else:
            raise RuntimeError('extent does not overlap with raster object')
    
    def __maskbyvector(self, vec, outname=None, format='GTiff', nodata=0):
        
        if outname is not None:
            driver_name = format
        else:
            driver_name = 'MEM'
        
        with rasterize(vec, self) as vecmask:
            mask = vecmask.matrix()
        
        driver = gdal.GetDriverByName(driver_name)
        outDataset = driver.Create(outname if outname is not None else '',
                                   self.cols, self.rows, self.bands, Dtype(self.dtype).gdalint)
        driver = None
        outDataset.SetMetadata(self.raster.GetMetadata())
        outDataset.SetGeoTransform([self.geo[x] for x in ['xmin', 'xres', 'rotation_x', 'ymax', 'rotation_y', 'yres']])
        if self.projection is not None:
            outDataset.SetProjection(self.projection)
        for i in range(1, self.bands + 1):
            outband = outDataset.GetRasterBand(i)
            outband.SetNoDataValue(nodata)
            mat = self.matrix(band=i)
            mat = mat * mask
            outband.WriteArray(mat)
            del mat
            outband.FlushCache()
            outband = None
        if outname is not None:
            if format == 'GTiff':
                outDataset.SetMetadataItem('TIFFTAG_DATETIME', strftime('%Y:%m:%d %H:%M:%S', gmtime()))
            elif format == 'ENVI':
                with HDRobject(outname + '.hdr') as hdr:
                    hdr.band_names = self.bandnames
                    hdr.write()
            outDataset = None
        else:
            out = Raster(outDataset)
            out.bandnames = self.bandnames
            return out
    
    def allstats(self, approximate=False):
        """
        Compute some basic raster statistics

        Parameters
        ----------
        approximate: bool
            approximate statistics from overviews or a subset of all tiles?

        Returns
        -------
        list of dicts
            a list with a dictionary of statistics for each band. Keys: `min`, `max`, `mean`, `sdev`.
            See :osgeo:meth:`gdal.Band.ComputeStatistics`.
        """
        statcollect = []
        for x in self.layers():
            try:
                stats = x.ComputeStatistics(approximate)
            except RuntimeError:
                stats = None
            stats = dict(zip(['min', 'max', 'mean', 'sdev'], stats))
            statcollect.append(stats)
        return statcollect
    
    def array(self):
        """
        read all raster bands into a numpy ndarray

        Returns
        -------
        numpy.ndarray
            the array containing all raster data
        """
        if self.bands == 1:
            return self.matrix()
        else:
            arr = self.raster.ReadAsArray().transpose(1, 2, 0)
            if isinstance(self.nodata, list):
                for i in range(0, self.bands):
                    arr[:, :, i][arr[:, :, i] == self.nodata[i]] = np.nan
            else:
                arr[arr == self.nodata] = np.nan
            return arr
    
    def assign(self, array, band):
        """
        assign an array to an existing Raster object

        Parameters
        ----------
        array: numpy.ndarray
            the array to be assigned to the Raster object
        band: int
            the index of the band to assign to

        Returns
        -------

        """
        self.__data[band] = array
    
    @property
    def bands(self):
        """

        Returns
        -------
        int
            the number of image bands
        """
        return self.raster.RasterCount
    
    @property
    def bandnames(self):
        """

        Returns
        -------
        list
            the names of the bands
        """
        return self.__bandnames
    
    @bandnames.setter
    def bandnames(self, names):
        """
        set the names of the raster bands

        Parameters
        ----------
        names: list of str
            the names to be set; must be of same length as the number of bands

        Returns
        -------

        """
        if not isinstance(names, list):
            raise TypeError('the names to be set must be of type list')
        if len(names) != self.bands:
            raise ValueError(
                'length mismatch of names to be set ({}) and number of bands ({})'.format(len(names), self.bands))
        self.__bandnames = names
    
    def bbox(self, outname=None, driver=None, overwrite=True):
        """
        Parameters
        ----------
        outname: str or None
            the name of the file to write; If `None`, the bounding box is returned as vector object
        format: str
            The file format to write
        overwrite: bool
            overwrite an already existing file?

        Returns
        -------
        Vector or None
            the bounding box vector object
        """
        if outname is None:
            return bbox(self.geo, self.proj4)
        else:
            bbox(self.geo, self.proj4, outname=outname, driver=driver, overwrite=overwrite)
    
    def close(self):
        """
        closes the GDAL raster file connection
        Returns
        -------

        """
        self.raster = None
    
    @property
    def cols(self):
        """

        Returns
        -------
        int
            the number of image columns
        """
        return self.raster.RasterXSize
    
    @property
    def dim(self):
        """

        Returns
        -------
        tuple
            (rows, columns, bands)
        """
        return (self.rows, self.cols, self.bands)
    
    @property
    def driver(self):
        """

        Returns
        -------
        :osgeo:class:`gdal.Driver`
            a GDAL raster driver object.
        """
        return self.raster.GetDriver()
    
    @property
    def dtype(self):
        """

        Returns
        -------
        str
            the data type description; e.g. `Float32`
        """
        return gdal.GetDataTypeName(self.raster.GetRasterBand(1).DataType)
    
    @property
    def epsg(self):
        """

        Returns
        -------
        int
            the CRS EPSG code
        """
        return crsConvert(self.projection, 'epsg')
    
    @property
    def extent(self):
        """
        
        Returns
        -------
        dict
            the extent of the image
        """
        return {key: self.geo[key] for key in ['xmin', 'xmax', 'ymin', 'ymax']}
    
    def extract(self, px, py, radius=1, nodata=None):
        """
        extract weighted average of pixels intersecting with a defined radius to a point.

        Parameters
        ----------
        px: int or float
            the x coordinate in units of the Raster SRS
        py: int or float
            the y coordinate in units of the Raster SRS
        radius: int or float
            the radius around the point to extract pixel values from; defined as multiples of the pixel resolution
        nodata: int
            a value to ignore from the computations; If `None`, the nodata value of the Raster object is used

        Returns
        -------
        int or float
            the the weighted average of all pixels within the defined radius

        """
        if not self.geo['xmin'] <= px <= self.geo['xmax']:
            raise RuntimeError('px is out of bounds')
        
        if not self.geo['ymin'] <= py <= self.geo['ymax']:
            raise RuntimeError('py is out of bounds')
        
        if nodata is None:
            nodata = self.nodata
        
        xres, yres = self.res
        
        hx = xres / 2.0
        hy = yres / 2.0
        
        xlim = float(xres * radius)
        ylim = float(yres * radius)
        
        # compute minimum x and y pixel coordinates
        xmin = int(floor((px - self.geo['xmin'] - xlim) / xres))
        ymin = int(floor((self.geo['ymax'] - py - ylim) / yres))
        
        xmin = xmin if xmin >= 0 else 0
        ymin = ymin if ymin >= 0 else 0
        
        # compute maximum x and y pixel coordinates
        xmax = int(ceil((px - self.geo['xmin'] + xlim) / xres))
        ymax = int(ceil((self.geo['ymax'] - py + ylim) / yres))
        
        xmax = xmax if xmax <= self.cols else self.cols
        ymax = ymax if ymax <= self.rows else self.rows
        
        # load array subset
        if self.__data[0] is not None:
            array = self.__data[0][ymin:ymax, xmin:xmax]
            # print('using loaded array of size {}, '
            #       'indices [{}:{}, {}:{}] (row/y, col/x)'.format(array.shape, ymin, ymax, xmin, xmax))
        else:
            array = self.raster.GetRasterBand(1).ReadAsArray(xmin, ymin, xmax - xmin, ymax - ymin)
            # print('loading array of size {}, '
            #       'indices [{}:{}, {}:{}] (row/y, col/x)'.format(array.shape, ymin, ymax, xmin, xmax))
        
        sum = 0
        counter = 0
        weightsum = 0
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                # check whether point is a valid image index
                val = array[y - ymin, x - xmin]
                if val != nodata:
                    # compute distances of pixel center coordinate to requested point
                    
                    xc = x * xres + hx + self.geo['xmin']
                    yc = self.geo['ymax'] - y * yres + hy
                    
                    dx = abs(xc - px)
                    dy = abs(yc - py)
                    
                    # check whether point lies within ellipse: if ((dx ** 2) / xlim ** 2) + ((dy ** 2) / ylim ** 2) <= 1
                    weight = sqrt(dx ** 2 + dy ** 2)
                    sum += val * weight
                    weightsum += weight
                    counter += 1
        
        array = None
        
        if counter > 0:
            return sum / weightsum
        else:
            return nodata
    
    @property
    def files(self):
        """

        Returns
        -------
        list of str
            a list of all absolute names of files associated with this raster data set
        """
        fl = self.raster.GetFileList()
        if fl is not None:
            return [os.path.abspath(x) for x in fl]
    
    @property
    def format(self):
        """

        Returns
        -------
        str
            the name of the image format
        """
        return self.driver.ShortName
    
    @property
    def geo(self):
        """
        General image geo information.

        Returns
        -------
        dict
            a dictionary with keys `xmin`, `xmax`, `xres`, `rotation_x`, `ymin`, `ymax`, `yres`, `rotation_y`
        """
        out = dict(zip(['xmin', 'xres', 'rotation_x', 'ymax', 'rotation_y', 'yres'],
                       self.raster.GetGeoTransform()))
        
        # note: yres is negative!
        out['xmax'] = out['xmin'] + out['xres'] * self.cols
        out['ymin'] = out['ymax'] + out['yres'] * self.rows
        return out
    
    @property
    def geogcs(self):
        """

        Returns
        -------
        str or None
            an identifier of the geographic coordinate system
        """
        return self.srs.GetAttrValue('geogcs')
    
    def is_valid(self):
        """
        Check image integrity.
        Tries to compute the checksum for each raster layer and returns False if this fails.
        See this forum entry:
        `How to check if image is valid? <https://lists.osgeo.org/pipermail/gdal-dev/2013-November/037520.html>`_.

        Returns
        -------
        bool
            is the file valid?
        """
        for i in range(self.raster.RasterCount):
            try:
                checksum = self.raster.GetRasterBand(i + 1).Checksum()
            except RuntimeError:
                return False
        return True
    
    def layers(self):
        """

        Returns
        -------
        list of :osgeo:class:`gdal.Band`
            a list containing a :osgeo:class:`gdal.Band` object for each image band
        """
        return [self.raster.GetRasterBand(band) for band in range(1, self.bands + 1)]
    
    def load(self):
        """
        load all raster data to internal memory arrays.
        This shortens the read time of other methods like :meth:`matrix`.
        """
        for i in range(1, self.bands + 1):
            self.__data[i - 1] = self.matrix(i)
    
    def matrix(self, band=1, mask_nan=True):
        """
        read a raster band (subset) into a numpy ndarray

        Parameters
        ----------
        band: int
            the band to read the matrix from; 1-based indexing
        mask_nan: bool
            convert nodata values to :obj:`numpy.nan`? As :obj:`numpy.nan` requires at least float values, any integer array is cast
            to float32.

        Returns
        -------
        numpy.ndarray
            the matrix (subset) of the selected band
        """
        
        mat = self.__data[band - 1]
        if mat is None:
            mat = self.raster.GetRasterBand(band).ReadAsArray()
            if mask_nan:
                if isinstance(self.nodata, list):
                    nodata = self.nodata[band - 1]
                else:
                    nodata = self.nodata
                try:
                    mat[mat == nodata] = np.nan
                except ValueError:
                    mat = mat.astype('float32')
                    mat[mat == nodata] = np.nan
        return mat
    
    @property
    def nodata(self):
        """

        Returns
        -------
        float or list
            the raster nodata value(s)
        """
        nodatas = [self.raster.GetRasterBand(i).GetNoDataValue()
                   for i in range(1, self.bands + 1)]
        if len(list(set(nodatas))) == 1:
            return nodatas[0]
        else:
            return nodatas
    
    @property
    def projcs(self):
        """
        Returns
        -------
        str or None
            an identifier of the projected coordinate system; If the CRS is not projected `None` is returned
        """
        return self.srs.GetAttrValue('projcs') if self.srs.IsProjected() else None
    
    @property
    def projection(self):
        """

        Returns
        -------
        str
            the CRS Well Known Text (WKT) description
        """
        return self.raster.GetProjection()
    
    @property
    def proj4(self):
        """

        Returns
        -------
        str
            the CRS PROJ4 description
        """
        try:
            return crsConvert(self.projection, 'proj4')
        except TypeError:
            return None
    
    @property
    def proj4args(self):
        """

        Returns
        -------
        dict
            the PROJ4 string arguments as a dictionary
        """
        args = [x.split('=') for x in re.split('[+ ]+', self.proj4) if len(x) > 0]
        return dict([(x[0], None) if len(x) == 1 else tuple(x) for x in args])
    
    @property
    def res(self):
        """
        the raster resolution in x and y direction

        Returns
        -------
        tuple
            (xres, yres)
        """
        return (abs(float(self.geo['xres'])), abs(float(self.geo['yres'])))
    
    def rescale(self, fun):
        """
        perform raster computations with custom functions and assign them to the existing raster object in memory

        Parameters
        ----------
        fun: function
            the custom function to compute on the data

        Examples
        --------
        >>> with Raster('filename') as ras:
        >>>     ras.rescale(lambda x: 10 * x)

        """
        if self.bands != 1:
            raise ValueError('only single band images are currently supported')
        
        # load array
        mat = self.matrix()
        
        # scale values
        scaled = fun(mat)
        
        # assign newly computed array to raster object
        self.assign(scaled, band=0)
    
    @property
    def rows(self):
        """

        Returns
        -------
        int
            the number of image rows
        """
        return self.raster.RasterYSize
    
    @property
    def srs(self):
        """

        Returns
        -------
        :osgeo:class:`osr.SpatialReference`
            the spatial reference system of the data set.
        """
        return osr.SpatialReference(wkt=self.projection)
    
    def write(self, outname, dtype='default', format='ENVI', nodata='default', compress_tif=False, overwrite=False):
        """
        write the raster object to a file.

        Parameters
        ----------
        outname: str
            the file to be written
        dtype: str
            the data type of the written file;
            data type notations of GDAL (e.g. `Float32`) and numpy (e.g. `int8`) are supported.
        format: str
            the file format; e.g. 'GTiff'
        nodata: int or float
            the nodata value to write to the file
        compress_tif: bool
            if the format is GeoTiff, compress the written file?
        overwrite: bool
            overwrite an already existing file?

        Returns
        -------

        """
        
        if os.path.isfile(outname) and not overwrite:
            raise RuntimeError('target file already exists')
        
        if format == 'GTiff' and not re.search(r'\.tif[f]*$', outname):
            outname += '.tif'
        
        dtype = Dtype(self.dtype if dtype == 'default' else dtype).gdalint
        nodata = self.nodata if nodata == 'default' else nodata
        
        options = []
        if format == 'GTiff' and compress_tif:
            options += ['COMPRESS=DEFLATE', 'PREDICTOR=2']
        
        driver = gdal.GetDriverByName(format)
        outDataset = driver.Create(outname, self.cols, self.rows, self.bands, dtype, options)
        driver = None
        outDataset.SetMetadata(self.raster.GetMetadata())
        outDataset.SetGeoTransform([self.geo[x] for x in ['xmin', 'xres', 'rotation_x', 'ymax', 'rotation_y', 'yres']])
        if self.projection is not None:
            outDataset.SetProjection(self.projection)
        for i in range(1, self.bands + 1):
            outband = outDataset.GetRasterBand(i)
            if nodata is not None:
                outband.SetNoDataValue(nodata)
            mat = self.matrix(band=i)
            dtype_mat = str(mat.dtype)
            dtype_ras = Dtype(dtype).numpystr
            if not np.can_cast(dtype_mat, dtype_ras):
                warnings.warn("writing band {}: unsafe casting from type {} to {}".format(i, dtype_mat, dtype_ras))
                if nodata is not None:
                    print('converting nan to nodata value {}'.format(nodata))
                    mat[np.isnan(mat)] = nodata
                    mat = mat.astype(dtype_ras)
            outband.WriteArray(mat)
            del mat
            outband.FlushCache()
            outband = None
        if format == 'GTiff':
            outDataset.SetMetadataItem('TIFFTAG_DATETIME', strftime('%Y:%m:%d %H:%M:%S', gmtime()))
        outDataset = None
        if format == 'ENVI':
            hdrfile = os.path.splitext(outname)[0] + '.hdr'
            with HDRobject(hdrfile) as hdr:
                hdr.band_names = self.bandnames
                hdr.write()
        
        # write a png image of three raster bands (provided in a list of 1-based integers); percent controls the size ratio of input and output
        # def png(self, bands, outname, percent=10):
        #     if len(bands) != 3 or max(bands) not in range(1, self.bands+1) or min(bands) not in range(1, self.bands+1):
        #         print 'band indices out of range'
        #         return
        #     if not outname.endswith('.png'):
        #         outname += '.png'
        #     exp_bands = ' '.join(['-b '+str(x) for x in bands]).split()
        #     exp_scale = [['-scale', self.getstat('min', x), self.getstat('max', x), 0, 255] for x in bands]
        #     exp_size = ['-outsize', str(percent)+'%', str(percent)+'%']
        #     cmd = dissolve([['gdal_translate', '-q', '-of', 'PNG', '-ot', 'Byte'], exp_size, exp_bands, exp_scale, self.filename, outname])
        #     sp.check_call([str(x) for x in cmd])
    
    # def reduce(self, outname=None, format='ENVI'):
    #     """
    #     remove all lines and columns containing only no data values
    #
    #     Args:
    #         outname:
    #         format:
    #
    #     Returns:
    #
    #     """
    #     if self.bands != 1:
    #         raise ValueError('only single band images supported')
    #
    #     stats = self.allstats[0]
    #
    #     if stats[0] == stats[1]:
    #         raise ValueError('file does not contain valid pixels')
    #
    #     # load raster layer into an array
    #     mat = self.matrix()
    #
    #     mask1 = ~np.all(mat == self.nodata, axis=0)
    #     mask2 = ~np.all(mat == self.nodata, axis=1)
    #     mask1_l = mask1.tolist()
    #     mask2_l = mask2.tolist()
    #
    #     left = mask1_l.index(True)
    #     cols = len(mask1_l) - mask1_l[::-1].index(True) - left
    #     top = mask2_l.index(True)
    #     rows = len(mask2_l) - mask2_l[::-1].index(True) - top
    #
    #     mat = mat[mask2, :]
    #     mat = mat[:, mask1]
    #
    #     if outname is None:
    #         self.assign(mat, dim=[left, top, cols, rows], index=0)
    #     else:
    #         self.write(outname, dim=[left, top, cols, rows], format=format)


def rasterize(vectorobject, reference, outname=None, burn_values=1, expressions=None, nodata=0, append=False):
    """
    rasterize a vector object

    Parameters
    ----------
    vectorobject: Vector
        the vector object to be rasterized
    reference: Raster
        a reference Raster object to retrieve geo information and extent from
    outname: str or None
        the name of the GeoTiff output file; if None, an in-memory object of type :class:`Raster` is returned and
        parameter outname is ignored
    burn_values: int or list
        the values to be written to the raster file
    expressions: list
        SQL expressions to filter the vector object by attributes
    nodata: int
        the nodata value of the target raster file
    append: bool
        if the output file already exists, update this file with new rasterized values?
        If True and the output file exists, parameters `reference` and `nodata` are ignored.

    Returns
    -------
    Raster or None
        if outname is `None`, a raster object pointing to an in-memory dataset else `None`
    Example
    -------
    >>> from spatialist import Vector, Raster, rasterize
    >>> outname1 = 'target1.tif'
    >>> outname2 = 'target2.tif'
    >>> with Vector('source.shp') as vec:
    >>>     with Raster('reference.tif') as ref:
    >>>         burn_values = [1, 2]
    >>>         expressions = ['ATTRIBUTE=1', 'ATTRIBUTE=2']
    >>>         rasterize(vec, reference, outname1, burn_values, expressions)
    >>>         expressions = ["ATTRIBUTE2='a'", "ATTRIBUTE2='b'"]
    >>>         rasterize(vec, reference, outname2, burn_values, expressions)
    """
    if expressions is None:
        expressions = ['']
    if isinstance(burn_values, (int, float)):
        burn_values = [burn_values]
    if len(expressions) != len(burn_values):
        raise RuntimeError('expressions and burn_values of different length')
    
    failed = []
    for exp in expressions:
        try:
            vectorobject.layer.SetAttributeFilter(exp)
        except RuntimeError:
            failed.append(exp)
    if len(failed) > 0:
        raise RuntimeError('failed to set the following attribute filter(s): ["{}"]'.format('", '.join(failed)))
    
    if append and outname is not None and os.path.isfile(outname):
        target_ds = gdal.Open(outname, GA_Update)
    else:
        if not isinstance(reference, Raster):
            raise RuntimeError("parameter 'reference' must be of type Raster")
        if outname is not None:
            target_ds = gdal.GetDriverByName('GTiff').Create(outname, reference.cols, reference.rows, 1, gdal.GDT_Byte)
        else:
            target_ds = gdal.GetDriverByName('MEM').Create('', reference.cols, reference.rows, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(reference.raster.GetGeoTransform())
        target_ds.SetProjection(reference.raster.GetProjection())
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(nodata)
        band.FlushCache()
        band = None
    for expression, value in zip(expressions, burn_values):
        vectorobject.layer.SetAttributeFilter(expression)
        gdal.RasterizeLayer(target_ds, [1], vectorobject.layer, burn_values=[value])
    vectorobject.layer.SetAttributeFilter('')
    if outname is None:
        return Raster(target_ds)
    else:
        target_ds = None


def reproject(rasterobject, reference, outname, targetres=None, resampling='bilinear', format='GTiff'):
    """
    reproject a raster file

    Parameters
    ----------
    rasterobject: Raster or str
        the raster image to be reprojected
    reference: Raster, Vector, str, int or osr.SpatialReference
        either a projection string or a spatial object with an attribute 'projection'
    outname: str
        the name of the output file
    targetres: tuple
        the output resolution in the target SRS; a two-entry tuple is required: (xres, yres)
    resampling: str
        the resampling algorithm to be used
    format: str
        the output file format

    Returns
    -------

    """
    if isinstance(rasterobject, str):
        rasterobject = Raster(rasterobject)
    if not isinstance(rasterobject, Raster):
        raise RuntimeError('rasterobject must be of type Raster or str')
    if isinstance(reference, (Raster, Vector)):
        projection = reference.projection
        if targetres is not None:
            xres, yres = targetres
        elif hasattr(reference, 'res'):
            xres, yres = reference.res
        else:
            raise RuntimeError('parameter targetres is missing and cannot be read from the reference')
    elif isinstance(reference, (int, str, osr.SpatialReference)):
        try:
            projection = crsConvert(reference, 'proj4')
        except TypeError:
            raise RuntimeError('reference projection cannot be read')
        if targetres is None:
            raise RuntimeError('parameter targetres is missing and cannot be read from the reference')
        else:
            xres, yres = targetres
    else:
        raise TypeError('reference must be of type Raster, Vector, osr.SpatialReference, str or int')
    options = {'format': format,
               'resampleAlg': resampling,
               'xRes': xres,
               'yRes': yres,
               'srcNodata': rasterobject.nodata,
               'dstNodata': rasterobject.nodata,
               'dstSRS': projection}
    gdalwarp(rasterobject, outname, options)


# todo improve speed until aborting when all target files already exist
def stack(srcfiles, dstfile, resampling, targetres, dstnodata, srcnodata=None, shapefile=None, layernames=None,
          sortfun=None, separate=False, overwrite=False, compress=True, cores=4):
    """
    function for mosaicking, resampling and stacking of multiple raster files into a 3D data cube

    Parameters
    ----------
    srcfiles: list
        a list of file names or a list of lists; each sub-list is treated as a task to mosaic its containing files
    dstfile: str
        the destination file or a directory (if `separate` is True)
    resampling: {near, bilinear, cubic, cubicspline, lanczos, average, mode, max, min, med, Q1, Q3}
        the resampling method; see `documentation of gdalwarp <https://www.gdal.org/gdalwarp.html>`_.
    targetres: tuple or list
        two entries for x and y spatial resolution in units of the source CRS
    srcnodata: int, float or None
        the nodata value of the source files; if left at the default (None), the nodata values are read from the files
    dstnodata: int or float
        the nodata value of the destination file(s)
    shapefile: str, Vector or None
        a shapefile for defining the spatial extent of the destination files
    layernames: list
        the names of the output layers; if `None`, the basenames of the input files are used; overrides sortfun
    sortfun: function
        a function for sorting the input files; not used if layernames is not None.
        This is first used for sorting the items in each sub-list of srcfiles;
        the basename of the first item in a sub-list will then be used as the name for the mosaic of this group.
        After mosaicing, the function is again used for sorting the names in the final output
        (only relevant if `separate` is False)
    separate: bool
        should the files be written to a single raster stack (ENVI format) or separate files (GTiff format)?
    overwrite: bool
        overwrite the file if it already exists?
    compress: bool
        compress the geotiff files?
    cores: int
        the number of CPU threads to use; this is only relevant if `separate` is True, in which case each
        mosaicing/resampling job is passed to a different CPU

    Returns
    -------

    Notes
    -----
    This function does not reproject any raster files. Thus, the CRS must be the same for all input raster files.
    This is checked prior to executing gdalwarp. In case a shapefile is defined, it is internally reprojected to the
    raster CRS prior to retrieving its extent.
    
    Examples
    --------
    
    .. code-block:: python
    
        from pyroSAR.ancillary import groupbyTime, find_datasets, seconds
        from spatialist.raster import stack
        
        # find pyroSAR files by metadata attributes
        archive_s1 = '/.../sentinel1/GRD/processed'
        scenes_s1 = find_datasets(archive_s1, sensor=('S1A', 'S1B'), acquisition_mode='IW')
        
        # group images by acquisition time
        groups = groupbyTime(images=scenes_s1, function=seconds, time=30)
        
        # mosaic individual groups and stack the mosaics to a single ENVI file
        # only files overlapping with the shapefile are selected and resampled to its extent
        stack(srcfiles=groups, dstfile='stack', resampling='bilinear', targetres=(20, 20),
              srcnodata=-99, dstnodata=-99, shapefile='site.shp', separate=False)
    """
    # perform some checks on the input data
    
    if len(dissolve(srcfiles)) == 0:
        raise RuntimeError('no input files provided to function raster.stack')
    
    if layernames is not None:
        if len(layernames) != len(srcfiles):
            raise RuntimeError('mismatch between number of source file groups and layernames')
    
    if not isinstance(targetres, (list, tuple)) or len(targetres) != 2:
        raise RuntimeError('targetres must be a list or tuple with two entries for x and y resolution')
    
    if len(srcfiles) == 1 and not isinstance(srcfiles[0], list):
        raise RuntimeError('only one file specified; nothing to be done')
    
    if resampling not in ['near', 'bilinear', 'cubic', 'cubicspline', 'lanczos',
                          'average', 'mode', 'max', 'min', 'med', 'Q1', 'Q3']:
        raise RuntimeError('resampling method not supported')
    
    projections = list()
    for x in dissolve(srcfiles):
        try:
            projection = Raster(x).projection
        except RuntimeError as e:
            print('cannot read file: {}'.format(x))
            raise e
        projections.append(projection)
    
    projections = list(set(projections))
    if len(projections) > 1:
        raise RuntimeError('raster projection mismatch')
    elif projections[0] == '':
        raise RuntimeError('could not retrieve the projection from any of the {} input images'.format(len(srcfiles)))
    else:
        srs = projections[0]
    ##########################################################################################
    # read shapefile bounding coordinates and reduce list of rasters to those overlapping with the shapefile
    
    if shapefile is not None:
        shp = shapefile.clone() if isinstance(shapefile, Vector) else Vector(shapefile)
        shp.reproject(srs)
        ext = shp.extent
        arg_ext = (ext['xmin'], ext['ymin'], ext['xmax'], ext['ymax'])
        for i, item in enumerate(srcfiles):
            group = item if isinstance(item, list) else [item]
            if layernames is None and sortfun is not None:
                group = sorted(group, key=sortfun)
            group = [x for x in group if intersect(shp, Raster(x).bbox())]
            if len(group) > 1:
                srcfiles[i] = group
            elif len(group) == 1:
                srcfiles[i] = group[0]
            else:
                srcfiles[i] = None
        shp.close()
        srcfiles = list(filter(None, srcfiles))
    else:
        arg_ext = None
    ##########################################################################################
    # set general options and parametrization
    
    dst_base = os.path.splitext(dstfile)[0]
    
    options_warp = {'options': ['-q'],
                    'format': 'GTiff' if separate else 'ENVI',
                    'outputBounds': arg_ext, 'multithread': True,
                    'dstNodata': dstnodata,
                    'xRes': targetres[0], 'yRes': targetres[1],
                    'resampleAlg': resampling}
    
    if overwrite:
        options_warp['options'] += ['-overwrite']
    
    if separate and compress:
        options_warp['options'] += ['-co', 'COMPRESS=DEFLATE', '-co', 'PREDICTOR=2']
    
    options_buildvrt = {'outputBounds': arg_ext}
    
    if srcnodata is not None:
        options_warp['srcNodata'] = srcnodata
        options_buildvrt['srcNodata'] = srcnodata
    ##########################################################################################
    # create VRT files for mosaicing
    
    for i, group in enumerate(srcfiles):
        if isinstance(group, list):
            if len(group) > 1:
                base = group[0]
                # in-memory VRT files cannot be shared between multiple processes on Windows
                # this has to do with different process forking behaviour
                # see function spatialist.ancillary.multicore and this link:
                # https://stackoverflow.com/questions/38236211/why-multiprocessing-process-behave-differently-on-windows-and-linux-for-global-o
                vrt_base = os.path.splitext(os.path.basename(base))[0] + '.vrt'
                if platform.system() == 'Windows':
                    vrt = os.path.join(tempfile.gettempdir(), vrt_base)
                else:
                    vrt = '/vsimem/' + vrt_base
                gdalbuildvrt(group, vrt, options_buildvrt)
                srcfiles[i] = vrt
            else:
                srcfiles[i] = group[0]
        else:
            srcfiles[i] = group
    ##########################################################################################
    # define the output band names
    
    # if no specific layernames are defined, sort files by custom function
    if layernames is None and sortfun is not None:
        srcfiles = sorted(srcfiles, key=sortfun)
    
    # use the file basenames without extension as band names if none are defined
    bandnames = [os.path.splitext(os.path.basename(x))[0] for x in srcfiles] if layernames is None else layernames
    
    if len(list(set(bandnames))) != len(bandnames):
        raise RuntimeError('output bandnames are not unique')
    ##########################################################################################
    # create the actual image files
    
    if separate:
        if not os.path.isdir(dstfile):
            os.makedirs(dstfile)
        dstfiles = [os.path.join(dstfile, x) + '.tif' for x in bandnames]
        jobs = [x for x in zip(srcfiles, dstfiles)]
        if not overwrite:
            jobs = [x for x in jobs if not os.path.isfile(x[1])]
            if len(jobs) == 0:
                print('all target tiff files already exist, nothing to be done')
                return
        srcfiles, dstfiles = map(list, zip(*jobs))
        
        multicore(gdalwarp, cores=cores, multiargs={'src': srcfiles, 'dst': dstfiles}, options=options_warp)
    else:
        if len(srcfiles) == 1:
            options_warp['format'] = 'GTiff'
            if not dstfile.endswith('.tif'):
                dstfile = os.path.splitext(dstfile)[0] + '.tif'
            gdalwarp(srcfiles[0], dstfile, options_warp)
        else:
            # create VRT for stacking
            vrt = '/vsimem/' + os.path.basename(dst_base) + '.vrt'
            options_buildvrt['options'] = ['-separate']
            gdalbuildvrt(srcfiles, vrt, options_buildvrt)
            
            # warp files
            gdalwarp(vrt, dstfile, options_warp)
            
            # edit ENVI HDR files to contain specific layer names
            with envi.HDRobject(dstfile + '.hdr') as hdr:
                hdr.band_names = bandnames
                hdr.write()


class Dtype(object):
    def __init__(self, dtype):
        if isinstance(dtype, int):
            if dtype in self.numpy2gdalint.values():
                self.gdalint = dtype
                self.numpystr = self.gdalint2numpystr[self.gdalint]
                self.gdalstr = self.gdalint2gdalstr[self.gdalint]
        elif isinstance(dtype, str):
            if dtype in self.gdalstr2gdalint.keys():
                self.gdalstr = dtype
                self.gdalint = self.gdalstr2gdalint[self.gdalstr]
                self.numpystr = self.gdalint2numpystr[self.gdalint]
            elif dtype in self.numpy2gdalint.keys():
                self.numpystr = dtype
                self.gdalint = self.numpy2gdalint[self.numpystr]
                self.gdalstr = self.gdalint2gdalstr[self.gdalint]
        else:
            raise TypeError('data type identifier must be of type int or str')
        
        required = ['gdalint', 'gdalstr', 'numpystr']
        if sum([x in dir(self) for x in required]) != len(required):
            raise ValueError('unknown data type identifer')
    
    @property
    def numpy2gdalint(self):
        """
        create a dictionary for mapping numpy data types to GDAL data type codes

        Returns
        -------
        dict
            the type map
        """
        if not hasattr(self, '__numpy2gdalint'):
            tmap = {}
            
            for group in ['int', 'uint', 'float', 'complex']:
                for dtype in np.sctypes[group]:
                    code = gdal_array.NumericTypeCodeToGDALTypeCode(dtype)
                    if code is not None:
                        tmap[dtype().dtype.name] = code
            self.__numpy2gdalint = tmap
        return self.__numpy2gdalint
    
    @property
    def gdalstr2gdalint(self):
        map = []
        for key in self.gdalint2numpystr.keys():
            map.append((gdal.GetDataTypeName(key), key))
        return dict(map)
    
    @property
    def gdalint2numpystr(self):
        code = 1
        map = []
        while True:
            out = gdal_array.GDALTypeCodeToNumericTypeCode(code)
            if out is None:
                break
            else:
                map.append((code, out().dtype.name))
                code += 1
        return dict(map)
    
    @property
    def gdalint2gdalstr(self):
        map = []
        for key in self.gdalint2numpystr.keys():
            map.append((key, gdal.GetDataTypeName(key)))
        return dict(map)
