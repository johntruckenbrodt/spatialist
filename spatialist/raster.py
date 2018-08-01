##############################################################
# GDAL wrapper for convenient raster data handling and processing
# John Truckenbrodt 2015-2018
##############################################################


# todo: function to write data with the same metadata as a given file

from __future__ import division
import os
import re
import shutil
import tempfile
from math import sqrt, floor, ceil
from time import gmtime, strftime
import numpy as np

from . import envi
from .auxil import gdalwarp, gdalbuildvrt
from .vector import Vector, bbox, crsConvert, intersect
from .ancillary import dissolve, multicore
from .envi import HDRobject

from osgeo import (gdal, gdal_array, osr)
from osgeo.gdalconst import (GA_ReadOnly, GA_Update, GDT_Byte, GDT_Int16, GDT_UInt16,
                             GDT_Int32, GDT_UInt32, GDT_Float32, GDT_Float64)

os.environ['GDAL_PAM_PROXY_DIR'] = tempfile.gettempdir()

gdal.UseExceptions()


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
    filename: str or gdal.Dataset
        the raster file/object to read
    """

    # todo: init a Raster object from array data not only from a filename
    def __init__(self, filename):
        if isinstance(filename, gdal.Dataset):
            self.raster = filename
            self.filename = self.files[0] if self.files is not None else None
        elif os.path.isfile(filename):
            self.filename = filename if os.path.isabs(filename) else os.path.join(os.getcwd(), filename)
            self.raster = gdal.Open(filename, GA_ReadOnly)
        else:
            raise OSError('file does not exist')

        # a list to contain arrays
        self.__data = [None] * self.bands

        if self.format == 'ENVI':
            self.bandnames = HDRobject(self.filename + '.hdr').band_names
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

            colmin = int(floor((ext_inter['xmin'] - ext_ras['xmin']) / xres))
            colmax = int(ceil((ext_inter['xmax'] - ext_ras['xmin']) / xres))
            rowmin = int(floor((ext_ras['ymax'] - ext_inter['ymax']) / yres))
            rowmax = int(ceil((ext_ras['ymax'] - ext_inter['ymin']) / yres))
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
                                   self.cols, self.rows, self.bands, dtypes(self.dtype))
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
            a list with a dictionary of statistics for each band. Keys: 'min', 'max', 'mean', 'sdev'.
            See `gdal.Band.ComputeStatistics <http://www.gdal.org/classGDALRasterBand.html#a48883c1dae195b21b37b51b10e910f9b>`_.
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

    def assign(self, array, band):
        """
        assign an array to an existing Raster object

        Parameters
        ----------
        array: np.ndarray
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

    def bbox(self, outname=None, format='ESRI Shapefile', overwrite=True):
        """
        Parameters
        ----------
        outname: str or None
            the name od the file to write; If None, the bounding box is returned as vector object
        format: str
            The file format to write
        overwrite: bool
            overwrite an already existing file?

        Returns
        -------
        spatialist.vector.Vector or None
            the bounding box vector object
        """
        if outname is None:
            return bbox(self.geo, self.proj4)
        else:
            bbox(self.geo, self.proj4, outname=outname, format=format, overwrite=overwrite)

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
        gdal.Driver
            a GDAL raster driver object. See `osgeo.gdal.Driver <http://gdal.org/python/osgeo.gdal.Driver-class.html>`_.
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
        return crsConvert(self.srs, 'epsg')

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
            a value to ignore from the computations; If None, the nodata value of the Raster object is used

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

        :return: (logical) is the file valid?
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
        list of gdal.Band
            a list containing a `gdal.Band <http://gdal.org/python/osgeo.gdal.Band-class.html>`_
            object for each image band
        """
        return [self.raster.GetRasterBand(band) for band in range(1, self.bands + 1)]

    def load(self):
        """
        load all raster data to arrays
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
            convert nodata values to numpy.nan? As numpy.nan requires at least float values, any integer array is cast
            to float32.

        Returns
        -------
        np.ndarray
            the matrix (subset) of the selected band
        """

        mat = self.__data[band - 1]
        if mat is None:
            mat = self.raster.GetRasterBand(band).ReadAsArray()
            if mask_nan:
                try:
                    mat[mat == self.nodata] = np.nan
                except ValueError:
                    mat = mat.astype('float32')
                    mat[mat == self.nodata] = np.nan
        return mat

    @property
    def nodata(self):
        """

        Returns
        -------
        float
            the raster nodata value
        """
        return self.raster.GetRasterBand(1).GetNoDataValue()

    @property
    def projcs(self):
        """
        Returns
        -------
        str or None
            an identifier of the projected coordinate system; If the CRS is not projected None is returned
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
        return self.srs.ExportToProj4()

    @property
    def proj4args(self):
        """

        Returns
        -------
        dict
            the proj4 string arguments as a dictionary
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
        osr.SpatialReference
            a spatial reference object.
            See `osr.SpatialReference <http://gdal.org/python/osgeo.osr.SpatialReference-class.html>`_
            for documentation.
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
            data type notations of GDAL (e.g. 'Float32') and numpy (e.g. 'int8') are supported.
        format:
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

        if format == 'GTiff' and not re.search('\.tif[f]*$', outname):
            outname += '.tif'

        dtype = dtypes(self.dtype if dtype == 'default' else dtype)
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
            outband.SetNoDataValue(nodata)
            mat = self.matrix(band=i)
            outband.WriteArray(mat)
            del mat
            outband.FlushCache()
            outband = None
        if format == 'GTiff':
            outDataset.SetMetadataItem('TIFFTAG_DATETIME', strftime('%Y:%m:%d %H:%M:%S', gmtime()))
        outDataset = None
        if format == 'ENVI':
            with HDRobject(outname + '.hdr') as hdr:
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


def dtypes(typestring):
    """
    translate raster data type descriptions to GDAl data type codes

    Args:
        typestring: str
            the data type string to be converted

    Returns:
    int
        the GDAL data type code
    """
    # create dictionary with GDAL style descriptions
    dictionary = {'Byte': GDT_Byte, 'Int16': GDT_Int16, 'UInt16': GDT_UInt16, 'Int32': GDT_Int32,
                  'UInt32': GDT_UInt32, 'Float32': GDT_Float32, 'Float64': GDT_Float64}

    # add numpy style descriptions
    dictionary.update(typemap())

    if typestring not in dictionary.keys():
        raise ValueError("unknown data type; use one of the following: ['{}']".format("', '".join(dictionary.keys())))

    return dictionary[typestring]


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
        the name of the GeoTiff output file; if None, an in-memory object of type Raster is returned and parameter
        outname is ignored
    burn_values: int or list
        the values to be written to the raster file
    expressions: list
        SQL expressions to filter the vector object by attributes
    nodata: int
        the nodata value of the target raster file
    append: bool
        if the output file already exists, update this file with new rasterized values?
        If set to True and the output file exists, parameters reference and nodata are ignored.

    Returns
    -------
    Raster or None
        if outname is None, a Raster object pointing to an in-memory dataset else None
    Example
    -------
    >>> from spatialist import Vector, Raster, rasterize
    >>> vec = Vector('source.shp')
    >>> ref = Raster('reference.tif')
    >>> outname = 'target.tif'
    >>> expressions = ['ATTRIBUTE=1', 'ATTRIBUTE=2']
    >>> burn_values = [1, 2]
    >>> rasterize(vec, reference, outname, burn_values, expressions)
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
def stack(srcfiles, dstfile, resampling, targetres, srcnodata, dstnodata, shapefile=None, layernames=None, sortfun=None,
          separate=False, overwrite=False, compress=True, cores=4):
    """
    function for mosaicking, resampling and stacking of multiple raster files into a 3D data cube

    Parameters
    ----------
    srcfiles: list
        a list of file names or a list of lists; each sub-list is treated as an order to mosaic its containing files
    dstfile: str
        the destination file (if sesparate) or a directory
    resampling: {near, bilinear, cubic, cubicspline, lanczos, average, mode, max, min, med, Q1, Q3}
        the resampling method; see documentation of gdalwarp
    targetres: tuple
        a list with two entries for x and y spatial resolution in units of the source CRS
    srcnodata: int or float
        the nodata value of the source files
    dstnodata: int or float
        the nodata value of the destination file(s)
    shapefile: str, spatial.vector.Vector or None
        a shapefile for defining the area of the destination files
    layernames: list
        the names of the output layers; if None, the basenames of the input files are used
    sortfun: function
        a function for sorting the input files; this is needed for defining the mosaicking order
    separate: bool
        should the files be written to a single raster block or separate files?
        If separate, each tile is written to geotiff.
    overwrite: bool
        overwrite the file if it already exists?
    compress: bool
        compress the geotiff files?
    cores: int
        the number of CPU threads to use; this is only relevant if separate = True

    Returns
    -------

    Notes
    -----
    This function does not reproject any raster files. Thus, the CRS must be the same for all input raster files.
    This is checked prior to executing gdalwarp. In case a shapefile is defined, it is reprojected internally prior to
    retrieving the extent.
    """
    if len(dissolve(srcfiles)) == 0:
        raise IOError('no input files provided to function raster.stack')

    if layernames is not None:
        if len(layernames) != len(srcfiles):
            raise IOError('mismatch between number of source file groups and layernames')

    if not isinstance(targetres, (list, tuple)) or len(targetres) != 2:
        raise RuntimeError('targetres must be a list or tuple with two entries for x and y resolution')

    if len(srcfiles) == 1 and not isinstance(srcfiles[0], list):
        raise IOError('only one file specified; nothing to be done')

    if resampling not in ['near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average', 'mode', 'max', 'min', 'med',
                          'Q1', 'Q3']:
        raise IOError('resampling method not supported')

    projections = list()
    for x in dissolve(srcfiles):
        try:
            projection = Raster(x).projection
        except OSError as e:
            print('cannot read file: {}'.format(x))
            raise e
        projections.append(projection)

    projections = list(set(projections))
    if len(projections) > 1:
        raise IOError('raster projection mismatch')
    elif len(projections) == 0:
        raise RuntimeError('could not retrieve the projection from any of the {} input images'.format(len(srcfiles)))
    else:
        srs = projections[0]

    # read shapefile bounding coordinates and reduce list of rasters to those overlapping with the shapefile
    if shapefile is not None:
        shp = shapefile if isinstance(shapefile, Vector) else Vector(shapefile)
        shp.reproject(srs)
        ext = shp.extent
        arg_ext = (ext['xmin'], ext['ymin'], ext['xmax'], ext['ymax'])
        for i in range(len(srcfiles)):
            group = sorted(srcfiles[i], key=sortfun) if isinstance(srcfiles[i], list) else [srcfiles[i]]
            group = [x for x in group if intersect(shp, Raster(x).bbox())]
            if len(group) > 1:
                srcfiles[i] = group
            elif len(group) == 1:
                srcfiles[i] = group[0]
            else:
                srcfiles[i] = None
        srcfiles = filter(None, srcfiles)
    else:
        arg_ext = None

    # create temporary directory for writing intermediate files
    dst_base = os.path.splitext(dstfile)[0]
    tmpdir = dst_base + '__tmp'
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)

    options_warp = {'options': ['-q'],
                    'format': 'GTiff' if separate else 'ENVI',
                    'outputBounds': arg_ext, 'multithread': True,
                    'srcNodata': srcnodata, 'dstNodata': dstnodata,
                    'xRes': targetres[0], 'yRes': targetres[1],
                    'resampleAlg': resampling}

    if overwrite:
        options_warp['options'] += ['-overwrite']

    if separate and compress:
        options_warp['options'] += ['-co', 'COMPRESS=DEFLATE', '-co', 'PREDICTOR=2']

    options_buildvrt = {'outputBounds': arg_ext, 'srcNodata': srcnodata}

    # create VRT files for mosaicing
    for i in range(len(srcfiles)):
        base = srcfiles[i][0] if isinstance(srcfiles[i], list) else srcfiles[i]
        vrt = os.path.join(tmpdir, os.path.splitext(os.path.basename(base))[0] + '.vrt')
        gdalbuildvrt(srcfiles[i], vrt, options_buildvrt)
        srcfiles[i] = vrt

    # if no specific layernames are defined and sortfun is not set to None,
    # sort files by custom function or, by default, the basename of the raster/VRT file
    if layernames is None and sortfun is not None:
        srcfiles = sorted(srcfiles, key=sortfun if sortfun else os.path.basename)

    bandnames = [os.path.splitext(os.path.basename(x))[0] for x in srcfiles] if layernames is None else layernames

    if separate or len(srcfiles) == 1:
        if not os.path.isdir(dstfile):
            os.makedirs(dstfile)
        dstfiles = [os.path.join(dstfile, x) + '.tif' for x in bandnames]
        if overwrite:
            files = [x for x in zip(srcfiles, dstfiles)]
        else:
            files = [x for x in zip(srcfiles, dstfiles) if not os.path.isfile(x[1])]
            if len(files) == 0:
                print('all target tiff files already exist, nothing to be done')
                shutil.rmtree(tmpdir)
                return
        srcfiles, dstfiles = map(list, zip(*files))

        multicore(gdalwarp, cores=cores, multiargs={'src': srcfiles, 'dst': dstfiles}, options=options_warp)
    else:
        # create VRT for stacking
        vrt = os.path.join(tmpdir, os.path.basename(dst_base) + '.vrt')
        options_buildvrt['options'] = ['-separate']
        gdalbuildvrt(srcfiles, vrt, options_buildvrt)

        # warp files
        gdalwarp(vrt, dstfile, options_warp)

        # edit ENVI HDR files to contain specific layer names
        with envi.HDRobject(dstfile + '.hdr') as hdr:
            hdr.band_names = bandnames
            hdr.write()

    # remove temporary directory and files
    shutil.rmtree(tmpdir)


def typemap():
    """
    create a dictionary for mapping numpy data types to GDAL data type codes

    Returns
    -------
    dict
        the type map
    """
    tmap = {}

    for group in ['int', 'uint', 'float', 'complex']:
        for dtype in np.sctypes[group]:
            code = gdal_array.NumericTypeCodeToGDALTypeCode(dtype)
            if code is not None:
                tmap[dtype().dtype.name] = code
    return tmap
