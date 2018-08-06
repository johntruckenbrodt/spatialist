
try:
    from osgeo import gdal
    if gdal.__version__ < '2.1':
        raise ImportError('GDAL version < 2.1. Please refer to the installation isntructions under\n'
                          '    https://github.com/johntruckenbrodt/spatialist')
except ImportError:
    raise ImportError('could not import GDAL. You can install it like this:\n'
                      '    Linux   : sudo apt-get install python3-gdal\n'
                      '    Anaconda: conda install gdal')

from . import vector
from . import envi
from . import ancillary
from . import raster
from . import sqlite_util
from .auxil import crsConvert, haversine, gdalbuildvrt, gdalwarp, gdal_translate, ogr2ogr, gdal_rasterize

from .vector import Vector, bbox, centerdist, intersect
from .raster import Raster, stack, rasterize, dtypes
from .sqlite_util import sqlite_setup, sqlite3

