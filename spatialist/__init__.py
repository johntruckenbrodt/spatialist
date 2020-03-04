from . import vector
from . import envi
from . import ancillary
from . import raster
from . import sqlite_util
from .auxil import crsConvert, haversine, gdalbuildvrt, gdalwarp, gdal_translate, ogr2ogr, gdal_rasterize, \
    utm_autodetect, coordinate_reproject, cmap_mpl2gdal

from .vector import Vector, bbox, centerdist, intersect
from .raster import Raster, stack, rasterize
from .sqlite_util import sqlite_setup, sqlite3

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
