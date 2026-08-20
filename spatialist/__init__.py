from . import vector
from . import envi
from . import ancillary
from . import raster
from . import sqlite_util
from .auxil import (crsConvert, haversine, gdalbuildvrt, gdalwarp, gdal_translate,
                    ogr2ogr, gdal_rasterize, utm_autodetect, coordinate_reproject,
                    cmap_mpl2gdal)

from .vector import (Vector, bbox, largest_polygon_exterior, intersect, vectorize)
from .raster import Raster, rasterize
from .sqlite_util import sqlite_setup, sqlite3

import importlib.metadata
__version__ = importlib.metadata.version(__name__)
