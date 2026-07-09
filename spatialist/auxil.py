##############################################################
# Convenience functions for general spatial applications
# John Truckenbrodt, 2016-2026
##############################################################
from __future__ import annotations

import math
import warnings
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .raster import Raster
    from .vector import Vector

from osgeo import osr, gdal, ogr
import progressbar as pb
from matplotlib import pyplot as plt
from packaging.version import Version

osr.UseExceptions()
ogr.UseExceptions()
gdal.UseExceptions()

GDS = str | gdal.Dataset
CRS = int | str | osr.SpatialReference


def crsConvert(
        crsIn: CRS,
        crsOut: str,
        wkt_format: str = 'DEFAULT'
) -> CRS:
    """
    convert between different types of spatial reference representations

    Parameters
    ----------
    crsIn
        the input CRS
    crsOut
        the output CRS type; supported options:
        
        - epsg
        - opengis
        - osr
        - prettyWkt
        - proj4
        - wkt
    wkt_format
        the format of the `wkt` string. See here for options:
        https://gdal.org/api/ogrspatialref.html#_CPPv4NK19OGRSpatialReference11exportToWktEPPcPPCKc

    Returns
    -------
        the output CRS

    Examples
    --------
    convert an integer EPSG code to PROJ.4:

    >>> crsConvert(4326, 'proj4')
    '+proj=longlat +datum=WGS84 +no_defs '

    convert the opengis URL back to EPSG:

    >>> crsConvert('https://www.opengis.net/def/crs/EPSG/0/4326', 'epsg')
    4326
    
    convert an EPSG compound CRS (WGS84 horizontal + EGM96 vertical) to PROJ.4
    
    >>> crsConvert('EPSG:4326+5773', 'proj4')
    '+proj=longlat +datum=WGS84 +geoidgrids=us_nga_egm96_15.tif +vunits=m +no_defs'
    """
    if isinstance(crsIn, osr.SpatialReference):
        srs = crsIn.Clone()
    else:
        srs = osr.SpatialReference()
        
        if isinstance(crsIn, int):
            crsIn = 'EPSG:{}'.format(crsIn)
        
        if isinstance(crsIn, str):
            try:
                srs.SetFromUserInput(crsIn)
            except RuntimeError:
                raise TypeError('crsIn not recognized; must be of type WKT, PROJ4 or EPSG\n'
                                '  was: "{}" of type {}'.format(crsIn, type(crsIn).__name__))
        else:
            raise TypeError('crsIn must be of type int, str or osr.SpatialReference')
    
    if Version(gdal.__version__) >= Version('3.0'):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    
    if crsOut == 'wkt':
        if wkt_format == 'DEFAULT':
            # keep downward compatibility
            return srs.ExportToWkt()
        else:
            return srs.ExportToWkt(['FORMAT={}'.format(wkt_format)])
    elif crsOut == 'prettyWkt':
        return srs.ExportToPrettyWkt()
    elif crsOut == 'proj4':
        return srs.ExportToProj4()
    elif crsOut == 'epsg':
        return __osr2epsg(srs)
    elif crsOut == 'opengis':
        code = __osr2epsg(srs)
        return 'https://www.opengis.net/def/crs/EPSG/0/{}'.format(code)
    elif crsOut == 'osr':
        return srs
    else:
        raise ValueError('crsOut not recognized; must be either wkt, prettyWkt, proj4, epsg, opengis or osr')


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    compute the distance in meters between two points in latlon

    Parameters
    ----------
    lat1
        the latitude of point 1
    lon1
        the longitude of point 1
    lat2
        the latitude of point 2
    lon2
        the longitude of point 2

    Returns
    -------
        the distance between point 1 and point 2 in meters

    """
    radius = 6371000
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return radius * c


def gdalwarp(
        src: GDS | list[GDS],
        dst: str,
        pbar: bool = False,
        **kwargs: Any
) -> None:
    """
    a simple wrapper for :func:`osgeo.gdal.Warp`

    Parameters
    ----------
    src
        the input data set
    dst
        the output data set
    pbar
        add a progressbar?
    **kwargs
        additional parameters passed to :func:`osgeo.gdal.Warp`; see :func:`osgeo.gdal.WarpOptions`
    """
    progress = None
    try:
        if pbar:
            kwargs = kwargs.copy()
            widgets = [pb.Percentage(), pb.Bar(), pb.Timer(), ' ', pb.ETA()]
            progress = pb.ProgressBar(max_value=100, widgets=widgets).start()
            kwargs['callback'] = __callback
            kwargs['callback_data'] = progress
        out = gdal.Warp(dst, src, options=gdal.WarpOptions(**kwargs))
        if progress is not None:
            progress.finish()
    except RuntimeError as e:
        msg = '{}:\n  src: {}\n  dst: {}\n  options: {}'
        raise RuntimeError(msg.format(str(e), src, dst, kwargs))
    finally:
        out = None


def gdalbuildvrt(
        src: GDS | list[GDS],
        dst: str,
        void: bool = True,
        **kwargs: Any
) -> gdal.Dataset | None:
    """
    a simple wrapper for :func:`osgeo.gdal.BuildVRT`

    Parameters
    ----------
    src
        the input data set(s)
    dst
        the output data set
    void
        just write the results and don't return anything? If not, the spatial object is returned.
    **kwargs
        additional parameters passed to :func:`osgeo.gdal.BuildVRT`; see :func:`osgeo.gdal.BuildVRTOptions`
    """
    
    if 'outputBounds' in kwargs.keys() and gdal.__version__ < '2.4.0':
        warnings.warn('\ncreating VRT files with subsetted extent is very likely to cause problems. '
                      'Please use GDAL version >= 2.4.0, which fixed the problem.\n'
                      'see here for a description of the problem:\n'
                      '  https://gis.stackexchange.com/questions/314333/'
                      'sampling-error-using-gdalwarp-on-a-subsetted-vrt\n'
                      'and here for the release note of GDAL 2.4.0:\n'
                      '  https://trac.osgeo.org/gdal/wiki/Release/2.4.0-News')
    
    out = gdal.BuildVRT(dst, src, options=gdal.BuildVRTOptions(**kwargs))
    out.FlushCache()
    if void:
        out = None
    return out


def gdal_translate(
        src: GDS,
        dst: str,
        void: bool = True,
        **kwargs: Any
) -> gdal.Dataset | None:
    """
    a simple wrapper for :func:`osgeo.gdal.Translate`

    Parameters
    ----------
    src
        the input data set
    dst
        the output data set
    void
        just write the results and don't return anything? If not, the spatial object is returned.
    **kwargs
        additional parameters passed to :func:`osgeo.gdal.Translate`;
        see :func:`osgeo.gdal.TranslateOptions`
    """
    out = gdal.Translate(dst, src, options=gdal.TranslateOptions(**kwargs))
    out.FlushCache()
    if void:
        out = None
    return out


def ogr2ogr(
        src: GDS,
        dst: str,
        void: bool = True,
        **kwargs: Any
) -> gdal.Dataset | None:
    """
    a simple wrapper for :func:`osgeo.gdal.VectorTranslate` aka `ogr2ogr <https://www.gdal.org/ogr2ogr.html>`_

    Parameters
    ----------
    src
        the input data set
    dst
        the output data set
    void
        just write the results and don't return anything? If not, the spatial object is returned.
    **kwargs
        additional parameters passed to :func:`osgeo.gdal.VectorTranslate`;
        see :func:`osgeo.gdal.VectorTranslateOptions`
    """
    out = gdal.VectorTranslate(destNameOrDestDS=dst, srcDS=src,
                               options=gdal.VectorTranslateOptions(**kwargs))
    out.FlushCache()
    if void:
        out = None
    return out


def gdal_rasterize(src: GDS, dst: str, **kwargs: Any) -> None:
    """
    a simple wrapper for :func:`osgeo.gdal.Rasterize`

    Parameters
    ----------
    src
        the input data set
    dst
        the output data set
    **kwargs
        Additional parameters passed to :func:`osgeo.gdal.Rasterize`.
        See :func:`osgeo.gdal.RasterizeOptions`.
    """
    out = gdal.Rasterize(dst, src, options=gdal.RasterizeOptions(**kwargs))
    out = None


def coordinate_reproject(
        x: float,
        y: float,
        s_crs: CRS,
        t_crs: CRS
) -> tuple[float, float]:
    """
    reproject a coordinate from one CRS to another
    
    Parameters
    ----------
    x
        the X coordinate component
    y
        the Y coordinate component
    s_crs
        the source CRS. See :func:`~spatialist.auxil.crsConvert` for options.
    t_crs
        the target CRS. See :func:`~spatialist.auxil.crsConvert` for options.
    """
    source = crsConvert(s_crs, 'osr')
    target = crsConvert(t_crs, 'osr')
    transform = osr.CoordinateTransformation(source, target)
    point = transform.TransformPoint(x, y)[:2]
    return point


def utm_autodetect(spatial: Raster | Vector, crsOut: str) -> CRS:
    """
    get the UTM CRS for a spatial object
    
    The bounding box of the object is extracted, reprojected to :epsg:`4326` and its
    center coordinate used for computing the best UTM zone fit.
    
    Parameters
    ----------
    spatial
        a spatial object in an arbitrary CRS
    crsOut
        the output CRS type; see function :func:`crsConvert` for options
    
    Returns
    -------
    out
        the output CRS
    """
    with spatial.bbox() as box:
        box.reproject(4326)
        ext = box.extent
    lon = (ext['xmax'] + ext['xmin']) / 2
    lat = (ext['ymax'] + ext['ymin']) / 2
    zone = int(1 + (lon + 180.0) / 6.0)
    north = lat > 0
    utm_cs = osr.SpatialReference()
    utm_cs.SetWellKnownGeogCS('WGS84')
    if gdal.__version__ >= '3.0':
        utm_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    utm_cs.SetUTM(zone, north)
    return crsConvert(utm_cs, crsOut)


def __callback(pct: float, msg: str, data: Any) -> int:
    """
    helper function to create a progress bar in function gdalwarp
    
    Parameters
    ----------
    pct
        the percentage progress
    msg
        the message to be printed on each progress step
    data
        the data to be modified during each progress step

    Returns
    -------

    """
    percent = int(pct * 100)
    data.update(percent)
    return 1


def __osr2epsg(srs: osr.SpatialReference) -> int:
    """
    helper function for crsConvert
    
    Parameters
    ----------
    srs
        an SRS to be converted

    Returns
    -------
        the EPSG code if one exists
    
    Raises
    ------
    RuntimeError
    """
    srs = srs.Clone()
    try:
        try:
            srs.AutoIdentifyEPSG()
        except RuntimeError:
            # Sometimes EPSG identification might fail
            # but a match exists for which it does not.
            matches = srs.FindMatches()
            for srs, confidence in matches:
                if confidence == 100:
                    srs.AutoIdentifyEPSG()
                    break
        code = int(srs.GetAuthorityCode(None))
        # make sure the EPSG code actually exists
        srsTest = osr.SpatialReference()
        srsTest.ImportFromEPSG(code)
        srsTest = None
    except RuntimeError:
        raise RuntimeError('CRS does not have an EPSG representation')
    finally:
        srs = None
    return code


def cmap_mpl2gdal(mplcolor: str, values: list[int] | range) -> gdal.ColorTable:
    """
    convert a matplotlib color table to a GDAL representation.
    
    Parameters
    ----------
    mplcolor
        a color table code
    values
        the integer data values for which to retrieve colors

    Returns
    -------
    cmap
        the color table in GDAL format
    
    Note
    ----
    This function is currently only developed for handling discrete integer
    data values in an 8 Bit file. Colors are thus scaled between 0 and 255.
    
    Examples
    --------
    >>> from osgeo import gdal
    >>> from spatialist.auxil import cmap_mpl2gdal
    >>> values = list(range(0, 100))
    >>> cmap = cmap_mpl2gdal(mplcolor='YlGnBu', values=values)
    >>> print(isinstance(cmap, gdal.ColorTable))
    True
    """
    
    cmap_plt = plt.get_cmap(mplcolor, len(values))
    
    cmap = gdal.ColorTable()
    
    for i in values:
        color = tuple(int(round(x * 255)) for x in cmap_plt(i))
        cmap.SetColorEntry(i, color)
    return cmap
