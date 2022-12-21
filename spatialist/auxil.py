##############################################################
# Convenience functions for general spatial applications
# John Truckenbrodt, 2016-2022
##############################################################
import math
import warnings
from osgeo import osr, gdal, ogr
import progressbar as pb
from matplotlib import pyplot as plt

osr.UseExceptions()
ogr.UseExceptions()
gdal.UseExceptions()


def crsConvert(crsIn, crsOut, wkt_format='DEFAULT'):
    """
    convert between different types of spatial reference representations

    Parameters
    ----------
    crsIn: int or str or osgeo.osr.SpatialReference
        the input CRS
    crsOut: str
        the output CRS type; supported options:
        
        - epsg
        - opengis
        - osr
        - prettyWkt
        - proj4
        - wkt
    wkt_format: str
        the format of the `wkt` string. See here for options:
        https://gdal.org/api/ogrspatialref.html#_CPPv4NK19OGRSpatialReference11exportToWktEPPcPPCKc

    Returns
    -------
    int or str or osgeo.osr.SpatialReference
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
                if gdal.__version__ >= '3.0':
                    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            except RuntimeError:
                raise TypeError('crsIn not recognized; must be of type WKT, PROJ4 or EPSG\n'
                                '  was: "{}" of type {}'.format(crsIn, type(crsIn).__name__))
        else:
            raise TypeError('crsIn must be of type int, str or osr.SpatialReference')
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


def haversine(lat1, lon1, lat2, lon2):
    """
    compute the distance in meters between two points in latlon

    Parameters
    ----------
    lat1: int or float
        the latitude of point 1
    lon1: int or float
        the longitude of point 1
    lat2: int or float
        the latitude of point 2
    lon2: int or float
        the longitude of point 2

    Returns
    -------
    float
        the distance between point 1 and point 2 in meters

    """
    radius = 6371000
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return radius * c


def gdalwarp(src, dst, pbar=False, **kwargs):
    """
    a simple wrapper for :func:`osgeo.gdal.Warp`

    Parameters
    ----------
    src: str or osgeo.ogr.DataSource or osgeo.gdal.Dataset or list[str or osgeo.ogr.DataSource or osgeo.gdal.Dataset]
        the input data set
    dst: str
        the output data set
    pbar: bool
        add a progressbar?
    **kwargs
        additional parameters passed to :func:`osgeo.gdal.Warp`; see :func:`osgeo.gdal.WarpOptions`

    Returns
    -------

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


def gdalbuildvrt(src, dst, void=True, **kwargs):
    """
    a simple wrapper for :func:`osgeo.gdal.BuildVRT`

    Parameters
    ----------
    src: str, list, :class:`osgeo.ogr.DataSource` or :class:`osgeo.gdal.Dataset`
        the input data set(s)
    dst: str
        the output data set
    void: bool
        just write the results and don't return anything? If not, the spatial object is returned
    **kwargs
        additional parameters passed to :func:`osgeo.gdal.BuildVRT`; see :func:`osgeo.gdal.BuildVRTOptions`

    Returns
    -------

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
    else:
        return out


def gdal_translate(src, dst, **kwargs):
    """
    a simple wrapper for :func:`osgeo.gdal.Translate`

    Parameters
    ----------
    src: str, osgeo.ogr.DataSource or osgeo.gdal.Dataset
        the input data set
    dst: str
        the output data set
    **kwargs
        additional parameters passed to :func:`osgeo.gdal.Translate`;
        see :func:`osgeo.gdal.TranslateOptions`

    Returns
    -------

    """
    out = gdal.Translate(dst, src, options=gdal.TranslateOptions(**kwargs))
    out = None


def ogr2ogr(src, dst, **kwargs):
    """
    a simple wrapper for :func:`osgeo.gdal.VectorTranslate` aka `ogr2ogr <https://www.gdal.org/ogr2ogr.html>`_

    Parameters
    ----------
    src: str or osgeo.ogr.DataSource
        the input data set
    dst: str
        the output data set
    **kwargs
        additional parameters passed to :func:`osgeo.gdal.VectorTranslate`;
        see :func:`osgeo.gdal.VectorTranslateOptions`

    Returns
    -------

    """
    out = gdal.VectorTranslate(dst, src, options=gdal.VectorTranslateOptions(**kwargs))
    out = None


def gdal_rasterize(src, dst, **kwargs):
    """
    a simple wrapper for :func:`osgeo.gdal.Rasterize`

    Parameters
    ----------
    src: str or osgeo.ogr.DataSource
        the input data set
    dst: str
        the output data set
    **kwargs
        additional parameters passed to :func:`osgeo.gdal.Rasterize`; see :func:`osgeo.gdal.RasterizeOptions`

    Returns
    -------

    """
    out = gdal.Rasterize(dst, src, options=gdal.RasterizeOptions(**kwargs))
    out = None


def coordinate_reproject(x, y, s_crs, t_crs):
    """
    reproject a coordinate from one CRS to another
    
    Parameters
    ----------
    x: int or float
        the X coordinate component
    y: int or float
        the Y coordinate component
    s_crs: int, str or osgeo.osr.SpatialReference
        the source CRS. See :func:`~spatialist.auxil.crsConvert` for options.
    t_crs: int, str or osgeo.osr.SpatialReference
        the target CRS. See :func:`~spatialist.auxil.crsConvert` for options.

    Returns
    -------
    tuple
    
    """
    source = crsConvert(s_crs, 'osr')
    target = crsConvert(t_crs, 'osr')
    transform = osr.CoordinateTransformation(source, target)
    point = transform.TransformPoint(x, y)[:2]
    return point


def utm_autodetect(spatial, crsOut):
    """
    get the UTM CRS for a spatial object
    
    The bounding box of the object is extracted, reprojected to :epsg:`4326` and its
    center coordinate used for computing the best UTM zone fit.
    
    Parameters
    ----------
    spatial: Raster or Vector
        a spatial object in an arbitrary CRS
    crsOut: str
        the output CRS type; see function :func:`crsConvert` for options
    
    Returns
    -------
    int or str or osgeo.osr.SpatialReference
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


def __callback(pct, msg, data):
    """
    helper function to create a progress bar in function gdalwarp
    
    Parameters
    ----------
    pct: float
        the percentage progress
    msg: str
        the message to be printed on each progress step
    data
        the data to be modified during each progress step

    Returns
    -------

    """
    percent = int(pct * 100)
    data.update(percent)
    return 1


def __osr2epsg(srs):
    """
    helper function for crsConvert
    
    Parameters
    ----------
    srs: osgeo.osr.SpatialReference
        an SRS to be converted

    Returns
    -------
    int
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


def cmap_mpl2gdal(mplcolor, values):
    """
    convert a matplotlib color table to a GDAL representation.
    
    Parameters
    ----------
    mplcolor: str
        a color table code
    values: list[int] or range
        the integer data values for which to retrieve colors

    Returns
    -------
    osgeo.gdal.ColorTable
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
