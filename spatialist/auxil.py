##############################################################
# Convenience functions for general spatial applications
# John Truckenbrodt, 2016-2020
##############################################################
import math
import warnings
from osgeo import osr, gdal, ogr
import progressbar as pb
from matplotlib import pyplot as plt

osr.UseExceptions()
ogr.UseExceptions()
gdal.UseExceptions()


def crsConvert(crsIn, crsOut):
    """
    convert between different types of spatial references

    Parameters
    ----------
    crsIn: int, str, :osgeo:class:`osr.SpatialReference`
        the input CRS
    crsOut: {'wkt', 'proj4', 'epsg', 'osr', 'opengis' or 'prettyWkt'}
        the output CRS type

    Returns
    -------
    int, str, :osgeo:class:`osr.SpatialReference`
        the output CRS

    Examples
    --------
    convert an integer EPSG code to PROJ4:

    >>> crsConvert(4326, 'proj4')
    '+proj=longlat +datum=WGS84 +no_defs '

    convert a PROJ4 string to an opengis URL:

    >>> crsConvert('+proj=longlat +datum=WGS84 +no_defs ', 'opengis')
    'http://www.opengis.net/def/crs/EPSG/0/4326'

    convert the opengis URL back to EPSG:

    >>> crsConvert('http://www.opengis.net/def/crs/EPSG/0/4326', 'epsg')
    4326
    
    convert an EPSG compound CRS (WGS84 horizontal + EGM96 vertical)
    
    >>> crsConvert('EPSG:4326+5773', 'proj4')
    '+proj=longlat +datum=WGS84 +geoidgrids=egm96_15.gtx +vunits=m +no_defs '
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
        return srs.ExportToWkt()
    elif crsOut == 'prettyWkt':
        return srs.ExportToPrettyWkt()
    elif crsOut == 'proj4':
        return srs.ExportToProj4()
    elif crsOut == 'epsg':
        srs.AutoIdentifyEPSG()
        code = int(srs.GetAuthorityCode(None))
        # make sure the ESPG code actually exists
        try:
            srsTest = osr.SpatialReference()
            srsTest.ImportFromEPSG(code)
        except RuntimeError:
            raise RuntimeError('CRS does not have a EPSG representation')
        return code
    elif crsOut == 'opengis':
        srs.AutoIdentifyEPSG()
        return 'http://www.opengis.net/def/crs/EPSG/0/{}'.format(srs.GetAuthorityCode(None))
    elif crsOut == 'osr':
        return srs
    else:
        raise ValueError('crsOut not recognized; must be either wkt, proj4, opengis or epsg')


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


def gdalwarp(src, dst, options, pbar=False):
    """
    a simple wrapper for :osgeo:func:`gdal.Warp`

    Parameters
    ----------
    src: str, :osgeo:class:`ogr.DataSource` or :osgeo:class:`gdal.Dataset`
        the input data set
    dst: str
        the output data set
    options: dict
        additional parameters passed to gdal.Warp; see :osgeo:func:`gdal.WarpOptions`
    pbar: bool
        add a progressbar?

    Returns
    -------

    """
    try:
        if pbar:
            options = options.copy()
            widgets = [pb.Percentage(), pb.Bar(), pb.Timer(), ' ', pb.ETA()]
            progress = pb.ProgressBar(max_value=100, widgets=widgets).start()
            options['callback'] = __callback
            options['callback_data'] = progress
        out = gdal.Warp(dst, src, options=gdal.WarpOptions(**options))
        if pbar:
            progress.finish()
    except RuntimeError as e:
        raise RuntimeError('{}:\n  src: {}\n  dst: {}\n  options: {}'.format(str(e), src, dst, options))
    out = None


def gdalbuildvrt(src, dst, options=None, void=True):
    """
    a simple wrapper for :osgeo:func:`gdal.BuildVRT`

    Parameters
    ----------
    src: str, list, :osgeo:class:`ogr.DataSource` or :osgeo:class:`gdal.Dataset`
        the input data set(s)
    dst: str
        the output data set
    options: dict
        additional parameters passed to gdal.BuildVRT; see :osgeo:func:`gdal.BuildVRTOptions`
    void: bool
        just write the results and don't return anything? If not, the spatial object is returned

    Returns
    -------

    """
    options = {} if options is None else options
    
    if 'outputBounds' in options.keys() and gdal.__version__ < '2.4.0':
        warnings.warn('\ncreating VRT files with subsetted extent is very likely to cause problems. '
                      'Please use GDAL version >= 2.4.0, which fixed the problem.\n'
                      'see here for a description of the problem:\n'
                      '  https://gis.stackexchange.com/questions/314333/'
                      'sampling-error-using-gdalwarp-on-a-subsetted-vrt\n'
                      'and here for the release note of GDAL 2.4.0:\n'
                      '  https://trac.osgeo.org/gdal/wiki/Release/2.4.0-News')
    
    out = gdal.BuildVRT(dst, src, options=gdal.BuildVRTOptions(**options))
    out.FlushCache()
    if void:
        out = None
    else:
        return out


def gdal_translate(src, dst, options):
    """
    a simple wrapper for `gdal.Translate <https://gdal.org/python/osgeo.gdal-module.html#Translate>`_

    Parameters
    ----------
    src: str, :osgeo:class:`ogr.DataSource` or :osgeo:class:`gdal.Dataset`
        the input data set
    dst: str
        the output data set
    options: dict
        additional parameters passed to gdal.Translate;
        see `gdal.TranslateOptions <http://gdal.org/python/osgeo.gdal-module.html#TranslateOptions>`_

    Returns
    -------

    """
    out = gdal.Translate(dst, src, options=gdal.TranslateOptions(**options))
    out = None


def ogr2ogr(src, dst, options):
    """
    a simple wrapper for gdal.VectorTranslate aka `ogr2ogr <https://www.gdal.org/ogr2ogr.html>`_

    Parameters
    ----------
    src: str or :osgeo:class:`ogr.DataSource`
        the input data set
    dst: str
        the output data set
    options: dict
        additional parameters passed to gdal.VectorTranslate;
        see `gdal.VectorTranslateOptions <http://gdal.org/python/osgeo.gdal-module.html#VectorTranslateOptions>`_

    Returns
    -------

    """
    out = gdal.VectorTranslate(dst, src, options=gdal.VectorTranslateOptions(**options))
    out = None


def gdal_rasterize(src, dst, options):
    """
    a simple wrapper for gdal.Rasterize

    Parameters
    ----------
    src: str or :osgeo:class:`ogr.DataSource`
        the input data set
    dst: str
        the output data set
    options: dict
        additional parameters passed to gdal.Rasterize; see :osgeo:func:`gdal.RasterizeOptions`

    Returns
    -------

    """
    out = gdal.Rasterize(dst, src, options=gdal.RasterizeOptions(**options))
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
    s_crs: int, str or :osgeo:class:`osr.SpatialReference`
        the source CRS. See :func:`~spatialist.auxil.crsConvert` for options.
    t_crs: int, str or :osgeo:class:`osr.SpatialReference`
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
    int or str or :osgeo:class:`osr.SpatialReference`
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


def cmap_mpl2gdal(mplcolor, values):
    """
    convert a matplotlib color table to a GDAL representation.
    
    Parameters
    ----------
    mplcolor: str
        a color table code
    values: list
        the integer data values for which to retrieve colors

    Returns
    -------
    :osgeo:class:`gdal.ColorTable`
        the color table in GDAL format
    
    Notes
    -----
    This function is currently only developed for handling discrete integer data values in an 8 Bit file.
    Colors are thus scaled between 0 and 255.
    
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
