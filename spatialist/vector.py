# -*- coding: utf-8 -*-
################################################################
# OGR wrapper for convenient vector data handling and processing
# John Truckenbrodt 2015-2026
################################################################
from __future__ import annotations

import os
import yaml
from datetime import datetime, timezone, timedelta
from osgeo import ogr, osr, gdal
from osgeo.gdalconst import GDT_Byte
from types import TracebackType
from typing import Any, TYPE_CHECKING
from numpy.typing import NDArray
from packaging.version import Version

if TYPE_CHECKING:
    from .raster import Raster

from .auxil import crsConvert, ogr2ogr
from .ancillary import parse_literal
from .sqlite_util import sqlite_setup

import pandas as pd
import geopandas as gpd
from shapely.wkb import loads as wkb_loads

ogr.UseExceptions()
osr.UseExceptions()
gdal.UseExceptions()

# typing
CRS = int | str | osr.SpatialReference
EXT = dict[str, int | float]


class Vector:
    """
    This is intended as a vector meta information handler with options for reading and writing vector data in a
    convenient manner by simplifying the numerous options provided by the OGR python binding.

    Parameters
    ----------
    filename
        the vector file to read; if filename is `None`, a new in-memory Vector object is created.
        In this case `driver` is overridden and set to 'MEM'. The following file extensions are auto-detected:
        
        .. list_drivers:: vector
        
    driver
        the vector file format; needs to be defined if the format cannot be auto-detected from the filename extension
    """
    
    filename: str | None
    
    def __init__(self, filename: str | None = None, driver: str | None = None) -> None:
        
        memory_driver_name = 'MEM' if Version(gdal.__version__) >= Version('3.11') else 'Memory'
        
        if filename is None:
            driver = memory_driver_name
        elif isinstance(filename, str):
            if not os.path.isfile(filename):
                raise OSError('file does not exist')
            if driver is None:
                driver = self.__driver_autodetect(filename)
        else:
            raise TypeError('filename must either be str or None')
        
        self.filename = filename
        
        self.driver = ogr.GetDriverByName(driver)
        
        if driver == memory_driver_name:
            self.vector = self.driver.CreateDataSource('out')
        else:
            self.vector = self.driver.Open(filename)
        
        nlayers = self.vector.GetLayerCount()
        if nlayers > 1:
            raise RuntimeError('multiple layers are currently not supported')
        elif nlayers == 1:
            self.init_layer()
    
    def __getitem__(self, expression: int | str) -> Vector | None:
        """
        subset the vector object by index or attribute.

        Parameters
        ----------
        expression
            the key or expression to be used for subsetting.
            See :meth:`osgeo.ogr.Layer.SetAttributeFilter` for details on the expression syntax.

        Returns
        -------
        out
            a vector object matching the specified criteria
        
        Examples
        --------
        Assuming we have a shapefile called `testsites.shp`, which has an attribute `sitename`,
        we can subset individual sites and write them to new files like so:
        
        >>> from spatialist import Vector
        >>> filename = 'testsites.shp'
        >>> with Vector(filename)["sitename='site1'"] as site1:
        >>>     site1.write('site1.shp')
        """
        if not isinstance(expression, (int, str)):
            raise RuntimeError('expression must be of type int or str')
        expression = parse_literal(expression) if isinstance(expression, str) else expression
        if isinstance(expression, int):
            feat = self.getFeatureByIndex(expression)
        else:
            self.layer.SetAttributeFilter(expression)
            feat = self.getfeatures()
            feat = feat if len(feat) > 0 else None
            self.layer.SetAttributeFilter('')
        if feat is None:
            return None
        else:
            return feature2vector(feat, ref=self)
    
    def __enter__(self) -> Vector:
        return self
    
    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None
    ) -> None:
        self.close()
    
    def __str__(self) -> str:
        vals = dict()
        vals['proj4'] = self.proj4
        vals.update(self.extent)
        vals['filename'] = self.filename if self.filename is not None else 'memory'
        vals['geomtype'] = ', '.join(list(set(self.geomTypes)))
        
        info = 'class         : spatialist Vector object\n' \
               'geometry type : {geomtype}\n' \
               'extent        : {xmin:.3f}, {xmax:.3f}, {ymin:.3f}, {ymax:.3f} (xmin, xmax, ymin, ymax)\n' \
               'coord. ref.   : {proj4}\n' \
               'data source   : {filename}'.format(**vals)
        return info
    
    @staticmethod
    def __driver_autodetect(filename: str) -> str:
        path = os.path.dirname(os.path.realpath(__file__))
        drivers = yaml.safe_load(open(os.path.join(path, 'drivers_vector.yml')))
        extension = os.path.splitext(filename)[1][1:]
        if extension not in drivers.keys():
            message = "the file extension '{}' is not supported. " \
                      "Please provide the OGR format descriptor via " \
                      "parameter 'driver' or use one of the supported extensions:\n- .{}"
            message = message.format(extension, '\n- .'.join(drivers.keys()))
            raise RuntimeError(message)
        else:
            return drivers[extension]
    
    def addfeature(self, geometry: ogr.Geometry, fields: dict[str, Any] | None = None) -> None:
        """
        add a feature to the vector object from a geometry

        Parameters
        ----------
        geometry
            the geometry to add as a feature
        fields
            the field names and values to assign to the new feature
        """
        
        feature = ogr.Feature(self.layerdef)
        feature.SetGeometry(geometry)
        
        if fields is not None:
            for field_name, value in fields.items():
                if field_name not in self.fieldnames:
                    raise IOError('field "{}" is missing'.format(field_name))
                field_defn = feature.GetFieldDefnRef(field_name)
                field_type = field_defn.GetType()
                field_type_name = field_defn.GetTypeName()
                try:
                    set_field(target=feature, name=field_name,
                              type=field_type, values=value)
                except Exception as e:
                    message = str(e) + (f'\ntrying to set field {field_name} '
                                        f'(type {field_type_name}) to value '
                                        f'{value} (type {type(value)})')
                    raise RuntimeError(message) from e
        
        self.layer.CreateFeature(feature)
        feature = None
        self.init_features()
    
    def addfield(self, name: str, type: int, width: int = 10, values: list[Any] | None = None) -> None:
        """
        add a field to the vector layer

        Parameters
        ----------
        name
            the field name
        type
            the OGR Field Type (OFT), e.g. ogr.OFTString.
            See :class:`osgeo.ogr.FieldDefn`.
        width
            the width of the new field (only for ogr.OFTString fields)
        values
            an optional list with values for each feature to assign to the new field.
            The length must be identical to the number of features.
        """
        set_field(self, name, type, width=width, values=values)
    
    def addlayer(self, name: str, srs: CRS, geomType: int) -> None:
        """
        add a layer to the vector layer

        Parameters
        ----------
        name
            the layer name
        srs
            the spatial reference system. See :func:`spatialist.auxil.crsConvert` for options.
        geomType
            an OGR well-known binary data type.
            See `Module ogr <https://gdal.org/python/osgeo.ogr-module.html>`_.
        """
        self.vector.CreateLayer(name, crsConvert(srs, 'osr'), geomType)
        self.init_layer()
    
    def addvector(self, vec: Vector) -> None:
        """
        add a vector object to the layer of the current Vector object

        Parameters
        ----------
        vec
            the vector object to add
        merge: bool
            merge overlapping polygons?
        """
        vec.layer.ResetReading()
        for feature in vec.layer:
            self.layer.CreateFeature(feature)
        self.init_features()
        vec.layer.ResetReading()
    
    def bbox(
            self,
            outname: str | None = None,
            driver: str | None = None,
            overwrite: bool = True,
            split_antimeridian: bool = True,
    ) -> Vector | None:
        """
        create a bounding box from the extent of the Vector object

        Parameters
        ----------
        outname
            the name of the vector file to be written; if None, a Vector object is returned
        driver
            the name of the file format to write. Ignored if `outname=None`.
        overwrite
            overwrite an already existing file? Ignored if `outname=None`.
        split_antimeridian
            split polygons into multipolygons if they're crossing the antimeridian?
            It is assumed that `xmax` < `xmin` as check for antimeridian crossing.
            Only applied to geographic CRSs.

        Returns
        -------
            the bounding box Vector object id `outname=None` and `None` otherwise.
        """
        return bbox(coordinates=self.extent, crs=self.srs,
                    outname=outname, driver=driver, overwrite=overwrite,
                    split_antimeridian=split_antimeridian)
    
    def clone(self) -> Vector:
        return feature2vector(self.getfeatures(), ref=self)
    
    def close(self) -> None:
        """
        closes the OGR vector file connection
        """
        self.vector = None
        for feature in self.__features:
            if feature is not None:
                feature = None
    
    def convert2wkt(self, set3D: bool = True) -> list[str]:
        """
        export the geometry of each feature as a wkt string

        Parameters
        ----------
        set3D
            keep the third (height) dimension?
        """
        features = self.getfeatures()
        for feature in features:
            try:
                feature.geometry().Set3D(set3D)
            except AttributeError:
                dim = 3 if set3D else 2
                feature.geometry().SetCoordinateDimension(dim)
        
        return [feature.geometry().ExportToWkt() for feature in features]
    
    @property
    def extent(self) -> EXT:
        """
        The extent of the vector object.
        
        Note
        ----
        The extent is auto-split along the antimeridian.
        Use method :meth:`~spatialist.vector.Vector.get_extent` if you do not require this.

        Returns
        -------
            a dictionary with keys `xmin`, `xmax`, `ymin`, `ymax`
        """
        return self.get_extent(split_antimeridian=True)
    
    def get_extent(self, split_antimeridian: bool = True) -> EXT:
        """
        Get the extent of the vector object.
        Optionally splits along the antimeridian.
        
        Parameters
        ----------
        split_antimeridian
            split the extent along the antimeridian?

        Returns
        -------
            a dictionary with keys `xmin`, `xmax`, `ymin`, `ymax`
        
        Example
        -------
        >>> from spatialist.vector import bbox
        >>> extent = {'xmin': 178, 'xmax': -178, 'ymin': 50, 'ymax': 51}
        >>> box = bbox(coordinates=extent, crs=4326)
        >>> print(box.get_extent(split_antimeridian=False))
        {'xmin': -180.0, 'xmax': 180.0, 'ymin': 50.0, 'ymax': 51.0}
        >>> print(box.get_extent(split_antimeridian=True))
        {'xmin': 178.0, 'xmax': -178.0, 'ymin': 50.0, 'ymax': 51.0}
        >>> box.close()
        """
        extent_plain = dict(zip(
            ['xmin', 'xmax', 'ymin', 'ymax'],
            self.layer.GetExtent()
        ))
        if not split_antimeridian:
            return extent_plain
        else:
            if self.layer.GetSpatialRef().IsGeographic():
                extent_parts = self.get_extent_parts()
                xmin = [part['xmin'] for part in extent_parts]
                xmax = [part['xmax'] for part in extent_parts]
                ymin = [part['ymin'] for part in extent_parts]
                ymax = [part['ymax'] for part in extent_parts]
                if 180 in xmax and -180 in xmin:
                    return {'xmin': max(xmin), 'xmax': min(xmax),
                            'ymin': min(ymin), 'ymax': max(ymax)}
                else:
                    return extent_plain
            else:
                return extent_plain
    
    def get_extent_parts(self) -> list[EXT]:
        """
        Get extents for individual geometry parts of all features.
        Multipolygons are split into polygon parts.
        """
        
        def iter_geometry_parts(geom):
            """Yield polygon parts; for MultiPolygon yields each Polygon."""
            if geom is None or geom.IsEmpty():
                return
            
            gtype = ogr.GT_Flatten(geom.GetGeometryType())
            
            if gtype == ogr.wkbMultiPolygon:
                for i in range(geom.GetGeometryCount()):
                    yield geom.GetGeometryRef(i)
            
            elif gtype == ogr.wkbPolygon:
                yield geom
            
            else:
                yield geom
        
        self.layer.ResetReading()
        
        extent_parts = []
        
        for feat in self.layer:
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            
            keys = ['xmin', 'xmax', 'ymin', 'ymax']
            for i, part in enumerate(iter_geometry_parts(geom)):
                extent_parts.append(dict(zip(keys, part.GetEnvelope())))
        self.layer.ResetReading()
        return extent_parts
    
    @property
    def fieldDefs(self) -> list[ogr.FieldDefn]:
        """

        Returns
        -------
            the field definition for each field of the Vector object
        """
        return [self.layerdef.GetFieldDefn(x) for x in range(0, self.nfields)]
    
    @property
    def fieldnames(self) -> list[str]:
        """

        Returns
        -------
            the names of the fields
        """
        return sorted([field.GetName() for field in self.fieldDefs])
    
    @property
    def geomType(self) -> int:
        """

        Returns
        -------
            the layer geometry type
        """
        return self.layerdef.GetGeomType()
    
    @property
    def geomTypes(self) -> list[str]:
        """

        Returns
        -------
            the geometry type of each feature
        """
        return [feat.GetGeometryRef().GetGeometryName() for feat in self.getfeatures()]
    
    def getArea(self) -> float:
        """

        Returns
        -------
            the area of the vector geometries
        """
        return sum([x.GetGeometryRef().GetArea() for x in self.getfeatures()])
    
    def getFeatureByAttribute(
            self,
            fieldname: str,
            attribute: int | str
    ) -> ogr.Feature | list[ogr.Feature] | None:
        """
        get features by field attribute

        Parameters
        ----------
        fieldname
            the name of the queried field
        attribute
            the field value of interest

        Returns
        -------
            the feature(s) matching the search query
        """
        attr = attribute.strip() if isinstance(attribute, str) else attribute
        if fieldname not in self.fieldnames:
            raise KeyError('invalid field name')
        out = []
        self.layer.ResetReading()
        for feature in self.layer:
            field = feature.GetField(fieldname)
            field = field.strip() if isinstance(field, str) else field
            if field == attr:
                out.append(feature.Clone())
        self.layer.ResetReading()
        if len(out) == 0:
            return None
        elif len(out) == 1:
            return out[0]
        else:
            return out
    
    def getFeatureByIndex(self, index: int) -> ogr.Feature | None:
        """
        get features by numerical (positional) index

        Parameters
        ----------
        index
            the queried index

        Returns
        -------
            the requested feature
        """
        feature = self.layer[index]
        if feature is None:
            feature = self.getfeatures()[index]
        return feature
    
    def getfeatures(self) -> list[ogr.Feature]:
        """

        Returns
        -------
            a list of cloned features
        """
        self.layer.ResetReading()
        features = [x.Clone() for x in self.layer]
        self.layer.ResetReading()
        return features
    
    def getProjection(self, type: str) -> CRS:
        """
        get the CRS of the Vector object. See :func:`spatialist.auxil.crsConvert`.

        Parameters
        ----------
        type
            the type of projection required.

        Returns
        -------
            the output CRS
        """
        return crsConvert(self.layer.GetSpatialRef(), type)
    
    def getUniqueAttributes(self, fieldname: str) -> list[int | str]:
        """

        Parameters
        ----------
        fieldname
            the name of the field of interest

        Returns
        -------
            the unique attributes of the field
        """
        self.layer.ResetReading()
        attributes = list(set([x.GetField(fieldname) for x in self.layer]))
        self.layer.ResetReading()
        return sorted(attributes)
    
    def init_features(self) -> None:
        """
        delete all in-memory features
        """
        del self.__features
        self.__features = [None] * self.nfeatures
    
    def init_layer(self) -> None:
        """
        initialize a layer object
        """
        self.layer = self.vector.GetLayer()
        self.__features = [None] * self.nfeatures
    
    @property
    def layerdef(self) -> ogr.FeatureDefn:
        """

        Returns
        -------
            the layer's feature definition
        """
        return self.layer.GetLayerDefn()
    
    @property
    def layername(self) -> str:
        """

        Returns
        -------
            the name of the layer
        """
        return self.layer.GetName()
    
    def load(self) -> None:
        """
        load all feature into memory
        """
        self.layer.ResetReading()
        for i in range(self.nfeatures):
            if self.__features[i] is None:
                self.__features[i] = self.layer[i]
    
    @property
    def nfeatures(self) -> int:
        """

        Returns
        -------
            the number of features
        """
        return len(self.layer)
    
    @property
    def nfields(self) -> int:
        """

        Returns
        -------
            the number of fields
        """
        return self.layerdef.GetFieldCount()
    
    @property
    def nlayers(self) -> int:
        """

        Returns
        -------
            the number of layers
        """
        return self.vector.GetLayerCount()
    
    @property
    def proj4(self) -> str:
        """

        Returns
        -------
            the CRS in PRO4 format
        """
        return self.srs.ExportToProj4().strip()
    
    def reproject(self, projection: CRS, split_antimeridian: bool = True) -> None:
        """
        In-memory reprojection. Antimeridian splitting is performed automatically.

        Parameters
        ----------
        projection
            the target CRS. See :func:`spatialist.auxil.crsConvert`.
        split_antimeridian
            split geometries along the antimeridian if projecting to a geographic CRS?
        """
        srs_out = crsConvert(projection, 'osr')
        
        if split_antimeridian and srs_out.IsGeographic():
            geometryType = 'PROMOTE_TO_MULTI'
            options = ['-wrapdateline']
        else:
            geometryType = None
            options = []
        
        if self.getProjection('epsg') != crsConvert(projection, 'epsg'):
            ds = ogr2ogr(
                src=self.vector,
                dst='',
                format='MEM',
                dstSRS=srs_out,
                reproject=True,
                geometryType=geometryType,
                options=options,
                void=False
            )
            self.__init__()
            self.vector = ds
            self.init_layer()
    
    def setCRS(self, crs: CRS) -> None:
        """
        directly reset the spatial reference system of the vector object.
        This is not going to reproject the Vector object, see :meth:`reproject` instead.

        Parameters
        ----------
        crs
            the input CRS

        Example
        -------
        >>> site = Vector('shape.shp')
        >>> site.setCRS('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ')

        """
        # try to convert the input crs to osr.SpatialReference
        srs_out = crsConvert(crs, 'osr')
        
        # save all relevant info from the existing vector object
        layername = self.layername
        geomType = self.geomType
        layer_definition = ogr.Feature(self.layer.GetLayerDefn())
        fields = [layer_definition.GetFieldDefnRef(x) for x in range(layer_definition.GetFieldCount())]
        features = self.getfeatures()
        
        # initialize a new vector object and create a layer
        self.__init__()
        self.addlayer(layername, srs_out, geomType)
        
        # add the fields to new layer
        self.layer.CreateFields(fields)
        
        # add the features to the newly created layer
        for feat in features:
            self.layer.CreateFeature(feat)
        self.init_features()
    
    @property
    def srs(self) -> osr.SpatialReference:
        """

        Returns
        -------
            the geometry's spatial reference system
        """
        return self.layer.GetSpatialRef()
    
    def to_geopandas(self) -> gpd.GeoDataFrame:
        """
        Convert the object to a geopandas GeoDataFrame.
        `DateTime` fields are converted to :class:`pandas.Timestamp`
        using :func:`pandas.to_datetime`.
        
        Returns
        -------
            the dataframe object
        
        See Also
        --------
        osgeo.ogr.Feature.items
        """
        field_types = {x.GetName(): x.GetTypeName() for x in self.fieldDefs}
        features = []
        self.layer.ResetReading()
        for feature in self.layer:
            geom = feature.GetGeometryRef()
            geom_wkb = geom.ExportToWkb()
            properties = feature.items()
            properties["geometry"] = wkb_loads(bytes(geom_wkb))
            features.append(properties)
        self.layer.ResetReading()
        gdf = gpd.GeoDataFrame(features, crs=self.srs.ExportToWkt())
        for field_name, field_type in field_types.items():
            if field_type == "DateTime":
                gdf[field_name] = pd.to_datetime(arg=gdf[field_name],
                                                 format='ISO8601')
        return gdf
    
    def wrap_dateline(self, offset: int | float = 10) -> None:
        """
        Split geometries crossing the antimeridian in-place.

        Parameters
        ----------
        offset
            Distance in degrees from the antimeridian where geometries are considered
            for splitting. This corresponds to ogr2ogr's ``-datelineoffset`` option.
        """
        if not self.srs.IsGeographic():
            return
        
        options = [
            "-wrapdateline",
            "-datelineoffset", str(offset),
        ]
        
        ds = ogr2ogr(
            src=self.vector,
            dst="",
            format="MEM",
            dstSRS=self.srs,
            geometryType="PROMOTE_TO_MULTI",
            options=options,
            void=False,
        )
        
        self.__init__()
        self.vector = ds
        self.init_layer()
    
    def write(self, outfile: str, driver: str | None = None, overwrite: bool = True) -> None:
        """
        write the Vector object to a file

        Parameters
        ----------
        outfile
            the name of the file to write; the following extensions are automatically detected
            for determining the format driver:
            
            .. list_drivers:: vector
            
        driver
            the output file format; default None: try to autodetect from the file name extension
        overwrite
            overwrite an already existing file?
        """
        
        if driver is None:
            driver = self.__driver_autodetect(outfile)
        
        driver = ogr.GetDriverByName(driver)
        
        if os.path.exists(outfile):
            if overwrite:
                driver.DeleteDataSource(outfile)
            else:
                raise RuntimeError('target file already exists')
        
        ds_out = driver.CreateDataSource(outfile)
        ds_out.CopyLayer(self.layer, self.layer.GetName())
        ds_out = driver = None


def bbox(
        coordinates: EXT,
        crs: CRS,
        outname: str | None = None,
        driver: str | None = None,
        overwrite: bool = True,
        buffer: int | float | tuple[int | float, int | float] | None = None,
        split_antimeridian: bool = True
) -> Vector | None:
    """
    create a bounding box vector object or file.
    The CRS can be in either WKT, EPSG or PROJ4 format
    
    Parameters
    ----------
    coordinates
        a dictionary containing numerical variables with keys
        `xmin`, `xmax`, `ymin` and `ymax`.
    crs
        the coordinate reference system of the `coordinates`.
        See :func:`~spatialist.auxil.crsConvert` for options.
    outname
        the file to write to. If `None`, the bounding box is returned
        as :class:`~spatialist.vector.Vector` object.
    driver
        the output file format; needs to be defined if the format
        cannot be auto-detected from the filename extension.
    overwrite
        overwrite an existing file?
    buffer
        a buffer to add around `coordinates`. Default None: do not add
        a buffer. A tuple is interpreted as (x buffer, y buffer).
    split_antimeridian
        split polygons into multipolygons if they're crossing the antimeridian?
        It is assumed that `xmax` < `xmin` as check for antimeridian crossing.
        Only applied to geographic CRSs.
    
    Returns
    -------
        the bounding box Vector object
    """
    if buffer is not None:
        coordinates = coordinates.copy()
        if isinstance(buffer, tuple):
            xbuffer, ybuffer = buffer
        else:
            xbuffer = ybuffer = buffer
        coordinates['xmin'] -= xbuffer
        coordinates['xmax'] += xbuffer
        coordinates['ymin'] -= ybuffer
        coordinates['ymax'] += ybuffer
    
    srs = crsConvert(crs, 'osr')
    
    def create_polygon(xmin, ymin, xmax, ymax):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(xmin, ymin)
        ring.AddPoint(xmax, ymin)
        ring.AddPoint(xmax, ymax)
        ring.AddPoint(xmin, ymax)
        ring.CloseRings()
        geom = ogr.Geometry(ogr.wkbPolygon)
        geom.AddGeometry(ring)
        return geom
    
    if split_antimeridian and srs.IsGeographic() and coordinates['xmax'] < coordinates['xmin']:
        geom1 = create_polygon(xmin=coordinates['xmin'], ymin=coordinates['ymin'],
                               xmax=180, ymax=coordinates['ymax'])
        geom2 = create_polygon(xmin=-180, ymin=coordinates['ymin'],
                               xmax=coordinates['xmax'], ymax=coordinates['ymax'])
        geom = ogr.Geometry(ogr.wkbMultiPolygon)
        geom.AddGeometry(geom1)
        geom.AddGeometry(geom2)
    else:
        geom = create_polygon(xmin=coordinates['xmin'], ymin=coordinates['ymin'],
                              xmax=coordinates['xmax'], ymax=coordinates['ymax'])
    
    geom.FlattenTo2D()
    
    out = Vector()
    out.addlayer('bbox', srs, geom.GetGeometryType())
    out.addfield('area', ogr.OFTReal)
    out.addfeature(geom, fields={'area': geom.Area()})
    geom = None
    if outname is None:
        return out
    else:
        out.write(outfile=outname, driver=driver, overwrite=overwrite)


def largest_polygon_exterior(
        vectorobject: Vector,
        expression: str | None = None,
        outname: str | None = None
) -> Vector | None:
    """
    Get the exterior of the largest polygon as new polygon vector object.
    This was developed to get the valid data extent of a raster image
    vectorized using :func:`vectorize` while omitting data gaps from masking.
    E.g. for a SAR image while ignoring areas masked due to layover/shadow.
    
    The following steps are performed:
    
     - find the largest polygon matching the expression
     - get the exterior ring of this polygon as new polygon
     - create a new :class:`Vector` object and add the newly created polygon

    Parameters
    ----------
    vectorobject
        the vector object containing multiple polygon geometries.
    expression
        the SQL expression to select the candidates for the largest polygon,
        e.g. all polygons with a certain value in one field.
    outname
        the name of the output vector file;
        if None, an in-memory object of type :class:`Vector` is returned.

    Returns
    -------
        if `outname` is `None`, a vector object pointing to an
        in-memory dataset else `None`
    """
    if not isinstance(vectorobject, Vector):
        raise TypeError("'vectorobject' must be of type Vector")
    
    geom_types = set(vectorobject.geomTypes)
    
    if geom_types != {"POLYGON"}:
        raise RuntimeError(
            "largest_polygon_exterior() only supports Polygon geometries; "
            f"found: {sorted(geom_types)}"
        )
    
    vectorobject.layer.ResetReading()
    
    if expression is not None:
        vectorobject.layer.SetAttributeFilter(expression)
    
    largest_geom = None
    largest_area = -1
    
    try:
        for feat in vectorobject.layer:
            geom = feat.GetGeometryRef()
            if geom is None or geom.IsEmpty():
                continue
            
            area = geom.GetArea()
            if area > largest_area:
                largest_geom = geom.Clone()
                largest_area = area
    finally:
        feat = geom = None
        if expression is not None:
            vectorobject.layer.SetAttributeFilter('')
        vectorobject.layer.ResetReading()
    
    if largest_geom is None:
        raise RuntimeError(
            "no polygon matched the supplied expression"
            if expression is not None
            else "no valid polygon geometry found"
        )
    
    exterior = largest_geom.GetGeometryRef(0)
    if exterior is None:
        raise RuntimeError("largest polygon has no exterior ring")
    
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(exterior.Clone())
    
    out = Vector()
    out.addlayer(
        name="largest_polygon_exterior",
        srs=vectorobject.srs,
        geomType=ogr.wkbPolygon,
    )
    out.addfield("area", ogr.OFTReal)
    out.addfeature(polygon, fields={"area": polygon.GetArea()})
    
    largest_geom = polygon = exterior = None
    
    if outname is not None:
        out.write(outname)
        out.close()
        return None
    
    return out


def centerdist(obj1: Vector, obj2: Vector) -> float:
    """
    Get the center distance between two vector objects.
    
    Parameters
    ----------
    obj1
    obj2

    Returns
    -------
        the distance in units of the source object's CRS
    """
    if not isinstance(obj1, Vector) or isinstance(obj2, Vector):
        raise IOError('both objects must be of type Vector')
    
    feature1 = obj1.getFeatureByIndex(0)
    geometry1 = feature1.GetGeometryRef()
    center1 = geometry1.Centroid()
    
    feature2 = obj2.getFeatureByIndex(0)
    geometry2 = feature2.GetGeometryRef()
    center2 = geometry2.Centroid()
    
    return center1.Distance(center2)


def dissolve(infile: str, outfile: str, field: str, layername: str | None = None) -> None:
    """
    dissolve the polygons of a vector file by an attribute field
    
    Parameters
    ----------
    infile
        the input vector file
    outfile
        the output shapefile
    field
        the field name to merge the polygons by
    layername
        the name of the output vector layer;
        If set to None the layername will be the basename of infile without extension
    """
    with Vector(infile) as vec:
        srs = vec.srs
        feat = vec.layer[0]
        d = feat.GetFieldDefnRef(field)
        width = d.width
        type = d.type
        feat = None
    
    layername = layername if layername is not None else os.path.splitext(os.path.basename(infile))[0]
    
    # the following can be used if GDAL was compiled with the spatialite extension
    # not tested, might need some additional/different lines
    # with Vector(infile) as vec:
    #     vec.vector.ExecuteSQL('SELECT ST_Union(geometry), {0} FROM {1} GROUP BY {0}'.format(field, vec.layername),
    #                          dialect='SQLITE')
    #     vec.write(outfile)
    
    conn = sqlite_setup(extensions=['spatialite', 'gdal'])
    conn.execute('CREATE VIRTUAL TABLE merge USING VirtualOGR("{}");'.format(infile))
    select = conn.execute('SELECT {0},asText(ST_Union(geometry)) as geometry FROM merge GROUP BY {0};'.format(field))
    fetch = select.fetchall()
    with Vector() as merge:
        merge.addlayer(layername, srs, ogr.wkbPolygon)
        merge.addfield(field, type=type, width=width)
        for i in range(len(fetch)):
            merge.addfeature(ogr.CreateGeometryFromWkt(fetch[i][1]), {field: fetch[i][0]})
        merge.write(outfile)
    conn.close()


def feature2vector(
        feature: ogr.Feature | list[ogr.Feature],
        ref: Vector,
        layername: str | None = None
) -> Vector:
    """
    create a Vector object from ogr features

    Parameters
    ----------
    feature
        a single feature or a list of features
    ref
        a reference Vector object to retrieve geo information from
    layername
        the name of the output layer; retrieved from `ref` if `None`

    Returns
    -------
        the new Vector object
    """
    features = feature if isinstance(feature, list) else [feature]
    layername = layername if layername is not None else ref.layername
    vec = Vector()
    vec.addlayer(layername, ref.srs, ref.geomType)
    feat_def = features[0].GetDefnRef()
    fields = [feat_def.GetFieldDefn(x) for x in range(0, feat_def.GetFieldCount())]
    vec.layer.CreateFields(fields)
    for feat in features:
        feat2 = ogr.Feature(vec.layer.GetLayerDefn())
        feat2.SetFrom(feat)
        vec.layer.CreateFeature(feat2)
    vec.init_features()
    return vec


def from_geopandas(gdf: gpd.GeoDataFrame, layer_name: str = "layer") -> Vector:
    """
    Convert a geopandas GeoDataFrame to a Vector object.
    
    Parameters
    ----------
    gdf
        The input GeoDataFrame. All features (columns) must have the same geometry type.
    layer_name
        The name of the Vector object's layer.
    """
    out = Vector()
    
    srs = osr.SpatialReference()
    srs.ImportFromWkt(gdf.crs.to_wkt())
    
    geom_types = list(set(gdf.geometry.dropna().geom_type.unique()))
    
    if len(geom_types) > 1:
        raise RuntimeError(f'Multiple geometry types are not supported. '
                           f'Found: {geom_types}.')
    
    geom_type = getattr(ogr, f'wkb{geom_types[0]}')
    
    out.addlayer(name=layer_name, srs=srs, geomType=geom_type)
    
    for name, dtype in gdf.drop(columns=gdf.geometry.name).dtypes.items():
        if dtype.kind in {"i", "u"}:
            field_type = ogr.OFTInteger64
        elif dtype.kind == "f":
            field_type = ogr.OFTReal
        else:
            field_type = ogr.OFTString
        out.addfield(name, field_type)
    
    layer_defn = out.layer.GetLayerDefn()
    
    for _, row in gdf.iterrows():
        feat = ogr.Feature(layer_defn)
        
        for name in gdf.columns:
            if name == gdf.geometry.name:
                continue
            value = row[name]
            if value is not None:
                feat.SetField(name, value)
        
        geom = ogr.CreateGeometryFromWkb(row.geometry.wkb)
        feat.SetGeometry(geom)
        out.layer.CreateFeature(feat)
    
    layer_defn = None
    feat = None
    
    return out


def intersect(obj1: Vector, obj2: Vector) -> Vector | None:
    """
    Intersect two (multi)polygon Vector objects.

    Parameters
    ----------
    obj1
        The first vector object ("input layer").
        This object is reprojected to the CRS of ``obj2`` if necessary.
    obj2
        The second vector object ("method layer").

    Returns
    -------
        The intersection of ``obj1`` and ``obj2`` if both intersect and ``None`` otherwise.
    
    See Also
    --------
    osgeo.ogr.Layer.Intersection
    """
    if not isinstance(obj1, Vector) or not isinstance(obj2, Vector):
        raise RuntimeError("both objects must be of type Vector")
    
    for vector in (obj1, obj2):
        if not all(gt in ("POLYGON", "MULTIPOLYGON") for gt in vector.geomTypes):
            raise RuntimeError(
                "intersect() only supports polygon and multipolygon geometries."
            )
    
    obj1 = obj1.clone()
    obj2 = obj2.clone()
    
    obj1.reproject(obj2.srs)
    
    if obj2.srs.IsGeographic():
        obj1.wrap_dateline()
        obj2.wrap_dateline()
    
    out = Vector()
    out.addlayer("intersect", obj2.srs, ogr.wkbMultiPolygon)
    
    err = obj1.layer.Intersection(
        method_layer=obj2.layer,
        result_layer=out.layer,
        options=[
            "SKIP_FAILURES=NO",  # abort and raise an error if any overlay fails
            "PROMOTE_TO_MULTI=YES",  #
            "KEEP_LOWER_DIMENSION_GEOMETRIES=NO",
        ],
    )
    
    if err != ogr.OGRERR_NONE:
        raise RuntimeError("OGR layer intersection failed")
    
    return out if out.nfeatures > 0 else None


def set_field(
        target: Vector | ogr.Feature,
        name: str,
        type: int,
        width: int = 10,
        values: Any = None
) -> None:
    """
    Wrapper for setting a field. DateTime fields are rounded to milliseconds.
    
    Parameters
    ----------
    target
        the object for which to set the field
    name
        the field name
    type
        the OGR Field Type (OFT), e.g. `ogr.OFTString`.
        See :class:`osgeo.ogr.FieldDefn`.
    width
        the width of the new field (only for `ogr.OFTString` fields)
    values
        an optional list with values for each feature to assign to the new field.
        If `target` is of type :class:`~spatialist.vector.Vector`, the length must
        be identical to the number of features.
    """
    type_name = ogr.GetFieldTypeName(type)
    field_defn = ogr.FieldDefn(name, type)
    if type == ogr.OFTString:
        field_defn.SetWidth(width)
    
    if isinstance(target, Vector):
        target.layer.CreateField(field_defn)
    
    if type_name in ['String', 'Integer', 'Real', 'Binary', 'DateTime']:
        method_name = 'SetField'
    elif type_name in ['StringList', 'IntegerList',
                       'Integer64', 'Integer64List']:
        method_name = f'SetField{type_name}'
    elif type_name == 'RealList':
        method_name = 'SetFieldDoubleList'
    else:
        raise ValueError(f'Unsupported field type: {type_name}')
    
    def setter(feature: ogr.Feature, value: Any, field_name: str, method_name: str, type_name: str) -> None:
        
        def tz_to_nTZFlag(dt: datetime) -> int:
            """
            Determine OGR nTZFlag from a timezone-aware datetime.
            """
            if dt.tzinfo is None:
                return 0  # unknown
            offset = dt.utcoffset()
            if offset == timezone.utc.utcoffset(None):
                return 100  # UTC
            return 1  # assume local (non-UTC, but known)
        
        index = feature.GetFieldIndex(field_name)
        method = getattr(feature, method_name)
        if type_name == 'DateTime':
            if isinstance(value, datetime):
                # Round to milliseconds and normalize
                value = value + timedelta(microseconds=500)  # for rounding, not truncation
                value = value.replace(microsecond=(value.microsecond // 1000) * 1000)
                value = [
                    value.year,
                    value.month,
                    value.day,
                    value.hour,
                    value.minute,
                    value.second + value.microsecond / 1_000_000,
                    tz_to_nTZFlag(value)
                ]
            else:
                raise TypeError("If 'type' is 'DateTime', the value "
                                "must be a datetime.datetime object")
            method(index, *value)
        else:
            method(index, value)
    
    if values is not None:
        if isinstance(target, Vector):
            if len(values) != target.nfeatures:
                raise RuntimeError('number of values does not match number of features')
            target.layer.ResetReading()
            for i, feature in enumerate(target.layer):
                setter(feature=feature, value=values[i], field_name=name,
                       method_name=method_name, type_name=type_name)
                target.layer.SetFeature(feature)
            target.layer.ResetReading()
        elif isinstance(target, ogr.Feature):
            setter(feature=target, value=values, field_name=name,
                   method_name=method_name, type_name=type_name)
        else:
            raise TypeError("'target' must be of type spatialist.vector.Vector "
                            "or osgeo.ogr.Feature")


def wkt2vector(wkt: str | list[str], srs: CRS, layername: str = 'wkt') -> Vector:
    """
    convert well-known text geometries to a Vector object.

    Parameters
    ----------
    wkt
        the well-known text description(s). Each geometry will be placed in a separate feature.
    srs
        the spatial reference system; see :func:`spatialist.auxil.crsConvert` for options.
    layername
        the name of the internal :class:`osgeo.ogr.Layer` object.

    Returns
    -------
        the vector representation

    Examples
    --------
    >>> from spatialist.vector import wkt2vector
    >>> wkt1 = 'POLYGON ((0. 0., 0. 1., 1. 1., 1. 0., 0. 0.))'
    >>> with wkt2vector(wkt1, srs=4326) as vec:
    >>>     print(vec.getArea())
    1.0
    >>> wkt2 = 'POLYGON ((1. 1., 1. 2., 2. 2., 2. 1., 1. 1.))'
    >>> with wkt2vector([wkt1, wkt2], srs=4326) as vec:
    >>>     print(vec.getArea())
    2.0
    """
    if isinstance(wkt, str):
        wkt = [wkt]
    srs = crsConvert(srs, 'osr')
    vec = Vector()
    area = []
    for item in wkt:
        geom = ogr.CreateGeometryFromWkt(item)
        geom.FlattenTo2D()
        if not hasattr(vec, 'layer'):
            vec.addlayer(layername, srs, geom.GetGeometryType())
        if geom.GetGeometryName() != 'POINT':
            area.append(geom.Area())
        else:
            area.append(None)
        vec.addfeature(geom)
        geom = None
    vec.addfield('area', ogr.OFTReal, values=area)
    return vec


def vectorize(
        target: NDArray[Any],
        reference: Raster,
        outname: str | None = None,
        layername: str = 'layer',
        fieldname: str = 'value',
        driver: str | None = None
) -> Vector | None:
    """
    Vectorization of an array using :func:`osgeo.gdal.Polygonize`.
    
    Parameters
    ----------
    target
        the input array. Each identified object of pixels with the same value will be converted into a vector feature.
    reference
        a reference Raster object to retrieve geo information and extent from.
    outname
        the name of the vector file. If `None` a vector object is returned.
    layername
        the name of the vector object layer.
    fieldname
        the name of the field to contain the raster value for the respective vector feature.
    driver
        the vector file type of `outname`. Several extensions are read automatically (see :meth:`Vector.write`).
        Is ignored if ``outname=None``.
    """
    cols = reference.cols
    rows = reference.rows
    meta = reference.raster.GetMetadata()
    geo = reference.raster.GetGeoTransform()
    proj = reference.raster.GetProjection()
    
    tmp_driver = gdal.GetDriverByName('MEM')
    tmp = tmp_driver.Create(layername, cols, rows, 1, GDT_Byte)
    tmp.SetMetadata(meta)
    tmp.SetGeoTransform(geo)
    tmp.SetProjection(proj)
    outband = tmp.GetRasterBand(1)
    outband.WriteArray(target, 0, 0)
    
    try:
        with Vector() as vec:
            vec.addlayer(name=layername, srs=proj,
                         geomType=ogr.wkbPolygon)
            vec.addfield(fieldname, ogr.OFTInteger)
            
            gdal.Polygonize(srcBand=outband, maskBand=None,
                            outLayer=vec.layer, iPixValField=0)
            if outname is not None:
                vec.write(outfile=outname, driver=driver)
            else:
                return vec.clone()
    except Exception as e:
        raise e
    finally:
        outband = None
        tmp_driver = None
        out = None
