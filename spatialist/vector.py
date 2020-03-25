# -*- coding: utf-8 -*-
################################################################
# OGR wrapper for convenient vector data handling and processing
# John Truckenbrodt 2015-2019
################################################################


import os
import yaml
from osgeo import ogr, osr

from .auxil import crsConvert
from .ancillary import parse_literal
from .sqlite_util import sqlite_setup

ogr.UseExceptions()
osr.UseExceptions()


class Vector(object):
    """
    This is intended as a vector meta information handler with options for reading and writing vector data in a
    convenient manner by simplifying the numerous options provided by the OGR python binding.

    Parameters
    ----------
    filename: str or None
        the vector file to read; if filename is `None`, a new in-memory Vector object is created.
        In this case `driver` is overridden and set to 'Memory'. The following file extensions are auto-detected:
        
        .. list_drivers:: vector
        
    driver: str
        the vector file format; needs to be defined if the format cannot be auto-detected from the filename extension
    """
    
    def __init__(self, filename=None, driver=None):
        
        if filename is None:
            driver = 'Memory'
        elif isinstance(filename, str):
            if not os.path.isfile(filename):
                raise OSError('file does not exist')
            if driver is None:
                driver = self.__driver_autodetect(filename)
        else:
            raise TypeError('filename must either be str or None')
        
        self.filename = filename
        
        self.driver = ogr.GetDriverByName(driver)
        
        self.vector = self.driver.CreateDataSource('out') if driver == 'Memory' else self.driver.Open(filename)
        
        nlayers = self.vector.GetLayerCount()
        if nlayers > 1:
            raise RuntimeError('multiple layers are currently not supported')
        elif nlayers == 1:
            self.init_layer()
    
    def __getitem__(self, expression):
        """
        subset the vector object by index or attribute.

        Parameters
        ----------
        expression: int or str
            the key or expression to be used for subsetting.
            See :osgeo:meth:`ogr.Layer.SetAttributeFilter` for details on the expression syntax.

        Returns
        -------
        Vector
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
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __str__(self):
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
    def __driver_autodetect(filename):
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
    
    def addfeature(self, geometry, fields=None):
        """
        add a feature to the vector object from a geometry

        Parameters
        ----------
        geometry: :osgeo:class:`ogr.Geometry`
            the geometry to add as a feature
        fields: dict or None
            the field names and values to assign to the new feature

        Returns
        -------

        """
        
        feature = ogr.Feature(self.layerdef)
        feature.SetGeometry(geometry)
        
        if fields is not None:
            for fieldname, value in fields.items():
                if fieldname not in self.fieldnames:
                    raise IOError('field "{}" is missing'.format(fieldname))
                try:
                    feature.SetField(fieldname, value)
                except NotImplementedError as e:
                    fieldindex = self.fieldnames.index(fieldname)
                    fieldtype = feature.GetFieldDefnRef(fieldindex).GetTypeName()
                    message = str(e) + '\ntrying to set field {} (type {}) to value {} (type {})'
                    message = message.format(fieldname, fieldtype, value, type(value))
                    raise (NotImplementedError(message))
        
        self.layer.CreateFeature(feature)
        feature = None
        self.init_features()
    
    def addfield(self, name, type, width=10):
        """
        add a field to the vector layer

        Parameters
        ----------
        name: str
            the field name
        type: int
            the OGR Field Type (OFT), e.g. ogr.OFTString.
            See `Module ogr <https://gdal.org/python/osgeo.ogr-module.html>`_.
        width: int
            the width of the new field (only for ogr.OFTString fields)

        Returns
        -------

        """
        fieldDefn = ogr.FieldDefn(name, type)
        if type == ogr.OFTString:
            fieldDefn.SetWidth(width)
        self.layer.CreateField(fieldDefn)
    
    def addlayer(self, name, srs, geomType):
        """
        add a layer to the vector layer

        Parameters
        ----------
        name: str
            the layer name
        srs: int, str or :osgeo:class:`osr.SpatialReference`
            the spatial reference system. See :func:`spatialist.auxil.crsConvert` for options.
        geomType: int
            an OGR well-known binary data type.
            See `Module ogr <https://gdal.org/python/osgeo.ogr-module.html>`_.

        Returns
        -------

        """
        self.vector.CreateLayer(name, srs, geomType)
        self.init_layer()
    
    def addvector(self, vec):
        """
        add a vector object to the layer of the current Vector object

        Parameters
        ----------
        vec: Vector
            the vector object to add
        merge: bool
            merge overlapping polygons?

        Returns
        -------

        """
        vec.layer.ResetReading()
        for feature in vec.layer:
            self.layer.CreateFeature(feature)
        self.init_features()
        vec.layer.ResetReading()
    
    def bbox(self, outname=None, driver=None, overwrite=True):
        """
        create a bounding box from the extent of the Vector object

        Parameters
        ----------
        outname: str or None
            the name of the vector file to be written; if None, a Vector object is returned
        driver: str
            the name of the file format to write
        overwrite: bool
            overwrite an already existing file?

        Returns
        -------
        Vector or None
            if outname is None, the bounding box Vector object
        """
        if outname is None:
            return bbox(self.extent, self.srs)
        else:
            bbox(self.extent, self.srs, outname=outname, driver=driver, overwrite=overwrite)
    
    def clone(self):
        return feature2vector(self.getfeatures(), ref=self)
    
    def close(self):
        """
        closes the OGR vector file connection

        Returns
        -------

        """
        self.vector = None
        for feature in self.__features:
            if feature is not None:
                feature = None
    
    def convert2wkt(self, set3D=True):
        """
        export the geometry of each feature as a wkt string

        Parameters
        ----------
        set3D: bool
            keep the third (height) dimension?

        Returns
        -------

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
    def extent(self):
        """
        the extent of the vector object

        Returns
        -------
        dict
            a dictionary with keys `xmin`, `xmax`, `ymin`, `ymax`
        """
        return dict(zip(['xmin', 'xmax', 'ymin', 'ymax'], self.layer.GetExtent()))
    
    @property
    def fieldDefs(self):
        """

        Returns
        -------
        list of :osgeo:class:`ogr.FieldDefn`
            the field definition for each field of the Vector object
        """
        return [self.layerdef.GetFieldDefn(x) for x in range(0, self.nfields)]
    
    @property
    def fieldnames(self):
        """

        Returns
        -------
        list of str
            the names of the fields
        """
        return sorted([field.GetName() for field in self.fieldDefs])
    
    @property
    def geomType(self):
        """

        Returns
        -------
        int
            the layer geometry type
        """
        return self.layerdef.GetGeomType()
    
    @property
    def geomTypes(self):
        """

        Returns
        -------
        list
            the geometry type of each feature
        """
        return [feat.GetGeometryRef().GetGeometryName() for feat in self.getfeatures()]
    
    def getArea(self):
        """

        Returns
        -------
        float
            the area of the vector geometries
        """
        return sum([x.GetGeometryRef().GetArea() for x in self.getfeatures()])
    
    def getFeatureByAttribute(self, fieldname, attribute):
        """
        get features by field attribute

        Parameters
        ----------
        fieldname: str
            the name of the queried field
        attribute: int or str
            the field value of interest

        Returns
        -------
        list of :osgeo:class:`ogr.Feature` or :osgeo:class:`ogr.Feature`
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
    
    def getFeatureByIndex(self, index):
        """
        get features by numerical (positional) index

        Parameters
        ----------
        index: int
            the queried index

        Returns
        -------
        :osgeo:class:`ogr.Feature`
            the requested feature
        """
        feature = self.layer[index]
        if feature is None:
            feature = self.getfeatures()[index]
        return feature
    
    def getfeatures(self):
        """

        Returns
        -------
        list of :osgeo:class:`ogr.Feature`
            a list of cloned features
        """
        self.layer.ResetReading()
        features = [x.Clone() for x in self.layer]
        self.layer.ResetReading()
        return features
    
    def getProjection(self, type):
        """
        get the CRS of the Vector object. See :func:`spatialist.auxil.crsConvert`.

        Parameters
        ----------
        type: str
            the type of projection required.

        Returns
        -------
        int, str or :osgeo:class:`osr.SpatialReference`
            the output CRS
        """
        return crsConvert(self.layer.GetSpatialRef(), type)
    
    def getUniqueAttributes(self, fieldname):
        """

        Parameters
        ----------
        fieldname: str
            the name of the field of interest

        Returns
        -------
        list of str or int
            the unique attributes of the field
        """
        self.layer.ResetReading()
        attributes = list(set([x.GetField(fieldname) for x in self.layer]))
        self.layer.ResetReading()
        return sorted(attributes)
    
    def init_features(self):
        """
        delete all in-memory features

        Returns
        -------

        """
        del self.__features
        self.__features = [None] * self.nfeatures
    
    def init_layer(self):
        """
        initialize a layer object

        Returns
        -------

        """
        self.layer = self.vector.GetLayer()
        self.__features = [None] * self.nfeatures
    
    @property
    def layerdef(self):
        """

        Returns
        -------
        :osgeo:class:`ogr.FeatureDefn`
            the layer's feature definition
        """
        return self.layer.GetLayerDefn()
    
    @property
    def layername(self):
        """

        Returns
        -------
        str
            the name of the layer
        """
        return self.layer.GetName()
    
    def load(self):
        """
        load all feature into memory

        Returns
        -------

        """
        self.layer.ResetReading()
        for i in range(self.nfeatures):
            if self.__features[i] is None:
                self.__features[i] = self.layer[i]
    
    @property
    def nfeatures(self):
        """

        Returns
        -------
        int
            the number of features
        """
        return len(self.layer)
    
    @property
    def nfields(self):
        """

        Returns
        -------
        int
            the number of fields
        """
        return self.layerdef.GetFieldCount()
    
    @property
    def nlayers(self):
        """

        Returns
        -------
        int
            the number of layers
        """
        return self.vector.GetLayerCount()
    
    @property
    def proj4(self):
        """

        Returns
        -------
        str
            the CRS in PRO4 format
        """
        return self.srs.ExportToProj4().strip()
    
    def reproject(self, projection):
        """
        in-memory reprojection

        Parameters
        ----------
        projection: int, str, :osgeo:class:`osr.SpatialReference`
            the target CRS. See :func:`spatialist.auxil.crsConvert`.

        Returns
        -------

        """
        srs_out = crsConvert(projection, 'osr')
        
        # the following check was found to not work in GDAL 3.0.1; likely a bug
        # if self.srs.IsSame(srs_out) == 0:
        if self.getProjection('epsg') != crsConvert(projection, 'epsg'):
            
            # create the CoordinateTransformation
            coordTrans = osr.CoordinateTransformation(self.srs, srs_out)
            
            layername = self.layername
            geomType = self.geomType
            features = self.getfeatures()
            feat_def = features[0].GetDefnRef()
            fields = [feat_def.GetFieldDefn(x) for x in range(0, feat_def.GetFieldCount())]
            
            self.__init__()
            self.addlayer(layername, srs_out, geomType)
            self.layer.CreateFields(fields)
            
            for feature in features:
                geom = feature.GetGeometryRef()
                geom.Transform(coordTrans)
                newfeature = feature.Clone()
                newfeature.SetGeometry(geom)
                self.layer.CreateFeature(newfeature)
                newfeature = None
            self.init_features()
    
    def setCRS(self, crs):
        """
        directly reset the spatial reference system of the vector object.
        This is not going to reproject the Vector object, see :meth:`reproject` instead.

        Parameters
        ----------
        crs: int, str, :osgeo:class:`osr.SpatialReference`
            the input CRS

        Returns
        -------

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
    def srs(self):
        """

        Returns
        -------
        :osgeo:class:`osr.SpatialReference`
            the geometry's spatial reference system
        """
        return self.layer.GetSpatialRef()
    
    def write(self, outfile, driver=None, overwrite=True):
        """
        write the Vector object to a file

        Parameters
        ----------
        outfile:
            the name of the file to write; the following extensions are automatically detected
            for determining the format driver:
            
            .. list_drivers:: vector
            
        driver: str
            the output file format; needs to be defined if the format cannot
            be auto-detected from the filename extension
        overwrite: bool
            overwrite an already existing file?

        Returns
        -------

        """
        
        if driver is None:
            driver = self.__driver_autodetect(outfile)
        
        driver = ogr.GetDriverByName(driver)
        
        if os.path.exists(outfile):
            if overwrite:
                driver.DeleteDataSource(outfile)
            else:
                raise RuntimeError('target file already exists')
        
        outdataset = driver.CreateDataSource(outfile)
        outlayer = outdataset.CreateLayer(name=self.layername,
                                          srs=self.srs,
                                          geom_type=self.geomType)
        outlayerdef = outlayer.GetLayerDefn()
        
        for fieldDef in self.fieldDefs:
            outlayer.CreateField(fieldDef)
        
        self.layer.ResetReading()
        for feature in self.layer:
            outFeature = ogr.Feature(outlayerdef)
            outFeature.SetGeometry(feature.GetGeometryRef())
            for name in self.fieldnames:
                outFeature.SetField(name, feature.GetField(name))
            # add the feature to the shapefile
            outlayer.CreateFeature(outFeature)
            outFeature = None
        self.layer.ResetReading()
        outdataset = None


def bbox(coordinates, crs, outname=None, driver=None, overwrite=True):
    """
    create a bounding box vector object or shapefile from coordinates and coordinate reference system.
    The CRS can be in either WKT, EPSG or PROJ4 format
    
    Parameters
    ----------
    coordinates: dict
        a dictionary containing numerical variables with keys `xmin`, `xmax`, `ymin` and `ymax`
    crs: int, str, :osgeo:class:`osr.SpatialReference`
        the CRS of the `coordinates`. See :func:`~spatialist.auxil.crsConvert` for options.
    outname: str
        the file to write to. If `None`, the bounding box is returned as :class:`~spatialist.vector.Vector` object
    driver: str
        the output file format; needs to be defined if the format cannot
            be auto-detected from the filename extension
    overwrite: bool
        overwrite an existing file?
    
    Returns
    -------
    Vector or None
        the bounding box Vector object
    """
    srs = crsConvert(crs, 'osr')
    ring = ogr.Geometry(ogr.wkbLinearRing)
    
    ring.AddPoint(coordinates['xmin'], coordinates['ymin'])
    ring.AddPoint(coordinates['xmin'], coordinates['ymax'])
    ring.AddPoint(coordinates['xmax'], coordinates['ymax'])
    ring.AddPoint(coordinates['xmax'], coordinates['ymin'])
    ring.CloseRings()
    
    geom = ogr.Geometry(ogr.wkbPolygon)
    geom.AddGeometry(ring)
    
    geom.FlattenTo2D()
    
    bbox = Vector(driver='Memory')
    bbox.addlayer('bbox', srs, geom.GetGeometryType())
    bbox.addfield('area', ogr.OFTReal)
    bbox.addfeature(geom, fields={'area': geom.Area()})
    geom = None
    if outname is None:
        return bbox
    else:
        bbox.write(outfile=outname, driver=driver, overwrite=overwrite)


def centerdist(obj1, obj2):
    if not isinstance(obj1, Vector) or isinstance(obj2, Vector):
        raise IOError('both objects must be of type Vector')
    
    feature1 = obj1.getFeatureByIndex(0)
    geometry1 = feature1.GetGeometryRef()
    center1 = geometry1.Centroid()
    
    feature2 = obj2.getFeatureByIndex(0)
    geometry2 = feature2.GetGeometryRef()
    center2 = geometry2.Centroid()
    
    return center1.Distance(center2)


def dissolve(infile, outfile, field, layername=None):
    """
    dissolve the polygons of a vector file by an attribute field
    Parameters
    ----------
    infile: str
        the input vector file
    outfile: str
        the output shapefile
    field: str
        the field name to merge the polygons by
    layername: str
        the name of the output vector layer;
        If set to None the layername will be the basename of infile without extension

    Returns
    -------

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
    with Vector(driver='Memory') as merge:
        merge.addlayer(layername, srs, ogr.wkbPolygon)
        merge.addfield(field, type=type, width=width)
        for i in range(len(fetch)):
            merge.addfeature(ogr.CreateGeometryFromWkt(fetch[i][1]), {field: fetch[i][0]})
        merge.write(outfile)
    conn.close()


def feature2vector(feature, ref, layername=None):
    """
    create a Vector object from ogr features

    Parameters
    ----------
    feature: list of :osgeo:class:`ogr.Feature` or :osgeo:class:`ogr.Feature`
        a single feature or a list of features
    ref: Vector
        a reference Vector object to retrieve geo information from
    layername: str or None
        the name of the output layer; retrieved from `ref` if `None`

    Returns
    -------
    Vector
        the new Vector object
    """
    features = feature if isinstance(feature, list) else [feature]
    layername = layername if layername is not None else ref.layername
    vec = Vector(driver='Memory')
    vec.addlayer(layername, ref.srs, ref.geomType)
    feat_def = features[0].GetDefnRef()
    fields = [feat_def.GetFieldDefn(x) for x in range(0, feat_def.GetFieldCount())]
    vec.layer.CreateFields(fields)
    for feat in features:
        vec.layer.CreateFeature(feat)
    vec.init_features()
    return vec


def intersect(obj1, obj2):
    """
    intersect two Vector objects

    Parameters
    ----------
    obj1: Vector
        the first vector object; this object is reprojected to the CRS of obj2 if necessary
    obj2: Vector
        the second vector object

    Returns
    -------
    Vector or None
        the intersect of obj1 and obj2 if both intersect and None otherwise
    """
    if not isinstance(obj1, Vector) or not isinstance(obj2, Vector):
        raise RuntimeError('both objects must be of type Vector')
    
    obj1 = obj1.clone()
    obj2 = obj2.clone()
    
    obj1.reproject(obj2.srs)
    
    #######################################################
    # create basic overlap
    union1 = ogr.Geometry(ogr.wkbMultiPolygon)
    # union all the geometrical features of layer 1
    for feat in obj1.layer:
        union1.AddGeometry(feat.GetGeometryRef())
    obj1.layer.ResetReading()
    union1.Simplify(0)
    # same for layer2
    union2 = ogr.Geometry(ogr.wkbMultiPolygon)
    for feat in obj2.layer:
        union2.AddGeometry(feat.GetGeometryRef())
    obj2.layer.ResetReading()
    union2.Simplify(0)
    # intersection
    intersect_base = union1.Intersection(union2)
    union1 = None
    union2 = None
    #######################################################
    # compute detailed per-geometry overlaps
    if intersect_base.GetArea() > 0:
        intersection = Vector(driver='Memory')
        intersection.addlayer('intersect', obj1.srs, ogr.wkbPolygon)
        fieldmap = []
        for index, fielddef in enumerate([obj1.fieldDefs, obj2.fieldDefs]):
            for field in fielddef:
                name = field.GetName()
                i = 2
                while name in intersection.fieldnames:
                    name = '{}_{}'.format(field.GetName(), i)
                    i += 1
                fieldmap.append((index, field.GetName(), name))
                intersection.addfield(name, type=field.GetType(), width=field.GetWidth())
        
        for feature1 in obj1.layer:
            geom1 = feature1.GetGeometryRef()
            if geom1.Intersects(intersect_base):
                for feature2 in obj2.layer:
                    geom2 = feature2.GetGeometryRef()
                    # select only the intersections
                    if geom2.Intersects(intersect_base):
                        intersect = geom2.Intersection(geom1)
                        fields = {}
                        for item in fieldmap:
                            if item[0] == 0:
                                fields[item[2]] = feature1.GetField(item[1])
                            else:
                                fields[item[2]] = feature2.GetField(item[1])
                        intersection.addfeature(intersect, fields)
        intersect_base = None
        return intersection


def wkt2vector(wkt, srs, layername='wkt'):
    """
    convert a well-known text string geometry to a Vector object

    Parameters
    ----------
    wkt: str
        the well-known text description
    srs: int, str
        the spatial reference system; see :func:`spatialist.auxil.crsConvert` for options.
    layername: str
        the name of the internal :osgeo:class:`ogr.Layer` object

    Returns
    -------
    Vector
        the vector representation
    
    Examples
    --------
    >>> from spatialist.vector import wkt2vector
    >>> wkt = 'POLYGON ((0. 0., 0. 1., 1. 1., 1. 0., 0. 0.))'
    >>> with wkt2vector(wkt, srs=4326) as vec:
    >>>     print(vec.getArea())
    1.0
    """
    geom = ogr.CreateGeometryFromWkt(wkt)
    geom.FlattenTo2D()
    
    srs = crsConvert(srs, 'osr')
    
    vec = Vector(driver='Memory')
    vec.addlayer(layername, srs, geom.GetGeometryType())
    if geom.GetGeometryName() != 'POINT':
        vec.addfield('area', ogr.OFTReal)
        fields = {'area': geom.Area()}
    else:
        fields = None
    vec.addfeature(geom, fields=fields)
    geom = None
    return vec
