Some General Examples
=====================
in-memory vector object rasterization
-------------------------------------
| Here we create a new raster data set with the same geo-information and extent as a reference data set
 and burn the geometries from a shapefile into it.
| In this example, the shapefile contains an attribute ``Site_name`` and one of the geometries in the shapefile has a
 value of ``my_testsite`` for this attribute.
| We use the ``expressions`` parameter to subset the shapefile and burn a value of 1 in the raster at all locations
 where the geometry selection overlaps. Multiple expressions can be defined together with multiple burn values.
| Also, burn values can be appended to an already existing raster data set. In this case, the rasterization is
 performed in-memory to further use it for e.g. plotting. Alternatively, an ``outname`` can be defined to directly write
 the result to disk as a GeoTiff.
| See :func:`spatialist.raster.rasterize` for further reference.

>>> from spatialist import Vector, Raster
>>> from spatialist.raster import rasterize
>>> import matplotlib.pyplot as plt
>>>
>>> shapefile = 'testsites.shp'
>>> rasterfile = 'extent.tif'
>>>
>>> with Raster(rasterfile) as ras:
>>>     with Vector(shapefile) as vec:
>>>         mask = rasterize(vec, reference=ras, burn_values=1, expressions=["Site_Name='my testsite'"])
>>>         plt.imshow(mask.matrix())
>>>         plt.show()
