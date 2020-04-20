Changelog
=========

v0.4
----


- :func:`spatialist.auxil.gdalwarp`: optional progressbar via new argument `pbar`

- :class:`spatialist.raster.Raster`

  * enabled reading data in zip and tar.gz archives
  * :meth:`~spatialist.raster.Raster.bbox`

    + renamed parameter `format` to `driver`
    + new parameter `source` to get coordinates from the image of the GCPs

- :func:`spatialist.raster.stack`

  * improved parallelization
  * new parameter `pbar` to make use of the new :func:`~spatialist.auxil.gdalwarp` functionality

- bug fixes

v0.5
----

- compatibility of SpatiaLite tools with Windows10
- compatibility with GDAL 3

- new function :func:`spatialist.ancillary.parallel_apply_along_axis`:
  like :func:`numpy.apply_along_axis` but using multiple threads

- new function :func:`spatialist.auxil.cmap_mpl2gdal`: convert matplotlib color sequences to GDAL color tables

- :class:`spatialist.raster.Raster`

  * method :meth:`~spatialist.raster.Raster.write`: new argument `cmap` to write color maps to a file; can be created with e.g. :func:`~spatialist.auxil.cmap_mpl2gdal`
  * subsetting: option to use map coordinates instead of just pixel coordinates
  * method :meth:`~spatialist.raster.Raster.array`:

    + automatically reduce dimensionality of returned arrays using :func:`numpy.squeeze`
    + cast arrays to `float32` if the native data type does not support :py:data:`numpy.nan` for masking missing data

  * option to read image data in all .tar* archives, not just tar.gz
  * new methods :meth:`~spatialist.raster.Raster.coord_map2img` and :meth:`~spatialist.raster.Raster.coord_img2map`
    to convert between pixel/image and map coordinates of a dataset

- :class:`spatialist.vector.Vector`

  * better representation of the object's geometry type(s) with new method :meth:`~spatialist.vector.Vector.geomTypes` and additional info when printing the object with :py:func:`print`

- :class:`spatialist.explorer.RasterViewer`

  * optionally pass custom functions to create additional plots using argument `custom`
