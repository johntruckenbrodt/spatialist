Changelog
=========

0.4 | 2019-12-05
----------------


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

0.5 | 2020-04-21
----------------

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

0.6 | 2020-07-17
----------------

- method :meth:`spatialist.raster.Raster.write`

  * optionally update an existing file with new arg `update`
  * partial writing with new args `xoff` and `yoff`
  * write external arrays with new arg `array`

- new function :func:`spatialist.raster.png`

- new function :func:`spatialist.raster.apply_along_time`

- bug fixes

0.7 | 2021-06-30
----------------

- :class:`spatialist.raster.Raster`: option to subset objects by

  * band names
  * time range

- :func:`spatialist.auxil.crsConvert`: raise a `RuntimeError` if no corresponding EPSG code was found

- module `spatialist.explorer` and associated demo data and notebook have been outsourced to https://github.com/johntruckenbrodt/spatialist_explorer

0.8 | 2021-09-07
----------------

- :class:`spatialist.raster.Raster`:

  * method :meth:`~spatialist.raster.Raster.write`

    + removed argument `compress_tif`
    + added arguments `options` and `overviews`

  * subsetting support for time stamps, color tables and nodata
  * option to convert band names to time stamps by passing a function as argument `timestamps`
  * bug fixes

- :mod:`spatialist.envi`: enabled reading of HDR files in ZIP archives

0.8.1 | 2021-10-05
------------------

- :func:`spatialist.auxil.crsConvert`:

  * use https for `opengis` links
  * apply ESPG validity checks when output format is `opengis` (not just when `epsg`)

- :func:`spatialist.ancillary.finder`:

  * raise :class:`RuntimeError` (not :class:`TypeError`) if target is a file but is neither zip nor tar
  * raise :class:`RuntimeError` (not :class:`TypeError`) if target is a string but is neither directory nor file
