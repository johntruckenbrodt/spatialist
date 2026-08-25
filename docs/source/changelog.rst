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

0.9.0 | 2022-01-25
------------------

- :class:`spatialist.raster.Raster`:

  * method :meth:`~spatialist.raster.Raster.write`:

    + remove unused argument `compress_tif`
    + add support for COG driver

- :class:`spatialist.vector.Vector`:

  * method :meth:`~spatialist.vector.Vector.addlayer`:

    + enable all SRS type options supported by :func:`~spatialist.auxil.crsConvert`

- :func:`spatialist.raster.rasterize`: allow value `None` for argument `nodata`

- new functions:

  * :func:`spatialist.vector.vectorize`
  * :func:`spatialist.vector.boundary`

0.10.0 | 2022-02-24
-------------------

- :class:`spatialist.raster.Raster`:

  * method :meth:`~spatialist.raster.Raster.write`:

    + TIFF tag writing via argument `options` (formats 'GTiff' and 'COG')
    + new argument `overview_resampling`
    + changed default format to 'GTiff'

  * improved mechanism for temporary VRT file writing:

    + old: written to :func:`tempfile.gettempdir` and never deleted
    + new: written to subdirectory 'spatialist' of :func:`~tempfile.gettempdir` and deleted
      during :meth:`~spatialist.raster.Raster.close`

- :func:`spatialist.raster.png`: new arguments 'vmin' and 'vmax'

0.10.1 | 2022-03-02
-------------------

- :func:`spatialist.vector.boundary` bug fix

0.11.0 | 2022-06-01
-------------------

- :func:`spatialist.auxil.crsConvert`: new argument `wkt_format`

- :meth:`spatialist.raster.Raster.bbox`: set default of argument 'driver' to `None`

- :func:`spatialist.ancillary.sampler`: new function

- bug fixes

0.12.0 | 2022-12-21
-------------------

- replace argument `options` with general keyword arguments `kwargs` in functions

    + :func:`spatialist.auxil.gdalwarp`
    + :func:`spatialist.auxil.gdalbuildvrt`
    + :func:`spatialist.auxil.gdal_translate`
    + :func:`spatialist.auxil.ogr2ogr`
    + :func:`spatialist.auxil.gdal_rasterize`

0.12.1 | 2023-11-16
-------------------

- installation via `pyproject.toml` instead of `setup.py`
- :func:`spatialist.ancillary.finder`: support for zipfiles with implicit directories

0.13.0 | 2024-04-11
-------------------

- add progress bar to :func:`spatialist.ancillary.multicore` (non-Windows only)

0.13.1 | 2024-04-11
-------------------

- :func:`spatialist.ancillary.multicore` bug fix

0.14.0 | 2024-10-01
-------------------

- :func:`spatialist.vector.feature2vector`: bug fix
- :meth:`spatialist.vector.Vector.addfield`: new argument `values`
- :func:`spatialist.vector.wkt2vector`: enable passing multiple geometries as list

0.15.0 | 2025-04-09
-------------------

- :meth:`spatialist.vector.Vector.write`: significantly reduced lines of code and removed
  bugs by making use of :meth:`osgeo.gdal.Dataset.CopyLayer`
- :meth:`spatialist.vector.Vector.to_geopandas`: new method
- :func:`spatialist.vector.set_field`: new function

  + code outsourced from :meth:`spatialist.vector.Vector.addfield`
  + used by :meth:`spatialist.vector.Vector.addfeature`
  + added support for `DateTime` fields

0.15.1 | 2025-05-09
-------------------

- :meth:`spatialist.vector.Vector.to_geopandas`: fixed bug in `DateTime` field parsing

0.16.0 | 2025-08-22
-------------------

- :func:`spatialist.vector.bbox`:

  + new argument `buffer`
  + change order of coordinates to counter-clockwise

0.16.1 | 2025-10-08
-------------------

- support for numpy>=2.0

0.16.2 | 2026-01-16
-------------------

- moved tests folder to top directory (so it is not included in distributions)
- :func:`spatialist.vector.set_field`: round DateTime fields to milliseconds

0.16.3 | 2026-02-27
-------------------

- use `importlib` instead of legacy `pkg_resources`
- call `UseExceptions()` on all imported `osgeo` submodules
- :func:`spatialist.raster.rasterize`: call :meth:`osgeo.gdal.Dataset.FlushCache` to avoid running into suppressed
  errors when closing the dataset with `target_ds = None`

0.17.0 | 2026-03-06
-------------------

- :func:`spatialist.ancillary.run`: also return the return code of the subprocess

0.17.1 | 2026-03-10
-------------------

- :func:`spatialist.ancillary.run`: bug fixes

  + do not encode `inlist` to `bytes`
  + make sure the function may not return `None`
  + pas right arguments to `sp.CalledProcessError`
  + use `with` context manager for the logfile to make sure it is closed

0.18.0 | 2026-03-12
-------------------

- class :class:`spatialist.raster.Raster`: new argument `driver`
- replaced usage of deprecated `Memory` driver with `MEM`

0.19.0 | 2026-04-02
-------------------

- method :meth:`spatialist.raster.Raster.array`: new argument `mask_nan`
- :mod:`spatialist.ancillary`: removed classes :class:`~spatialist.ancillary.Stack` and :class:`~spatialist.ancillary.Queue`, which are no longer needed
- added typing

0.20.0 | 2026-08-25
-------------------

- added antimeridian handling throughout the package

  * geographic extents crossing the antimeridian are represented with `xmin > xmax`
  * :func:`spatialist.vector.bbox` and :meth:`spatialist.vector.Vector.bbox` can split
    antimeridian-crossing polygons into multipolygons; buffering is antimeridian-safe
  * :attr:`spatialist.vector.Vector.extent` is antimeridian-aware; new methods
    :meth:`~spatialist.vector.Vector.get_extent` and :meth:`~spatialist.vector.Vector.get_extent_parts`
    provide more control over extent calculation
  * :meth:`spatialist.vector.Vector.reproject`:

    + reimplemented using :func:`spatialist.auxil.ogr2ogr`
    + new arguments `split_antimeridian`, `antimeridian_offset` and `inplace`
    + `projection` can be `None` to only perform antimeridian splitting
    + automatically promotes geometries and the layer to the corresponding multi-type if required by splitting

  * new method :meth:`spatialist.vector.Vector.wrap_antimeridian`
  * :attr:`spatialist.vector.Vector.__geo_interface__` returns an EPSG:4326 GeoJSON
    `FeatureCollection` with antimeridian wrapping applied
  * :func:`spatialist.vector.intersect` reimplemented with antimeridian handling and
    promotion of polygon results to multipolygons
  * :func:`spatialist.auxil.utm_autodetect` made antimeridian-safe
  * :meth:`spatialist.raster.Raster.__getitem__` detects unsupported vector subsetting
    across the antimeridian

- :class:`spatialist.vector.Vector`:

  * new method :meth:`~spatialist.vector.Vector.filter` for attribute filtering
  * new method :meth:`~spatialist.vector.Vector.orient_polygon_rings`
  * polygon geometries created or modified by spatialist are consistently oriented
    counter-clockwise for exterior rings and clockwise for interior rings
  * :meth:`~spatialist.vector.Vector.convert2wkt`: new argument `multi` to promote
    geometries to their corresponding multi-type
  * in-memory driver selection is compatible with both the legacy OGR `Memory` driver
    and the unified GDAL `MEM` driver introduced with GDAL 3.11

- new vector functions:

  * :func:`spatialist.vector.hull`: replaces the former `spatialist.vector.boundary`;
    supports point, line and polygon geometries, preserves disconnected polygon parts
    and can optionally connect them with a concave hull
  * :func:`spatialist.vector.combine_polygons`: combine multiple polygon vectors with
    options to explode multipolygons or create a single multipolygon
  * :func:`spatialist.vector.from_geopandas`: create a :class:`spatialist.vector.Vector`
    from a :class:`geopandas.GeoDataFrame`

- :mod:`spatialist.auxil`:

  * new functions :func:`~spatialist.auxil.latlon_clamp`,
    :func:`~spatialist.auxil.latlon_extent_center`,
    :func:`~spatialist.auxil.latlon_normalize` and
    :func:`~spatialist.auxil.longitude_shortest_interval`
  * new geometry iterator functions :func:`~spatialist.auxil.iter_geometries`
    and :func:`~spatialist.auxil.iter_points`
  * :func:`~spatialist.auxil.gdal_translate` and :func:`~spatialist.auxil.ogr2ogr`:
    new argument `void` to optionally return the generated GDAL dataset
  * :func:`~spatialist.auxil.crsConvert`: consistently applies traditional GIS axis order

- :mod:`spatialist.raster`:

  * :class:`~spatialist.raster.Dtype`: new property `bytes`
  * fixed datetime band slicing in :meth:`spatialist.raster.Raster.__getitem__`
  * fixed :meth:`spatialist.raster.Raster.write` to preserve source nodata values
    instead of masking them to NaN before writing
  * fixed handling of :class:`~spatialist.raster.Raster` and
    :class:`~spatialist.vector.Vector` references in :func:`spatialist.raster.reproject`

- :mod:`spatialist.ancillary`:

  * renamed function `union` to `list_intersection` to match its actual behavior
  * :func:`~spatialist.ancillary.parallel_apply_along_axis`: fixed single-core axis
    handling and creation of more chunks than available axis elements
  * :func:`~spatialist.ancillary.sampler`: fixed 2D index calculation and allow drawing
    more samples than matching positions if `replace=True`
  * :func:`~spatialist.ancillary.finder`: fixed ZIP path handling on POSIX systems
  * :func:`~spatialist.ancillary.multicore`: fixed handling of empty `multiargs`

- removed outdated or immature functions:

  * `spatialist.raster.stack`
  * `spatialist.vector.centerdist`
  * `spatialist.vector.boundary` (replaced by :func:`spatialist.vector.hull`)

- substantially extended and restructured the test suite using synthetic test data
