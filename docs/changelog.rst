Changelog
=========

v0.4
----


- :func:`spatialist.auxil.gdalwarp`: optional progressbar via new parameter `pbar`

- :class:`spatialist.raster.Raster`

  * enabled reading data in zip and tar.gz archives
  * :meth:`~spatialist.raster.Raster.bbox`

    + renamed parameter `format` to `driver`
    + new parameter `source` to get coordinates from the image of the GCPs

- :func:`spatialist.raster.stack`

  * improved parallelization
  * new parameter `pbar` to make use of the new :func:`~spatialist.auxil.gdalwarp` functionality

- bug fixes