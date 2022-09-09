"""
osgeo documentation hyperlink extension module for sphinx documentation
John Truckenbrodt 2018-2020

this enables to place short rst link directives for osgeo gdal module methods, classes and functions.
Instead of each time placing `gdal.Dataset <https://gdal.org/python/osgeo.gdal.Dataset-class.html>`__ into
a docstring for linking to the class documentation, one can just type the following:

:osgeo:class:`gdal.Dataset`             -> https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Dataset
:osgeo:meth:`gdal.Dataset.ReadAsArray`  -> https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Dataset.ReadAsArray
:osgeo:func:`ogr.CreateGeometryFromWkt` -> https://gdal.org/api/python/osgeo.ogr.html#osgeo.ogr.CreateGeometryFromWkt
:osgeo:module:`gdalconst`               -> https://gdal.org/api/python/osgeo.gdalconst.html

This is only necessary if the documentation is not built by sphinx. In other cases, like e.g. matplotlib, the
intersphinx extension is used.
"""
from docutils import nodes


def setup(app):
    app.add_role('osgeo:class', autolink('https://gdal.org/api/python/osgeo.{0}.html#osgeo.{0}.{1}'))
    app.add_role('osgeo:func', autolink('https://gdal.org/api/python/osgeo.{0}.html#osgeo.{0}.{1}'))
    app.add_role('osgeo:meth', autolink('https://gdal.org/api/python/osgeo.{0}.html#osgeo.{0}.{1}.{2}'))
    app.add_role('osgeo:module', autolink('https://gdal.org/api/python/osgeo.{0}.html'))


def autolink(pattern):
    def role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        url = pattern.format(*text.split('.'))
        node = nodes.reference(rawtext, text, refuri=url, **options)
        return [node], []
    return role
