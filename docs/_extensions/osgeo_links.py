"""
osgeo documentation hyperlink extension module for sphinx documentation
John Truckenbrodt 2018

this enables to place short rst link directives for osgeo gdal module methods, classes and functions.
Instead of each time placing `gdal.Dataset <https://gdal.org/python/osgeo.gdal.Dataset-class.html>`__ into
a docstring for linking to the class documentation, one can just type the following:

:osgeo:class:`gdal.Dataset`             -> https://gdal.org/python/osgeo.gdal.Dataset-class.html
:osgeo:meth:`gdal.Dataset.ReadAsArray`  -> https://gdal.org/python/osgeo.gdal.Dataset-class.html#ReadAsArray
:osgeo:func:`ogr.CreateGeometryFromWkt` -> https://gdal.org/python/osgeo.ogr-module.html#CreateGeometryFromWkt

This is only necessary if the documentation is not build by sphinx. In other cases, like e.g. matplotlib, the
intersphinx extension is used.
"""
from docutils import nodes


def setup(app):
    app.add_role('osgeo:class', autolink('https://gdal.org/python/osgeo.{0}.{1}-class.html'))
    app.add_role('osgeo:func', autolink('https://gdal.org/python/osgeo.{0}-module.html#{1}'))
    app.add_role('osgeo:meth', autolink('https://gdal.org/python/osgeo.{0}.{1}-class.html#{2}'))


def autolink(pattern):
    def role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        url = pattern.format(*text.split('.'))
        node = nodes.reference(rawtext, text, refuri=url, **options)
        return [node], []
    return role
