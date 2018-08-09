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
