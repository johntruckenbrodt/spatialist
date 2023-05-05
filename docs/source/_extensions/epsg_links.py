"""
epsg code hyperlink extension module for sphinx documentation
John Truckenbrodt 2019

With this, a convenient documentation link can be made to spatialreference.org.
Instead of each time typing:

`EPSG:4326 <https://spatialreference.org/ref/epsg/4326/>`_

one can simply place the following in the document:

:epsg:`4326`
"""
from docutils import nodes


def setup(app):
    app.add_role('epsg', autolink('https://spatialreference.org/ref/epsg/{0}/'))


def autolink(pattern):
    def role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        """
        
        Parameters
        ----------
        name
            The role name used in the document
        rawtext
            The entire markup snippet, with role
        text
            The text marked with the role
        lineno
            The line number where rawtext appears in the input
        inliner
            The inliner instance that called us
        options
            Directive options for customization
        content
            The directive content for customization

        Returns
        -------
        tuple
            2 part tuple containing list of nodes to insert into the document and a list of system messages.
            Both are allowed to be empty.
        """
        url = pattern.format(*text.split('.'))
        node = nodes.reference(rawsource=rawtext, text='EPSG:{}'.format(text), refuri=url, **options)
        return [node], []
    return role
