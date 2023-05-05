"""
YAML GDAL/OGR driver list extension module for sphinx documentation
John Truckenbrodt 2019

With this, spatialist's yaml files listing known file extensions can be read and listed in the sphinx documentation.
The argument is either 'vector' or 'raster' listing the content of files 'drivers_vector.yml' and 'drivers_raster.yml'
respectively.

Example usage in a docstring:

the following file extensions are auto-detected:

.. list_drivers:: vector

"""
import yaml
from docutils import nodes
from docutils.parsers.rst import Directive
from pkg_resources import resource_filename


class ListDrivers(Directive):
    required_arguments = 1
    
    def run(self):
        base = 'drivers_{}.yml'.format(self.arguments[0])
        filename = resource_filename('spatialist', base)
        drivers = yaml.safe_load(open(filename))
        
        lst = nodes.bullet_list()
        for extension, name in drivers.items():
            item = nodes.list_item()
            lst += item
            item += nodes.paragraph(text='.{} ({})'.format(extension, name))
        
        return [lst]


def setup(app):
    app.add_directive('list_drivers', ListDrivers)
