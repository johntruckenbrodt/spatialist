"""
YAML GDAL/OGR driver list extension module for sphinx documentation
John Truckenbrodt 2019, 2026

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
from importlib.resources import files


class ListDrivers(Directive):
    required_arguments = 1
    
    def run(self):
        base = 'drivers_{}.yml'.format(self.arguments[0])
        with files('spatialist').joinpath(base).open('r') as f:
            drivers = yaml.safe_load(f)
        
        lst = nodes.bullet_list()
        for extension, name in drivers.items():
            item = nodes.list_item()
            lst += item
            item += nodes.paragraph(text='.{} ({})'.format(extension, name))
        
        return [lst]


def setup(app):
    app.add_directive('list_drivers', ListDrivers)
