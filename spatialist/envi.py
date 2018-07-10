##############################################################
# ENVI header management
# John Truckenbrodt 2015-2018
##############################################################
"""
This module offers functionality for editing ENVI header files
"""
import re
from .ancillary import parse_literal


def hdr(data, filename):
    """
    write ENVI header files

    Parameters
    ----------
    data: str or dict
        the file or dictionary to get the info from
    filename: str
        the HDR file to write

    Returns
    -------

    """
    hdrobj = data if isinstance(data, HDRobject) else HDRobject(data)
    hdrobj.write(filename)


class HDRobject(object):
    """
    ENVI HDR info handler

    Parameters
    ----------
    data: str, dict or None
        the file or dictionary to get the info from; If None (default), an object with default values for an empty
        raster file is returned

    Returns
    -------
    the hdr metadata handler

    Examples
    --------
    # open a HDR file and rewrite it with changed band names
    >>>with HDRobject('E:/test.hdr') as hdr:
    >>>    hdr.band_names = ['one', 'two']
    >>>    hdr.write()
    """

    def __init__(self, data=None):
        self.filename = data if isinstance(data, str) else None
        if isinstance(data, str):
            if re.search('.hdr$', data):
                args = self.__hdr2dict()
                if 'band_names' in args.keys() and isinstance(args['band_names'], str):
                    args['band_names'] = [args['band_names']]
            else:
                raise RuntimeError('the data does not seem to be a ENVI HDR file')
        elif data is None:
            args = {'bands': 1,
                    'header_offset': 0,
                    'file_type': 'ENVI Standard',
                    'interleave': 'bsq',
                    'sensor_type': 'Unknown',
                    'byte_order': 1,
                    'wavelength_units': 'Unknown',
                    'samples': 0,
                    'lines': 0}
        elif isinstance(data, dict):
            args = data

        else:
            raise RuntimeError('parameter data must be of type str, dict or None')

        for arg in args:
            setattr(self, arg, args[arg])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def __str__(self):
        lines = ['ENVI']
        for item in ['description', 'samples', 'lines', 'bands', 'header_offset', 'file_type', 'data_type',
                     'interleave', 'sensor_type', 'byte_order', 'map_info',
                     'coordinate_system_string', 'wavelength_units', 'band_names']:
            if hasattr(self, item):
                value = getattr(self, item)
                if isinstance(value, list):
                    lines.append(item.replace('_', ' ') + ' = {' + ', '.join([str(x) for x in value]) + '}')
                elif item in ['description', 'band_names', 'coordinate_system_string']:
                    lines.append(item.replace('_', ' ') + ' = {' + value + '}')
                else:
                    lines.append(item.replace('_', ' ') + ' = ' + str(value) + '')
        return '\n'.join(lines)

    def __hdr2dict(self):
        """
        read a HDR file into a dictionary
        http://gis.stackexchange.com/questions/48618/how-to-read-write-envi-metadata-using-gdal
        Returns
        -------
        dict
            the hdr file metadata attributes
        """
        with open(self.filename, 'r') as infile:
            hdr = infile.read()
        # Get all 'key = {val}' type matches
        regex = re.compile(r'^(.+?)\s*=\s*({\s*.*?\n*.*?})$', re.M | re.I)
        matches = regex.findall(hdr)

        # Remove them from the header
        subhdr = regex.sub('', hdr)

        # Get all 'key = val' type matches
        regex = re.compile(r'^(.+?)\s*=\s*(.*?)$', re.M | re.I)
        matches.extend(regex.findall(subhdr))

        out = dict(matches)

        for key, val in out.items():
            out[key] = parse_literal(val)
            if re.search(' ', key):
                out[key.replace(' ', '_')] = out.pop(key)

        return out

    def write(self, filename='same'):
        """
        write object to an ENVI header file
        """
        if filename == 'same':
            filename = self.filename
        if not filename.endswith('.hdr'):
            filename += '.hdr'
        with open(filename, 'w') as out:
            out.write(self.__str__())
