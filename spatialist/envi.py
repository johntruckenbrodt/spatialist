##############################################################
# ENVI header management
# John Truckenbrodt 2015-2019
##############################################################
"""
This module offers functionality for editing ENVI header files
"""
import re
import zipfile as zf
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

    Examples
    --------

    >>> from spatialist.envi import HDRobject
    >>> with HDRobject('E:/test.hdr') as hdr:
    >>>     hdr.band_names = ['one', 'two']
    >>>     print(hdr)
    >>>     hdr.write()
    """
    
    def __init__(self, data=None):
        self.filename = data if isinstance(data, str) else None
        if isinstance(data, str):
            if re.search('.hdr$', data):
                args = self.__hdr2dict()
            else:
                raise RuntimeError('the data does not seem to be a ENVI HDR file')
        elif data is None:
            args = {'bands': 1,
                    'header_offset': 0,
                    'file_type': 'ENVI Standard',
                    'interleave': 'bsq',
                    'sensor_type': 'Unknown',
                    'byte_order': 0,
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
        for item in ['description', 'acquisition_time', 'samples', 'lines', 'bands', 'header_offset', 'file_type',
                     'data_type', 'data_ignore_value', 'interleave', 'sensor_type', 'byte_order', 'map_info',
                     'coordinate_system_string', 'wavelength_units', 'band_names']:
            if hasattr(self, item):
                value = getattr(self, item)
                if isinstance(value, (list, map)):
                    lines.append(item.replace('_', ' ') + ' = {' + ', '.join([str(x) for x in value]) + '}')
                elif item in ['description', 'band_names', 'coordinate_system_string']:
                    lines.append(item.replace('_', ' ') + ' = {' + str(value) + '}')
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
        if '.zip' in self.filename:
            match = re.search('.zip', self.filename)
            zip = self.filename[:match.end()]
            with zf.ZipFile(zip, 'r') as zip:
                member = self.filename[match.end():].strip('\\')
                content = zip.read(member)
            lines = content.decode().split('\n')
        else:
            with open(self.filename, 'r') as infile:
                lines = infile.readlines()
        i = 0
        out = dict()
        while i < len(lines):
            line = lines[i].strip('\r\n')
            if '=' in line:
                if '{' in line and '}' not in line:
                    while '}' not in line:
                        i += 1
                        line += lines[i].strip('\n').lstrip()
                line = list(filter(None, re.split(r'\s+=\s+', line)))
                line[1] = re.split(',[ ]*', line[1].strip('{}'))
                key = line[0].replace(' ', '_')
                val = line[1] if len(line[1]) > 1 else line[1][0]
                out[key] = parse_literal(val)
            i += 1
        if 'band_names' in out.keys() and not isinstance(out['band_names'], list):
            out['band_names'] = [out['band_names']]
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
