import os
import re
import platform
import zipfile as zf

from ctypes.util import find_library


def check_loading():
    try:
        conn = sqlite3.connect(':memory:')
        conn.enable_load_extension(True)
    except (sqlite3.OperationalError, AttributeError):
        raise RuntimeError


errormessage = 'sqlite3 does not support loading extensions and {}; ' \
               'please refer to the spatialist installation instructions'
try:
    import sqlite3
    
    check_loading()
except RuntimeError:
    try:
        from pysqlite2 import dbapi2 as sqlite3
        
        check_loading()
    except ImportError:
        raise RuntimeError(errormessage.format('pysqlite2 does not exist as alternative'))
    except RuntimeError:
        raise RuntimeError(errormessage.format('neither does pysqlite2'))


def sqlite_setup(driver=':memory:', extensions=None, verbose=False):
    """
    Setup a sqlite3 connection and load extensions to it.
    This function intends to simplify the process of loading extensions to `sqlite3`, which can be quite difficult
    depending on the version used.
    Particularly loading `spatialite` has caused quite some trouble. In recent distributions of Ubuntu this has
    become much easier due to a new apt package `libsqlite3-mod-spatialite`. For use in Windows, `spatialist` comes
    with its own `spatialite` DLL distribution.
    See `here <https://www.gaia-gis.it/fossil/libspatialite/wiki?name=mod_spatialite>`_ for more details on loading
    `spatialite` as an `sqlite3` extension.

    Parameters
    ----------
    driver: str
        the database file or (by default) an in-memory database
    extensions: list
        a list of extensions to load
    verbose: bool
        print loading information?

    Returns
    -------
    sqlite3.Connection
        the database connection

    Example
    -------
    >>> from spatialist.sqlite_util import sqlite_setup
    >>> conn = sqlite_setup(extensions=['spatialite'])
    """
    conn = __Handler(driver, extensions, verbose=verbose)
    return conn.conn


def spatialite_setup():
    if platform.system() == 'Windows':
        directory = os.path.join(os.path.expanduser('~'), '.spatialist')
        subdir = os.path.join(directory, 'mod_spatialite')
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        mod_spatialite = os.path.join(subdir, 'mod_spatialite.dll')
        if not os.path.isfile(mod_spatialite):
            source_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      'pkgs', 'mod_spatialite')
            # print('machine: {}'.format(platform.machine()))
            suffix = 'amd64' if platform.machine().endswith('64') else 'x86'
            source = os.path.join(source_dir, 'mod_spatialite-4.3.0a-win-{}.zip'.format(suffix))
            # print('extracting {} to {}'.format(os.path.basename(source), subdir))
            archive = zf.ZipFile(source, 'r')
            archive.extractall(subdir)
            archive.close()
        os.environ['PATH'] = '{}{}{}'.format(os.environ['PATH'], os.path.pathsep, subdir)


class __Handler(object):
    def __init__(self, driver=':memory:', extensions=None, verbose=False):
        self.conn = sqlite3.connect(driver)
        self.conn.enable_load_extension(True)
        self.extensions = []
        self.verbose = verbose
        if isinstance(extensions, list):
            for ext in extensions:
                self.load_extension(ext)
        elif extensions is not None:
            raise RuntimeError('extensions must either be a list or None')
        if verbose:
            print('using sqlite version {}'.format(self.version['sqlite']))
        if 'spatialite' in self.version.keys() and verbose:
            print('using spatialite version {}'.format(self.version['spatialite']))
    
    @property
    def version(self):
        out = {'sqlite': sqlite3.sqlite_version}
        cursor = self.conn.cursor()
        try:
            cursor.execute('SELECT spatialite_version()')
            spatialite_version = self.__encode(cursor.fetchall()[0][0])
            out['spatialite'] = spatialite_version
        except sqlite3.OperationalError:
            pass
        return out
    
    def get_tablenames(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM sqlite_master WHERE type="table"')
        names = [self.__encode(x[1]) for x in cursor.fetchall()]
        return names
    
    def load_extension(self, extension):
        if re.search('spatialite', extension):
            spatialite_setup()
            select = None
            # first try to load the dedicated mod_spatialite adapter
            for option in ['mod_spatialite', 'mod_spatialite.so', 'mod_spatialite.dll']:
                try:
                    self.conn.load_extension(option)
                    select = option
                    self.extensions.append(option)
                    if self.verbose:
                        print('loading extension {0} as {1}'.format(extension, option))
                    break
                except sqlite3.OperationalError as e:
                    if self.verbose:
                        print('{0}: {1}'.format(option, str(e)))
                    continue
            
            # if loading mod_spatialite fails try to load libspatialite directly
            if select is None:
                try:
                    self.__load_regular('spatialite')
                except RuntimeError as e:
                    raise type(e)(str(e) +
                                  '\nit can be installed via apt:'
                                  '\nsudo apt-get install libsqlite3-mod-spatialite')
            
            # initialize spatial support
            if 'spatial_ref_sys' not in self.get_tablenames():
                cursor = self.conn.cursor()
                if select is None:
                    # libspatialite extension
                    cursor.execute('SELECT InitSpatialMetaData();')
                else:
                    # mod_spatialite extension
                    cursor.execute('SELECT InitSpatialMetaData(1);')
                self.conn.commit()
        
        else:
            self.__load_regular(extension)
    
    def __load_regular(self, extension):
        options = []
        
        # create an extension library option starting with 'lib' without extension suffices;
        # e.g. 'libgdal' but not 'gdal.so'
        ext_base = self.__split_ext(extension)
        if not ext_base.startswith('lib'):
            ext_base = 'lib' + ext_base
        options.append(ext_base)
        
        # get the full extension library name; e.g. 'libgdal.so.20'
        ext_mod = find_library(extension.replace('lib', ''))
        if ext_mod is None:
            raise RuntimeError('no library found for extension {}'.format(extension))
        options.append(ext_mod)
        
        # loop through extension library name options and try to load them
        success = False
        for option in options:
            try:
                self.conn.load_extension(option)
                self.extensions.append(option)
                if self.verbose:
                    print('loading extension {0} as {1}'.format(extension, option))
                success = True
                break
            except sqlite3.OperationalError:
                continue
        
        if not success:
            raise RuntimeError('failed to load extension {}'.format(extension))
    
    @staticmethod
    def __split_ext(extension):
        base = extension
        while re.search(r'\.', base):
            base = os.path.splitext(base)[0]
        return base
    
    @staticmethod
    def __encode(string, encoding='utf-8'):
        if not isinstance(string, str):
            return string.encode(encoding)
        else:
            return string
