##############################################################
# core routines for software spatialist
# John Truckenbrodt 2014-2020
##############################################################
"""
This script gathers central functions and classes for general applications
"""
import dill
import string
import shutil
import tempfile
import platform
import tblib.pickling_support
from io import StringIO
from urllib.parse import urlparse, urlunparse, urlencode
from builtins import str
import re
import sys
import fnmatch
import inspect
import itertools
import os
import subprocess as sp
import tarfile as tf
import zipfile as zf
import numpy as np

try:
    import pathos.multiprocessing as mp
except ImportError:
    pass


class HiddenPrints:
    """
    | Suppress console stdout prints, i.e. redirect them to a temporary string object.
    | Adapted from https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python

    Examples
    --------
    >>> with HiddenPrints():
    >>>     print('foobar')
    >>> print('foobar')
    """
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = StringIO()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def decode_filter(text, encoding='utf-8'):
    """
    decode a binary object to str and filter out non-printable characters
    
    Parameters
    ----------
    text: bytes
        the binary object to be decoded
    encoding: str
        the encoding to be used

    Returns
    -------
    str
        the decoded and filtered string
    """
    if text is not None:
        text = text.decode(encoding, errors='ignore')
        printable = set(string.printable)
        text = filter(lambda x: x in printable, text)
        return ''.join(list(text))
    else:
        return None


def dictmerge(x, y):
    """
    merge two dictionaries
    """
    z = x.copy()
    z.update(y)
    return z


# todo consider using itertools.chain like in function finder
def dissolve(inlist):
    """
    list and tuple flattening
    
    Parameters
    ----------
    inlist: list
        the list with sub-lists or tuples to be flattened
    
    Returns
    -------
    list
        the flattened result
    
    Examples
    --------
    >>> dissolve([[1, 2], [3, 4]])
    [1, 2, 3, 4]
    
    >>> dissolve([(1, 2, (3, 4)), [5, (6, 7)]])
    [1, 2, 3, 4, 5, 6, 7]
    """
    out = []
    for i in inlist:
        i = list(i) if isinstance(i, tuple) else i
        out.extend(dissolve(i)) if isinstance(i, list) else out.append(i)
    return out


def finder(target, matchlist, foldermode=0, regex=False, recursive=True):
    """
    function for finding files/folders in folders and their subdirectories

    Parameters
    ----------
    target: str or list of str
        a directory, zip- or tar-archive or a list of them to be searched
    matchlist: list
        a list of search patterns
    foldermode: int
        * 0: only files
        * 1: files and folders
        * 2: only folders
    regex: bool
        are the search patterns in matchlist regular expressions or unix shell standard (default)?
    recursive: bool
        search target recursively into all subdirectories or only in the top level?
        This is currently only implemented for parameter `target` being a directory.

    Returns
    -------
    list of str
        the absolute names of files/folders matching the patterns
    """
    if foldermode not in [0, 1, 2]:
        raise ValueError("'foldermode' must be either 0, 1 or 2")
    
    # match patterns
    if isinstance(target, str):
        
        pattern = r'|'.join(matchlist if regex else [fnmatch.translate(x) for x in matchlist])
        
        if os.path.isdir(target):
            if recursive:
                out = dissolve([[os.path.join(root, x)
                                 for x in dirs + files
                                 if re.search(pattern, x)]
                                for root, dirs, files in os.walk(target)])
            else:
                out = [os.path.join(target, x)
                       for x in os.listdir(target)
                       if re.search(pattern, x)]
            
            if foldermode == 0:
                out = [x for x in out if not os.path.isdir(x)]
            if foldermode == 2:
                out = [x for x in out if os.path.isdir(x)]
            
            return sorted(out)
        
        elif os.path.isfile(target):
            if zf.is_zipfile(target):
                with zf.ZipFile(target, 'r') as zip:
                    out = [os.path.join(target, name)
                           for name in zip.namelist()
                           if re.search(pattern, os.path.basename(name.strip('/')))]
                
                if foldermode == 0:
                    out = [x for x in out if not x.endswith('/')]
                elif foldermode == 1:
                    out = [x.strip('/') for x in out]
                elif foldermode == 2:
                    out = [x.strip('/') for x in out if x.endswith('/')]
                
                return sorted(out)
            
            elif tf.is_tarfile(target):
                tar = tf.open(target)
                out = [name for name in tar.getnames()
                       if re.search(pattern, os.path.basename(name.strip('/')))]
                
                if foldermode == 0:
                    out = [x for x in out if not tar.getmember(x).isdir()]
                elif foldermode == 2:
                    out = [x for x in out if tar.getmember(x).isdir()]
                
                tar.close()
                
                out = [os.path.join(target, x) for x in out]
                
                return sorted(out)
            
            else:
                raise TypeError("if parameter 'target' is a file, "
                                "it must be a zip or tar archive:\n    {}"
                                .format(target))
        else:
            raise TypeError("if parameter 'target' is of type str, "
                            "it must be a directory or a file:\n    {}"
                            .format(target))
    
    elif isinstance(target, list):
        groups = [finder(x, matchlist, foldermode, regex, recursive) for x in target]
        return list(itertools.chain(*groups))
    
    else:
        raise TypeError("parameter 'target' must be of type str or list")


def multicore(function, cores, multiargs, **singleargs):
    """
    wrapper for multicore process execution

    Parameters
    ----------
    function
        individual function to be applied to each process item
    cores: int
        the number of subprocesses started/CPUs used;
        this value is reduced in case the number of subprocesses is smaller
    multiargs: dict
        a dictionary containing sub-function argument names as keys and lists of arguments to be
        distributed among the processes as values
    singleargs
        all remaining arguments which are invariant among the subprocesses

    Returns
    -------
    None or list
        the return of the function for all subprocesses

    Notes
    -----
    - all `multiargs` value lists must be of same length, i.e. all argument keys must be explicitly defined for each
      subprocess
    - all function arguments passed via `singleargs` must be provided with the full argument name and its value
      (i.e. argname=argval); default function args are not accepted
    - if the processes return anything else than None, this function will return a list of results
    - if all processes return None, this function will be of type void

    Examples
    --------
    >>> def add(x, y, z):
    >>>     return x + y + z
    >>> multicore(add, cores=2, multiargs={'x': [1, 2]}, y=5, z=9)
    [15, 16]
    >>> multicore(add, cores=2, multiargs={'x': [1, 2], 'y': [5, 6]}, z=9)
    [15, 17]

    See Also
    --------
    :mod:`pathos.multiprocessing`
    """
    tblib.pickling_support.install()
    
    # compare the function arguments with the multi and single arguments and raise errors if mismatches occur
    if sys.version_info >= (3, 0):
        check = inspect.getfullargspec(function)
        varkw = check.varkw
    else:
        check = inspect.getargspec(function)
        varkw = check.keywords
    
    if not check.varargs and not varkw:
        multiargs_check = [x for x in multiargs if x not in check.args]
        singleargs_check = [x for x in singleargs if x not in check.args]
        if len(multiargs_check) > 0:
            raise AttributeError('incompatible multi arguments: {0}'.format(', '.join(multiargs_check)))
        if len(singleargs_check) > 0:
            raise AttributeError('incompatible single arguments: {0}'.format(', '.join(singleargs_check)))
    
    # compare the list lengths of the multi arguments and raise errors if they are of different length
    arglengths = list(set([len(multiargs[x]) for x in multiargs]))
    if len(arglengths) > 1:
        raise AttributeError('multi argument lists of different length')
    
    # prevent starting more threads than necessary
    cores = cores if arglengths[0] >= cores else arglengths[0]
    
    # create a list of dictionaries each containing the arguments for individual
    # function calls to be passed to the multicore processes
    processlist = [dictmerge(dict([(arg, multiargs[arg][i]) for arg in multiargs]), singleargs)
                   for i in range(len(multiargs[list(multiargs.keys())[0]]))]
    
    if platform.system() == 'Windows':
        
        # in Windows parallel processing needs to strictly be in a "if __name__ == '__main__':" wrapper
        # it was thus necessary to outsource this to a different script and try to serialize all input for sharing objects
        # https://stackoverflow.com/questions/38236211/why-multiprocessing-process-behave-differently-on-windows-and-linux-for-global-o
        
        # a helper script to perform the parallel processing
        script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'multicore_helper.py')
        
        # a temporary file to write the serialized function variables
        tmpfile = os.path.join(tempfile.gettempdir(), 'spatialist_dump')
        
        # check if everything can be serialized
        if not dill.pickles([function, cores, processlist]):
            raise RuntimeError('cannot fully serialize function arguments;\n'
                               ' see https://github.com/uqfoundation/dill for supported types')
        
        # write the serialized variables
        with open(tmpfile, 'wb') as tmp:
            dill.dump([function, cores, processlist], tmp, byref=False)
        
        # run the helper script
        proc = sp.Popen([sys.executable, script], stdin=sp.PIPE, stderr=sp.PIPE)
        out, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(err.decode())
        
        # retrieve the serialized output of the processing which was written to the temporary file by the helper script
        with open(tmpfile, 'rb') as tmp:
            result = dill.load(tmp)
        return result
    else:
        results = None
        
        def wrapper(**kwargs):
            try:
                return function(**kwargs)
            except Exception as e:
                return ExceptionWrapper(e)
        
        # block printing of the executed function
        with HiddenPrints():
            # start pool of processes and do the work
            try:
                pool = mp.Pool(processes=cores)
            except NameError:
                raise ImportError("package 'pathos' could not be imported")
            results = pool.imap(lambda x: wrapper(**x), processlist)
            pool.close()
            pool.join()
        
        i = 0
        out = []
        for item in results:
            if isinstance(item, ExceptionWrapper):
                item.ee = type(item.ee)(str(item.ee) +
                                        "\n(called function '{}' with args {})"
                                        .format(function.__name__, processlist[i]))
                raise (item.re_raise())
            out.append(item)
            i += 1
        
        # evaluate the return of the processing function;
        # if any value is not None then the whole list of results is returned
        eval = [x for x in out if x is not None]
        if len(eval) == 0:
            return None
        else:
            return out


def add(x, y, z):
    """
    only a dummy function for testing the multicore function
    defining it in the test script is not possible since it cannot be serialized
    with a reference module that does not exist (i.e. the test script)
    """
    return x + y + z


class ExceptionWrapper(object):
    """
    | class for enabling traceback pickling in function multiprocess
    | https://stackoverflow.com/questions/6126007/python-getting-a-traceback-from-a-multiprocessing-process
    | https://stackoverflow.com/questions/34463087/valid-syntax-in-both-python-2-x-and-3-x-for-raising-exception
    """
    
    def __init__(self, ee):
        self.ee = ee
        __, __, self.tb = sys.exc_info()
    
    def re_raise(self):
        if sys.version_info[0] == 3:
            def reraise(tp, value, tb=None):
                raise tp.with_traceback(tb)
        else:
            exec("def reraise(tp, value, tb=None):\n    raise tp, value, tb\n")
        reraise(self.ee, None, self.tb)


def parse_literal(x):
    """
    return the smallest possible data type for a string or list of strings

    Parameters
    ----------
    x: str or list
        a string to be parsed

    Returns
    -------
    int, float or str
        the parsing result
    
    Examples
    --------
    >>> isinstance(parse_literal('1.5'), float)
    True
    
    >>> isinstance(parse_literal('1'), int)
    True
    
    >>> isinstance(parse_literal('foobar'), str)
    True
    """
    if isinstance(x, list):
        return [parse_literal(y) for y in x]
    elif isinstance(x, (bytes, str)):
        try:
            return int(x)
        except ValueError:
            try:
                return float(x)
            except ValueError:
                return x
    else:
        raise TypeError('input must be a string or a list of strings')


class Queue(object):
    """
    classical queue implementation
    """
    
    def __init__(self, inlist=None):
        self.stack = [] if inlist is None else inlist
    
    def empty(self):
        return len(self.stack) == 0
    
    def length(self):
        return len(self.stack)
    
    def push(self, x):
        self.stack.append(x)
    
    def pop(self):
        if not self.empty():
            val = self.stack[0]
            del self.stack[0]
            return val


def rescale(inlist, newrange=(0, 1)):
    """
    rescale the values in a list between the values in newrange (a tuple with the new minimum and maximum)
    """
    OldMax = max(inlist)
    OldMin = min(inlist)
    
    if OldMin == OldMax:
        raise RuntimeError('list contains of only one unique value')
    
    OldRange = OldMax - OldMin
    NewRange = newrange[1] - newrange[0]
    result = [(((float(x) - OldMin) * NewRange) / OldRange) + newrange[0] for x in inlist]
    return result


def run(cmd, outdir=None, logfile=None, inlist=None, void=True, errorpass=False, env=None):
    """
    | wrapper for subprocess execution including logfile writing and command prompt piping
    | this is a convenience wrapper around the :mod:`subprocess` module and calls
      its class :class:`~subprocess.Popen` internally.
    
    Parameters
    ----------
    cmd: list
        the command arguments
    outdir: str or None
        the directory to execute the command in
    logfile: str or None
        a file to write stdout to
    inlist: list or None
        a list of arguments passed to stdin, i.e. arguments passed to interactive input of the program
    void: bool
        return stdout and stderr?
    errorpass: bool
        if False, a :class:`subprocess.CalledProcessError` is raised if the command fails
    env: dict or None
        the environment to be passed to the subprocess

    Returns
    -------
    None or Tuple
        a tuple of (stdout, stderr) if `void` is False otherwise None
    """
    cmd = [str(x) for x in dissolve(cmd)]
    if outdir is None:
        outdir = os.getcwd()
    log = sp.PIPE if logfile is None else open(logfile, 'a')
    proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=log, stderr=sp.PIPE, cwd=outdir, env=env)
    instream = None if inlist is None \
        else ''.join([str(x) + '\n' for x in inlist]).encode('utf-8')
    out, err = proc.communicate(instream)
    out = decode_filter(out)
    err = decode_filter(err)
    if not errorpass and proc.returncode != 0:
        raise sp.CalledProcessError(proc.returncode, cmd, err)
    # add line for separating log entries of repeated function calls
    if logfile:
        log.write('#####################################################################\n')
        log.close()
    if not void:
        return out, err


class Stack(object):
    """
    classical stack implementation
    input can be a list, a single value or None (i.e. Stack())
    """
    
    def __init__(self, inlist=None):
        if isinstance(inlist, list):
            self.stack = inlist
        elif inlist is None:
            self.stack = []
        else:
            self.stack = [inlist]
    
    def empty(self):
        """
        check whether stack is empty
        """
        return len(self.stack) == 0
    
    def flush(self):
        """
        empty the stack
        """
        self.stack = []
    
    def length(self):
        """
        get the length of the stack
        """
        return len(self.stack)
    
    def push(self, x):
        """
        append items to the stack; input can be a single value or a list
        """
        if isinstance(x, list):
            for item in x:
                self.stack.append(item)
        else:
            self.stack.append(x)
    
    def pop(self):
        """
        return the last stack element and delete it from the list
        """
        if not self.empty():
            val = self.stack[-1]
            del self.stack[-1]
            return val


def union(a, b):
    """
    union of two lists
    """
    return list(set(a) & set(b))


def urlQueryParser(url, querydict):
    """
    parse a url query
    """
    address_parse = urlparse(url)
    return urlunparse(address_parse._replace(query=urlencode(querydict)))


def which(program, mode=os.F_OK | os.X_OK):
    """
    | mimics UNIX's which
    | taken from this post: http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    | can be replaced by :func:`shutil.which()` starting from Python 3.3
    
    Parameters
    ----------
    program: str
        the program to be found
    mode: os.F_OK or os.X_OK
        the mode of the found file, i.e. file exists or file  is executable; see :func:`os.access`

    Returns
    -------
    str or None
        the full path and name of the command
    """
    if sys.version_info >= (3, 3):
        return shutil.which(program, mode=mode)
    else:
        def is_exe(fpath, mode):
            return os.path.isfile(fpath) and os.access(fpath, mode)
        
        fpath, fname = os.path.split(program)
        if fpath:
            if is_exe(program, mode):
                return program
        else:
            for path in os.environ['PATH'].split(os.path.pathsep):
                path = path.strip('"')
                exe_files = [os.path.join(path, program), os.path.join(path, program + '.exe')]
                for exe_file in exe_files:
                    if is_exe(exe_file, mode):
                        return exe_file
        return None


def parallel_apply_along_axis(func1d, axis, arr, cores=4, *args, **kwargs):
    """
    Like :func:`numpy.apply_along_axis()` but using multiple threads.
    Adapted from `here <https://stackoverflow.com/questions/45526700/
    easy-parallelization-of-numpy-apply-along-axis>`_.

    Parameters
    ----------
    func1d: function
        the function to be applied
    axis: int
        the axis along which to apply `func1d`
    arr: numpy.ndarray
        the input array
    cores: int
        the number of parallel cores
    args: any
        Additional arguments to `func1d`.
    kwargs: any
        Additional named arguments to `func1d`.

    Returns
    -------
    numpy.ndarray
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)
    
    def unpack(arguments):
        func1d, axis, arr, args, kwargs = arguments
        return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
    
    if cores <= 0:
        raise ValueError('cores must be larger than 0')
    elif cores == 1:
        return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
    else:
        chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
                  for sub_arr in np.array_split(arr, mp.cpu_count())]
        
        pool = mp.Pool(cores)
        individual_results = pool.map(unpack, chunks)
        # Freeing the workers:
        pool.close()
        pool.join()
        
        return np.concatenate(individual_results)
