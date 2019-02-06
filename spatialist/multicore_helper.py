import os
import tempfile
import dill

try:
    import pathos.multiprocessing as mp
except ImportError:
    pass

from spatialist.ancillary import HiddenPrints

if __name__ == '__main__':
    
    tmpfile = os.path.join(tempfile.gettempdir(), 'spatialist_dump')
    with open(tmpfile, 'rb') as tmp:
        func, cores, processlist = dill.load(tmp)
    
    processlist = [dill.dumps([func, x]) for x in processlist]
    
    
    def wrapper(job):
        import dill
        function, proc = dill.loads(job)
        return function(**proc)
    
    
    with HiddenPrints():
        # start pool of processes and do the work
        try:
            pool = mp.Pool(processes=cores)
        except NameError:
            raise ImportError("package 'pathos' could not be imported")
        results = pool.imap(wrapper, processlist)
        pool.close()
        pool.join()
    
    outlist = list(results)
    
    # evaluate the return of the processing function;
    # if any value is not None then the whole list of results is returned
    eval = [x for x in outlist if x is not None]
    if len(eval) == 0:
        out = None
    else:
        out = outlist
    
    with open(tmpfile, 'wb') as tmp:
        dill.dump(out, tmp, byref=False)
