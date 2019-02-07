#################################################################
# helper script to be able to use function ancillary.multicore on
# Windows operating systems
# John Truckenbrodt 2019
#################################################################
import os
import tempfile
import dill

try:
    import pathos.multiprocessing as mp
except ImportError:
    pass

from spatialist.ancillary import HiddenPrints

if __name__ == '__main__':
    
    # de-serialize the arguments written by function ancillary.multicore
    tmpfile = os.path.join(tempfile.gettempdir(), 'spatialist_dump')
    with open(tmpfile, 'rb') as tmp:
        func, cores, processlist = dill.load(tmp)
    
    # serialize the job arguments to be able to pass them to the processes
    processlist = [dill.dumps([func, x]) for x in processlist]
    
    # a simple wrapper to execute the jobs in the sub-processes
    # re-import of modules and passing pickled variables is necessary since on
    # Windows the environment is not shared between parent and child processes
    def wrapper(job):
        import dill
        function, proc = dill.loads(job)
        return function(**proc)
    
    # hide print messages in the sub-processes
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
    
    # serialize and write the output list to be able to read it in function ancillary.multicore
    with open(tmpfile, 'wb') as tmp:
        dill.dump(out, tmp, byref=False)
