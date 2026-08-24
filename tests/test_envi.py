import os
import pytest
from spatialist.envi import hdr, HDRobject


def test_envi(tmpdir):
    with pytest.raises(RuntimeError):
        obj = HDRobject(1)
    with pytest.raises(RuntimeError):
        obj = HDRobject('foobar')
    outname = os.path.join(str(tmpdir), 'test')
    with HDRobject() as header:
        header.band_names = ['band1']
        header.write(outname)
    outname += '.hdr'
    with HDRobject(outname) as header:
        assert header.band_names == ['band1']
        vals = vars(header)
    with HDRobject(vals) as header:
        assert header.byte_order == 0
    hdr(vals, outname + '2')
