import pytest
from spatialist.sqlite_util import sqlite_setup, __Handler


def test_sqlite():
    with pytest.raises(RuntimeError):
        con = sqlite_setup(extensions='spatialite')
    con = sqlite_setup(extensions=['spatialite'])
    con.close()
    con = __Handler()
    assert sorted(con.version.keys()) == ['sqlite']
    
    con = __Handler(extensions=['spatialite'])
    assert sorted(con.version.keys()) == ['spatialite', 'sqlite']
    assert 'spatial_ref_sys' in con.get_tablenames()
