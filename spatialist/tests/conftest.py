import os
import pytest


@pytest.fixture
def travis():
    return 'TRAVIS' in os.environ.keys()


@pytest.fixture
def appveyor():
    return 'APPVEYOR' in os.environ.keys()


@pytest.fixture
def testdir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


@pytest.fixture
def testdata(testdir):
    out = {
        'tif': os.path.join(testdir, 'S1A__IW___A_20150309T173017_VV_grd_mli_geo_norm_db.tif'),
        'tif2': os.path.join(testdir, 'S1A__IW___D_20170309T054356_VV_grd_mli_geo_norm_db.tif'),
        'tif3': os.path.join(testdir, 'S1A__IW___D_20171210T054359_VV_grd_mli_geo_norm_db.tif'),
        'zip': os.path.join(testdir, 'demo_finder.zip'),
        'tar': os.path.join(testdir, 'demo_finder.tar.xz')
    }
    return out
