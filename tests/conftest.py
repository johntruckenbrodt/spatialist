import os
import pytest


@pytest.fixture
def travis():
    return 'TRAVIS' in os.environ.keys()
