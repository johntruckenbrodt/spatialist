import pytest

from spatialist.auxil import utm_autodetect
from spatialist.vector import bbox


@pytest.mark.parametrize(
    "extent, expected_epsg",
    [
        # normal Northern Hemisphere case, Germany -> UTM zone 32N
        (
                {"xmin": 11.5, "xmax": 11.7, "ymin": 50.8, "ymax": 51.0},
                32632,
        ),
        
        # normal Southern Hemisphere case -> UTM zone 32S
        (
                {"xmin": 11.5, "xmax": 11.7, "ymin": -51.0, "ymax": -50.8},
                32732,
        ),
        
        # antimeridian crossing, Northern Hemisphere -> UTM zone 60N
        (
                {"xmin": 178.0, "xmax": -178.0, "ymin": 50.0, "ymax": 51.0},
                32660,
        ),
        
        # antimeridian crossing, Southern Hemisphere -> UTM zone 60S
        (
                {"xmin": 178.0, "xmax": -178.0, "ymin": -51.0, "ymax": -50.0},
                32760,
        ),
        
        # western hemisphere case -> UTM zone 10N
        (
                {"xmin": -124.0, "xmax": -123.0, "ymin": 45.0, "ymax": 46.0},
                32610,
        ),
        
        # near 180 but not crossing -> still UTM zone 60N
        (
                {"xmin": 176.0, "xmax": 179.0, "ymin": 10.0, "ymax": 11.0},
                32660,
        ),
        
        # just east of -180, not crossing -> UTM zone 1N
        (
                {"xmin": -179.0, "xmax": -176.0, "ymin": 10.0, "ymax": 11.0},
                32601,
        ),
    ],
)
def test_utm_autodetect(extent, expected_epsg):
    with bbox(extent, crs=4326) as vec:
        assert utm_autodetect(vec, crsOut="epsg") == expected_epsg
