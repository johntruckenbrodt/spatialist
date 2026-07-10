import pytest
from osgeo import ogr
from spatialist.vector import bbox, intersect2 as intersect


@pytest.mark.parametrize(
    "extent1, extent2, expected_extent, expected_area",
    [
        # simple partial overlap
        (
                {"xmin": 0, "xmax": 2, "ymin": 0, "ymax": 2},
                {"xmin": 1, "xmax": 3, "ymin": 1, "ymax": 3},
                {"xmin": 1, "xmax": 2, "ymin": 1, "ymax": 2},
                1,
        ),
        
        # full containment
        (
                {"xmin": 0, "xmax": 4, "ymin": 0, "ymax": 4},
                {"xmin": 1, "xmax": 2, "ymin": 1, "ymax": 2},
                {"xmin": 1, "xmax": 2, "ymin": 1, "ymax": 2},
                1,
        ),
        
        # touching edge only: no polygon area intersection
        (
                {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1},
                {"xmin": 1, "xmax": 2, "ymin": 0, "ymax": 1},
                None,
                0,
        ),
        
        # no overlap
        (
                {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1},
                {"xmin": 2, "xmax": 3, "ymin": 2, "ymax": 3},
                None,
                0,
        ),
    ],
)
def test_intersect_bbox(extent1, extent2, expected_extent, expected_area):
    with bbox(extent1, crs=4326) as vec1, bbox(extent2, crs=4326) as vec2:
        out = intersect(vec1, vec2)
        
        if expected_extent is None:
            assert out is None
            return
        
        assert out is not None
        assert out.nfeatures == 1
        assert out.getArea() == pytest.approx(expected_area)
        
        for key, value in expected_extent.items():
            assert out.extent[key] == pytest.approx(value)


def test_intersect_preserves_fields():
    with bbox(
            {"xmin": 0, "xmax": 2, "ymin": 0, "ymax": 2},
            crs=4326,
    ) as vec1, bbox(
        {"xmin": 1, "xmax": 3, "ymin": 1, "ymax": 3},
        crs=4326,
    ) as vec2:
        vec1.addfield("name", ogr.OFTString, values=["a"])
        vec2.addfield("name", ogr.OFTString, values=["b"])
        
        out = intersect(vec1, vec2)
        
        assert out is not None
        assert out.nfeatures == 1
        
        # duplicate field names should be disambiguated
        assert "input_name" in out.fieldnames
        assert "method_name" in out.fieldnames
        
        feat = out.getFeatureByIndex(0)
        assert feat.GetField("input_name") == "a"
        assert feat.GetField("method_name") == "b"
