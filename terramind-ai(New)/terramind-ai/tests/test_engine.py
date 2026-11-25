"""Engine tests — run with:  python -m pytest -q   (no Streamlit, torch or network needed)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from terramind import armenia, recommendations, risk, services, vision  # noqa: E402

SAMPLES = ROOT / "assets" / "samples"


def _run(name, lat=None, lon=None):
    return services.analyze_image((SAMPLES / name).read_bytes(), name, gsd_m=10, sensitivity=1.0,
                                  engine="auto", lat=lat, lon=lon, title=name)


@pytest.mark.parametrize("name,expected", [
    ("forest_clearing_tavush.png", "deforestation"),
    ("lake_sevan_bloom.png", "water_pollution"),
    ("ararat_drought_farmland.png", "soil_degradation"),
    ("mining_tailings_lori.png", "mining"),
])
def test_samples_detect_expected_issue(name, expected):
    o = _run(name)
    assert o.issues[expected]["detected"], o.issues[expected]
    json.dumps(o.record(), default=str)  # serialisable for the database


def test_clean_forest_has_low_risk():
    rng = np.random.default_rng(0)
    img = np.clip(np.array([34, 90, 36]) + rng.normal(0, 14, (400, 400, 3)), 0, 255).astype(np.uint8)
    res = vision.analyze(img, gsd_m=10)
    assert not res.issues["deforestation"]["detected"]
    assert risk.assess(res.issues).level == "Low"


def test_multispectral_path():
    h = w = 64
    bands = {k: np.full((h, w), v, np.float32) for k, v in
             dict(blue=0.04, green=0.06, red=0.04, red_edge=0.15, nir=0.35, swir1=0.15, swir2=0.08).items()}
    bands["nir"][:, :20] = 0.02; bands["swir1"][:, :20] = 0.01; bands["green"][:, :20] = 0.08  # water strip
    rgb = (np.dstack([bands["red"], bands["green"], bands["blue"]]) * 2550).clip(0, 255).astype(np.uint8)
    res = vision.analyze(rgb, bands, gsd_m=10)
    assert "բազմասպեկտրալ" in res.engine
    assert res.fractions["healthy_vegetation"] > 0.5
    assert res.fractions["clean_water"] + res.fractions["algal_bloom"] + res.fractions["turbid_water"] > 0.2


def test_location_context_raises_risk():
    o_far = _run("lake_sevan_bloom.png")
    o_near = _run("lake_sevan_bloom.png", 40.38, 45.25)
    assert o_near.risk.overall >= o_far.risk.overall


def test_reference_profile_offline():
    h = armenia.get_hotspot("ararat_aquifer")
    o = services.analyze_location(h.lat, h.lon, use_gee=False)
    assert o.issues["drought"]["detected"]
    assert o.plan["actions"]


def test_recommendation_markdown():
    plan = recommendations.recommend({"deforestation": {"severity": 70}}, marz="Տավուշ")
    md = recommendations.to_markdown(plan)
    assert "Fagus orientalis" in md and "0–6 ամիս" in md
