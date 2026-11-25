"""High-level workflows used by the UI (and usable from notebooks / scripts)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import armenia, gee, recommendations, risk, vision


@dataclass
class Outcome:
    """Everything a page needs to render an analysis."""
    title: str
    source: str                       # upload | sample | gee | hotspot
    lat: float | None
    lon: float | None
    marz: str | None
    issues: dict[str, dict[str, Any]]
    risk: risk.RiskReport
    risk_image_only: risk.RiskReport
    plan: dict
    vision: vision.AnalysisResult | None = None
    gee_stats: dict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def top_issue(self) -> str | None:
        det = {k: v for k, v in self.issues.items() if v.get("severity", 0) >= 15}
        return max(det, key=lambda k: det[k]["severity"]) if det else None

    def record(self) -> dict:
        """Row for the database."""
        fr = self.vision.fractions if self.vision else {}
        return {
            "source": self.source, "title": self.title, "lat": self.lat, "lon": self.lon, "marz": self.marz,
            "engine": self.vision.engine if self.vision else ("Earth Engine" if self.gee_stats else "տեղեկատու"),
            "area_ha": self.vision.total_area_ha if self.vision else None,
            "risk_score": round(self.risk.overall, 1), "risk_level": self.risk.level, "top_issue": self.top_issue,
            "fractions": {k: round(v, 4) for k, v in fr.items()},
            "issues": {k: {kk: vv for kk, vv in v.items() if kk != "mask"} for k, v in self.issues.items()},
            "risk": self.risk.to_dict(), "recommendations": self.plan,
        }


def _marz_for(lat, lon):
    if lat is None:
        return None
    near = armenia.nearest_hotspots(lat, lon, k=1, max_km=80)
    return near[0][0].marz if near else None


def analyze_image(data: bytes, filename: str, *, gsd_m: float | None, sensitivity: float, engine: str,
                  lat: float | None, lon: float | None, title: str, source: str = "upload",
                  segmenter=None, detector=None) -> Outcome:
    rgb, bands = vision.load_image(data, filename)
    res = vision.analyze(rgb, bands, gsd_m=gsd_m, sensitivity=sensitivity, engine=engine,
                         segmenter=segmenter, detector=detector)
    marz = _marz_for(lat, lon)
    r_ctx = risk.assess(res.issues, lat, lon)
    r_img = risk.assess(res.issues)
    plan = recommendations.recommend(res.issues, lat, lon, marz)
    return Outcome(title, source, lat, lon, marz, res.issues, r_ctx, r_img, plan, vision=res, notes=res.notes)


def analyze_location(lat: float, lon: float, radius_m: float = 3000, use_gee: bool = True,
                     segmenter=None, detector=None) -> Outcome:
    """Live Earth Engine analysis when available; otherwise a reference profile from hotspot data."""
    marz = _marz_for(lat, lon)
    title = f"Տեղանք {lat:.4f}, {lon:.4f}"
    notes: list[str] = []

    if use_gee and gee.init():
        stats = gee.location_stats(round(lat, 4), round(lon, 4), radius_m)
        issues = gee.issues_from_stats(stats)
        res = None
        try:  # also run computer vision on the Sentinel-2 true-colour thumbnail
            png = gee.thumbnail_png(lat, lon, radius_m)
            rgb, _ = vision.load_image(png, "s2.png")
            res = vision.analyze(rgb, None, gsd_m=2 * radius_m / rgb.shape[1], segmenter=segmenter, detector=detector)
            for k, v in res.issues.items():  # fuse: keep stronger evidence
                if k in issues and v["severity"] > issues[k]["severity"]:
                    issues[k]["severity"] = round((issues[k]["severity"] + v["severity"]) / 2, 1)
                    issues[k]["detected"] = issues[k]["severity"] >= 15
            if res.issues["mining"]["severity"] > issues["mining"]["severity"]:
                issues["mining"].update(severity=res.issues["mining"]["severity"],
                                        detected=res.issues["mining"]["detected"],
                                        evidence="Sentinel-2 պատկերում հայտնաբերվել են մերկ ապարներ / պոչանքներ")
        except Exception as exc:
            notes.append(f"Sentinel-2 պատկերի վերլուծությունը բաց է թողնվել․ {exc}")
        r_ctx = risk.assess(issues, lat, lon)
        r_img = risk.assess(issues)
        plan = recommendations.recommend(issues, lat, lon, marz)
        return Outcome(title, "gee", lat, lon, marz, issues, r_ctx, r_img, plan, vision=res,
                       gee_stats=stats, notes=notes)

    # Offline fallback: reference profile from the hotspot register
    _, err = gee.status()
    notes.append("Google Earth Engine-ը միացված չէ" + (f" ({err})" if err else "") +
                 "։ Ցույց է տրվում մոտակա փաստագրված թեժ կետերի վրա հիմնված տեղեկատու պատկեր, "
                 "ոչ թե իրական ժամանակի չափում։")
    issues = {k: {"label": vision.ISSUES[k]["label"], "detected": False, "severity": 0.0,
                  "evidence": "Մոտակայքում փաստագրված թեժ կետ չկա"} for k in vision.ISSUES}
    for h, d in armenia.nearest_hotspots(lat, lon, k=4, max_km=40):
        w = max(0.0, 1 - d / 40)
        for cat, mult in [(h.category, 1.0), *[(s, 0.6) for s in h.secondary]]:
            sev = round(h.severity * 18 * w * mult, 1)
            if sev > issues[cat]["severity"]:
                issues[cat].update(severity=sev, detected=sev >= 15, evidence=f"{h.name} ({d:.0f} կմ)")
    r = risk.assess(issues)
    plan = recommendations.recommend(issues, lat, lon, marz)
    return Outcome(title, "hotspot", lat, lon, marz, issues, r, r, plan, notes=notes)
