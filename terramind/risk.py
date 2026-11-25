"""Environmental risk assessment: combines detected issue severity with location context."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import armenia

# Status palette (good / warning / serious / critical) — always shown with a text label.
LEVELS = [(25, "Low", "#0ca30c"), (50, "Moderate", "#fab219"), (75, "High", "#ec835a"), (101, "Critical", "#d03b3b")]
# Internal level keys stay English (database constraint); this is what users see.
LEVEL_HY = {"Low": "Ցածր", "Moderate": "Չափավոր", "High": "Բարձր", "Critical": "Կրիտիկական"}

# Weight of each issue in the overall score (reflects impact on people & ecosystems in Armenia).
WEIGHTS = {
    "water_pollution": 1.0,
    "drought": 0.9,
    "deforestation": 0.9,
    "mining": 0.85,
    "soil_degradation": 0.8,
    "wildfire": 0.7,
}


def level_for(score: float) -> tuple[str, str]:
    for thr, name, color in LEVELS:
        if score < thr:
            return name, color
    return "Critical", "#d03b3b"


@dataclass
class RiskReport:
    overall: float
    level: str
    color: str
    components: dict[str, float]
    context: list[dict] = field(default_factory=list)
    drivers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"overall": round(self.overall, 1), "level": self.level, "color": self.color,
                "components": {k: round(v, 1) for k, v in self.components.items()},
                "context": self.context, "drivers": self.drivers}


def assess(issues: dict, lat: float | None = None, lon: float | None = None) -> RiskReport:
    """issues: mapping issue -> {"severity": 0–100, ...} (from vision or GEE analysis)."""
    comps = {k: float(issues.get(k, {}).get("severity", 0.0)) for k in WEIGHTS}

    context: list[dict] = []
    if lat is not None and lon is not None:
        for h, d in armenia.nearest_hotspots(lat, lon, k=3, max_km=40):
            # Known hotspots raise the floor of matching components (decays with distance).
            boost = h.severity * 12 * max(0.0, 1 - d / 40)
            for cat in [h.category, *h.secondary]:
                if cat in comps:
                    comps[cat] = min(100.0, max(comps[cat], comps[cat] * 0.6 + boost))
            context.append({"hotspot": h.name, "distance_km": round(d, 1), "category": h.category,
                            "severity": h.severity})

    # Overall: weighted top-heavy mean — the worst problems dominate, others add up.
    ranked = sorted(((v * WEIGHTS[k], k) for k, v in comps.items()), reverse=True)
    weights = [1.0, 0.5, 0.25, 0.15, 0.1, 0.05]
    overall = sum(v * w for (v, _), w in zip(ranked, weights)) / sum(weights[:3])
    overall = min(100.0, overall)
    name, color = level_for(overall)
    from .vision import ISSUES

    drivers = [f"{ISSUES[k]['label']} ({v:.0f}/100)" for v, k in ranked if v >= 25][:4]
    return RiskReport(overall, name, color, comps, context, drivers)


def hotspot_risk_matrix() -> list[dict]:
    """Likelihood × impact for the hotspot register (used by the risk page)."""
    trend_like = {"worsening": 5, "uncertain": 3, "stable": 3, "improving": 2}
    rows = []
    for h in armenia.HOTSPOTS:
        likelihood = round((trend_like[h.trend] + h.severity) / 2, 1)
        impact = h.severity
        rows.append({"id": h.id, "name": h.name, "category": armenia.CATEGORIES[h.category]["label"],
                     "marz": h.marz, "likelihood": likelihood, "impact": impact,
                     "score": round(likelihood * impact * 4, 1), "trend": h.trend_hy})
    return sorted(rows, key=lambda r: r["score"], reverse=True)
