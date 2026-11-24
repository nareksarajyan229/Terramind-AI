"""Restoration recommendation engine tailored to Armenian ecosystems.

Rules map each detected issue (and its severity) to concrete actions, a timeframe,
relative cost, a monitoring KPI and typical Armenian partners. Species lists are
native or long-established in Armenia and grouped by eco-zone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

SPECIES_BY_ZONE = {
    "humid_forest": {
        "label": "North-east humid forest belt (Lori, Tavush, Dilijan)",
        "species": [
            ("Oriental beech", "Fagus orientalis", "Shade-tolerant; under-plant gaps at 1,000–2,000 m"),
            ("Georgian oak", "Quercus iberica", "Drought-tolerant backbone species for lower slopes"),
            ("Caucasian hornbeam", "Carpinus caucasica", "Fast canopy closure in mixed stands"),
            ("Field maple", "Acer campestre", "Diversifies stands, good for edges"),
            ("Wild pear / wild apple", "Pyrus caucasica / Malus orientalis", "Food for wildlife and communities"),
        ],
    },
    "arid_south": {
        "label": "Arid & semi-arid south (Ararat, Vayots Dzor, lower Syunik)",
        "species": [
            ("Multi-seed juniper", "Juniperus polycarpos", "Very drought-hardy; for steep dry slopes and post-fire sites"),
            ("Wild pistachio", "Pistacia mutica", "Native dryland tree for rocky ground"),
            ("Willow-leaved pear", "Pyrus salicifolia", "Tolerates heat and poor soil"),
            ("Sumac", "Rhus coriaria", "Shrub cover against sheet erosion"),
            ("Sea buckthorn", "Hippophae rhamnoides", "Nitrogen-fixing, binds gullies and riverbanks"),
        ],
    },
    "highland": {
        "label": "Highland plateaus & lake basins (Gegharkunik, Shirak, Aragatsotn, Kotayk)",
        "species": [
            ("Caucasian pine", "Pinus sylvestris var. hamata", "Shelterbelts and slope stabilisation up to ~2,300 m"),
            ("High-mountain oak", "Quercus macranthera", "Cold-hardy native oak"),
            ("White willow", "Salix alba", "Riparian buffers on Sevan tributaries"),
            ("Sea buckthorn", "Hippophae rhamnoides", "Erosion control, berries for income"),
            ("Mountain ash", "Sorbus aucuparia", "Pioneer for forest edges"),
        ],
    },
}

ZONE_BY_MARZ = {
    "Lori": "humid_forest", "Tavush": "humid_forest",
    "Ararat": "arid_south", "Armavir": "arid_south", "Vayots Dzor": "arid_south", "Yerevan": "arid_south",
    "Syunik": "arid_south",
    "Gegharkunik": "highland", "Shirak": "highland", "Aragatsotn": "highland", "Kotayk": "highland",
}


def zone_for(lat: float | None, lon: float | None, marz: str | None = None) -> str:
    if marz in ZONE_BY_MARZ:
        return ZONE_BY_MARZ[marz]
    if lat is None or lon is None:
        return "highland"
    if lat > 40.65 and lon > 44.3:
        return "humid_forest"
    if lat < 40.25:
        return "arid_south"
    return "highland"


@dataclass
class Action:
    issue: str
    title: str
    detail: str
    priority: str      # "Immediate" | "High" | "Medium"
    phase: str         # "0–6 months" | "6–24 months" | "2–10 years"
    cost: str          # "$" | "$$" | "$$$"
    kpi: str
    partners: str

    def to_dict(self) -> dict:
        return asdict(self)


RULES: dict[str, list[tuple[int, Action]]] = {
    "deforestation": [
        (0, Action("deforestation", "Map and patrol clearings",
                   "Export detected clearing polygons to forest rangers; verify on the ground and with "
                   "Hansen/GLAD alerts; prioritise fresh cuts near roads.",
                   "Immediate", "0–6 months", "$", "Share of alerts verified within 14 days",
                   "Hayantar (state forest agency), Environmental Protection and Mining Inspection Body")),
        (25, Action("deforestation", "Assisted natural regeneration",
                    "Fence gaps against livestock, protect seedlings and mother trees; cheaper and more "
                    "resilient than planting where seed sources remain.",
                    "High", "6–24 months", "$$", "Seedling density per hectare; NDVI recovery",
                    "Local communities, WWF-Armenia")),
        (45, Action("deforestation", "Native-species reforestation",
                    "Plant mixed native stands (see species list) at 1,500–2,500 seedlings/ha on the "
                    "largest clearings; use local seed provenance and community nurseries.",
                    "High", "2–10 years", "$$$", "Canopy cover %, 3-year survival rate ≥ 70%",
                    "Armenia Tree Project, My Forest Armenia, Hayantar")),
        (60, Action("deforestation", "Alternatives to fuelwood",
                    "Support efficient stoves, gas connection or solar heating in villages that cut "
                    "nearby forest for fuel.",
                    "Medium", "6–24 months", "$$", "Fuelwood m³ per household per year",
                    "Municipalities, energy-efficiency programmes")),
    ],
    "water_pollution": [
        (0, Action("water_pollution", "Confirm with water sampling",
                   "Sample at the detected plume or bloom: chlorophyll-a, cyanotoxins (microcystin), "
                   "nutrients, turbidity and heavy metals (Cu, Mo, Zn, Pb) near mining sites.",
                   "Immediate", "0–6 months", "$", "Samples analysed vs. national water-quality norms",
                   "Hydrometeorology & Monitoring Center, university labs")),
        (25, Action("water_pollution", "Riparian buffer strips",
                    "Restore 15–30 m vegetated buffers of willow, poplar and grasses along tributaries "
                    "to trap nutrients and sediment before they reach the lake or river.",
                    "High", "6–24 months", "$$", "Buffer length restored (km); nitrate/phosphate trend",
                    "Communities, Sevan National Park, Ministry of Environment")),
        (40, Action("water_pollution", "Stop untreated discharge",
                    "Prioritise wastewater treatment for settlements and resorts in the basin; inspect "
                    "industrial outfalls and tailings pipelines flagged by the analysis.",
                    "High", "6–24 months", "$$$", "% of discharge treated; incident count",
                    "Water supply utilities, Inspection Body")),
        (55, Action("water_pollution", "Bloom early-warning",
                    "Monitor Sentinel-2 NDCI weekly from June to October and alert beaches and fisheries "
                    "when chlorophyll rises.",
                    "Medium", "0–6 months", "$", "Days of warning before bloom peak",
                    "Sevan National Park, TerraMind AI")),
        (70, Action("water_pollution", "Tailings and mine-water remediation",
                    "Seal or line leaking tailings, treat acid mine drainage (lime dosing, constructed "
                    "wetlands) and revegetate dry tailings to stop dust.",
                    "High", "2–10 years", "$$$", "Downstream metal concentrations",
                    "Mining operators, Inspection Body")),
    ],
    "drought": [
        (0, Action("drought", "Water-use audit",
                   "Meter all wells and irrigation intakes; close illegal and abandoned wells in the "
                   "Ararat artesian basin.",
                   "Immediate", "0–6 months", "$", "% of wells metered; aquifer level trend",
                   "Water Committee, Ministry of Environment, USAID-supported programmes")),
        (25, Action("drought", "Efficient irrigation",
                    "Shift from flood to drip or sprinkler irrigation for orchards and vineyards; "
                    "schedule irrigation with soil-moisture data.",
                    "High", "6–24 months", "$$", "m³ water per tonne of crop",
                    "Farmers' cooperatives, agricultural extension")),
        (40, Action("drought", "Recirculating aquaculture",
                    "Require fish farms to adopt recirculating or water-reuse systems and to reuse "
                    "outflow for irrigation.",
                    "High", "2–10 years", "$$$", "Groundwater abstraction by fish farms (m³/yr)",
                    "Fish farm operators, regulators")),
        (55, Action("drought", "Shelterbelts and drought-tolerant crops",
                    "Plant windbreaks of drought-hardy natives and diversify into drought-tolerant crops.",
                    "Medium", "2–10 years", "$$", "Evapotranspiration and yield stability",
                    "Communities, Armenia Tree Project")),
    ],
    "soil_degradation": [
        (0, Action("soil_degradation", "Grazing management plan",
                   "Rotate herds to under-used remote summer pastures; rest overgrazed areas near villages.",
                   "High", "0–6 months", "$", "Pasture NDVI, bare-patch fraction",
                   "Community pasture committees")),
        (25, Action("soil_degradation", "Erosion control structures",
                    "Build check dams, gabions and contour trenches in active gullies; mulch and seed "
                    "bare slopes with native grasses.",
                    "High", "6–24 months", "$$", "Gully head retreat (m/yr)",
                    "Municipalities, FAO projects")),
        (40, Action("soil_degradation", "Slope revegetation",
                    "Terrace and plant nitrogen-fixing shrubs (sea buckthorn) and dryland trees on "
                    "slopes above 15°.",
                    "Medium", "2–10 years", "$$", "Vegetation cover on treated slopes",
                    "Armenia Tree Project, Hayantar")),
        (55, Action("soil_degradation", "Salinity remediation",
                    "In the Ararat plain: restore drainage, leach salts with controlled irrigation and "
                    "add gypsum or organic matter.",
                    "Medium", "2–10 years", "$$$", "Soil electrical conductivity (EC)",
                    "Farmers, agricultural institutes")),
    ],
    "mining": [
        (0, Action("mining", "Footprint and compliance check",
                   "Compare the disturbed area with permitted boundaries and environmental impact "
                   "assessment commitments; report discrepancies.",
                   "Immediate", "0–6 months", "$", "Disturbed area vs. permit (ha)",
                   "Inspection Body, civil-society monitors")),
        (35, Action("mining", "Progressive rehabilitation",
                    "Re-shape waste dumps, cap with topsoil and revegetate closed areas while the mine "
                    "operates; plant metal-tolerant species on tailings.",
                    "High", "2–10 years", "$$$", "% of disturbed area rehabilitated",
                    "Mining operators")),
        (55, Action("mining", "Community health monitoring",
                    "Monitor dust, soil and crop metal levels in nearby villages; restrict irrigation "
                    "with contaminated river water.",
                    "High", "6–24 months", "$$", "Metal levels in soil and crops",
                    "Ministry of Health, NGOs")),
    ],
    "wildfire": [
        (0, Action("wildfire", "Post-fire assessment",
                   "Map burn severity with dNBR, protect soils from erosion before the next rains "
                   "(log barriers, straw mulch).",
                   "Immediate", "0–6 months", "$", "Burned area by severity class",
                   "Reserve administration, Ministry of Emergency Situations")),
        (30, Action("wildfire", "Fire prevention",
                    "Firebreaks, ranger patrols in dry season, early detection with satellite hotspots "
                    "(FIRMS).",
                    "High", "0–6 months", "$$", "Detection-to-response time",
                    "Reserve administration, communities")),
        (50, Action("wildfire", "Restore burned woodland",
                    "Seed and plant juniper and oak in burned patches where natural regrowth is "
                    "absent after 2 seasons.",
                    "Medium", "2–10 years", "$$", "Post-fire NDVI recovery",
                    "Reserve administration, WWF-Armenia")),
    ],
}


def recommend(issues: dict, lat: float | None = None, lon: float | None = None,
              marz: str | None = None, min_severity: float = 15.0) -> dict:
    """Return {"actions": [...], "zone": {...}, "summary": str}."""
    actions: list[Action] = []
    for key, info in sorted(issues.items(), key=lambda kv: -kv[1].get("severity", 0)):
        sev = float(info.get("severity", 0))
        if sev < min_severity:
            continue
        actions.extend(a for thr, a in RULES.get(key, []) if sev >= thr)

    order = {"Immediate": 0, "High": 1, "Medium": 2}
    actions.sort(key=lambda a: order[a.priority])
    zone_key = zone_for(lat, lon, marz)
    zone = SPECIES_BY_ZONE[zone_key]
    top = [k for k, v in issues.items() if v.get("severity", 0) >= min_severity]
    summary = ("No significant problems detected — continue routine monitoring every season."
               if not top else
               f"{len(actions)} actions for {', '.join(t.replace('_', ' ') for t in top)}. "
               f"Start with the 'Immediate' items, then the restoration work.")
    return {"actions": [a.to_dict() for a in actions], "zone": {"key": zone_key, **zone}, "summary": summary}


def to_markdown(plan: dict, title: str = "TerraMind AI — Restoration plan") -> str:
    lines = [f"# {title}", "", plan["summary"], ""]
    for phase in ["0–6 months", "6–24 months", "2–10 years"]:
        items = [a for a in plan["actions"] if a["phase"] == phase]
        if not items:
            continue
        lines += [f"## {phase}", ""]
        for a in items:
            lines += [f"### {a['title']}  ({a['priority']}, cost {a['cost']})",
                      a["detail"], f"- **KPI:** {a['kpi']}", f"- **Partners:** {a['partners']}", ""]
    z = plan["zone"]
    lines += [f"## Recommended native species — {z['label']}", "", "| Common name | Latin | Use |", "|---|---|---|"]
    lines += [f"| {c} | *{l}* | {u} |" for c, l, u in z["species"]]
    return "\n".join(lines)
