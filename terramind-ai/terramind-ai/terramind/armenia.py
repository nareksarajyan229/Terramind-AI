"""Reference data for environmental hotspots and national indicators in Armenia.

Severity scores (1–5) are indicative editorial ratings used to prioritise the map
and risk matrix; they are not official government classifications. Every hotspot
lists sources so users can verify and update the information.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

ARMENIA_CENTER = (40.15, 44.95)
ARMENIA_BBOX = (38.84, 43.45, 41.30, 46.63)  # (min_lat, min_lon, max_lat, max_lon)

CATEGORIES = {
    "water_pollution": {"label": "Water pollution", "color": "#2a78d6", "icon": "tint"},
    "drought": {"label": "Drought & groundwater", "color": "#eda100", "icon": "sun-o"},
    "deforestation": {"label": "Deforestation", "color": "#008300", "icon": "tree"},
    "soil_degradation": {"label": "Soil degradation", "color": "#eb6834", "icon": "leaf"},
    "mining": {"label": "Mining & heavy metals", "color": "#6250d6", "icon": "industry"},
    "wildfire": {"label": "Wildfire", "color": "#e34948", "icon": "fire"},
}

MARZER = [
    "Yerevan", "Aragatsotn", "Ararat", "Armavir", "Gegharkunik", "Kotayk",
    "Lori", "Shirak", "Syunik", "Tavush", "Vayots Dzor",
]


@dataclass
class Hotspot:
    id: str
    name: str
    lat: float
    lon: float
    marz: str
    category: str
    severity: int  # 1–5 indicative
    trend: str  # "worsening" | "stable" | "improving" | "uncertain"
    summary: str
    drivers: list[str]
    indicators: list[str]
    secondary: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


HOTSPOTS: list[Hotspot] = [
    Hotspot(
        "sevan", "Lake Sevan — cyanobacteria blooms", 40.38, 45.25, "Gegharkunik",
        "water_pollution", 5, "uncertain",
        "Armenia's largest freshwater body. Soviet-era drawdown lowered the lake from "
        "about 1916 m to 1897 m; recurring summer blooms of Dolichospermum and other "
        "cyanobacteria start in the littoral zone and spread offshore.",
        ["Water-level fluctuation", "Nutrient (P, N) inflow from settlements and agriculture",
         "Warm, calm summer water", "Organic matter from flooded shoreline vegetation"],
        ["NDCI / chlorophyll-a (Sentinel-2 B5/B4)", "Surface water temperature", "Lake level", "Secchi depth"],
        ["soil_degradation"],
        ["https://journals.ysu.am/index.php/proceedings-chem-biol/article/view/vol54_no2_2020_pp168-176",
         "https://journals.ysu.am/index.php/proceedings-chem-biol/article/download/vol53_no3_2019_pp206-211/pdf/8338",
         "https://www.civilnet.am/en/news/382051/armenias-lake-sevan-faces-serious-threat-from-algae-bloom/"],
    ),
    Hotspot(
        "ararat_aquifer", "Ararat Valley — artesian aquifer depletion", 40.05, 44.20, "Armavir",
        "drought", 5, "worsening",
        "Semi-arid plain holding Armenia's second-largest freshwater reserve. Groundwater in "
        "the artesian aquifer has fallen by roughly two-thirds since the early 1980s, driven "
        "largely by 200+ fish farms; dozens of communities lost free-flowing wells.",
        ["Over-abstraction for aquaculture", "Unmetered and illegal wells", "Rainfall under ~300 mm/yr",
         "Rising temperatures"],
        ["Well flow / piezometric level", "GRACE groundwater anomaly", "Crop NDVI in irrigated fields",
         "Soil salinity (SWIR indices)"],
        ["soil_degradation"],
        ["https://ebrdgeff.com/armenia/?p=4936", "https://evnreport.com/?p=14655",
         "https://www.usgs.gov/centers/dakota-water/science/armenia-building-groundwater-management-capacity-ararat-valley"],
    ),
    Hotspot(
        "ararat_salinity", "Ararat plain — salinization & desertification", 39.93, 44.55, "Ararat",
        "soil_degradation", 4, "worsening",
        "Low-lying irrigated lands where falling groundwater, evaporation and poor drainage "
        "drive salt accumulation and loss of productivity.",
        ["Inefficient flood irrigation", "Poor drainage", "High evaporation"],
        ["Salinity index (SWIR)", "Bare soil index", "Crop yield"],
        ["drought"],
        ["https://evnreport.com/?p=43536"],
    ),
    Hotspot(
        "debed_alaverdi", "Debed River — Alaverdi copper smelter", 41.10, 44.66, "Lori",
        "mining", 5, "uncertain",
        "Sediment samples below industrial sites in Lori show copper far above background "
        "levels; the Debed irrigates riverside farmland and flows into Georgia.",
        ["Smelter emissions and effluent", "Tailings seepage", "Pipeline failures"],
        ["Turbidity / iron-oxide colour", "Heavy-metal sampling", "Riparian vegetation stress"],
        ["water_pollution"],
        ["https://arnika.org/en/news/heavy-metals-polluting-armenia",
         "https://armenpress.am/en/article/941486"],
    ),
    Hotspot(
        "akhtala_nahatak", "Akhtala — Nahatak tailings dump", 41.15, 44.77, "Lori",
        "mining", 5, "worsening",
        "Tailings facility of the Akhtala Mining and Processing Plant. A pipeline rupture "
        "polluted the Debed in August 2025; reclamation was ordered without a plan for "
        "existing river contamination.",
        ["Tailings pipeline failures", "Acid mine drainage", "Dust from dry tailings"],
        ["Tailings pond extent", "Water colour anomalies downstream", "Vegetation die-back"],
        ["water_pollution"],
        ["https://www.business-humanrights.org/en/latest-news/armenia-debed-river-polluted-after-accident-in-nahatak-tailing-dump-pipeline/",
         "https://iwpr.net/752ov4gc"],
    ),
    Hotspot(
        "teghut", "Teghut — open-pit mine in forest", 41.11, 44.85, "Lori",
        "deforestation", 4, "stable",
        "Copper-molybdenum open pit developed inside broadleaf forest; forest clearing, "
        "waste-rock dumps and a tailings facility alter the Shnogh river catchment.",
        ["Forest clearing for pit and dumps", "Tailings storage", "Road construction"],
        ["Forest loss (Hansen GFC)", "NDVI change", "Pit and dump footprint"],
        ["mining", "water_pollution"],
        ["https://arnika.org/en/news/heavy-metals-polluting-armenia"],
    ),
    Hotspot(
        "kajaran", "Kajaran — Zangezur copper-molybdenum combine", 39.15, 46.15, "Syunik",
        "mining", 4, "stable",
        "Armenia's largest mining operation, with large open pit and tailings dams in the "
        "Voghji river basin.",
        ["Open-pit expansion", "Tailings dams (Artsvanik, Geghanush)", "Dust"],
        ["Pit/tailings extent", "River turbidity", "Vegetation stress near dumps"],
        ["water_pollution", "soil_degradation"],
        [],
    ),
    Hotspot(
        "amulsar", "Amulsar — proposed gold mine near Jermuk", 39.73, 45.72, "Vayots Dzor",
        "mining", 3, "uncertain",
        "Contested gold project in the Arpa and Vorotan headwaters; concerns centre on acid "
        "rock drainage and links to the Arpa–Sevan water transfer.",
        ["Potential acid rock drainage", "Heap-leach facility", "Proximity to Jermuk springs"],
        ["Land disturbance footprint", "Surface water quality", "Alpine meadow NDVI"],
        ["water_pollution"],
        [],
    ),
    Hotspot(
        "tavush_forests", "Tavush & Lori forests — fuelwood and illegal logging", 40.88, 45.15, "Tavush",
        "deforestation", 4, "uncertain",
        "Most of Armenia's remaining forest is in the north-east. Fuelwood demand and illegal "
        "logging degrade beech-oak-hornbeam stands; national forest cover is only about 9–11%.",
        ["Fuelwood collection", "Illegal logging", "Pests and disease", "Grazing in forest"],
        ["Canopy cover / NDVI", "Hansen forest loss", "Gap detection"],
        ["soil_degradation"],
        ["https://www.worldbank.org/en/news/feature/2021/06/07/preserving-forests-is-key-to-armenia-s-climate-goals-and-economic-sustainability",
         "https://armenia.un.org/en/node/315011"],
    ),
    Hotspot(
        "dilijan", "Dilijan National Park — forest health", 40.74, 44.86, "Tavush",
        "deforestation", 3, "stable",
        "Protected mountain forests under pressure from pests, edge logging and tourism growth.",
        ["Forest pests", "Edge logging", "Infrastructure"],
        ["NDVI anomaly", "Red-edge chlorophyll index"],
        [],
        [],
    ),
    Hotspot(
        "khosrov", "Khosrov Forest Reserve — wildfire scars", 39.95, 44.92, "Ararat",
        "wildfire", 4, "improving",
        "One of the oldest reserves in the region; the large 2017 fire burned juniper and oak "
        "woodland on dry slopes where natural regrowth is slow.",
        ["Hot, dry summers", "Steep arid slopes", "Human ignition"],
        ["Burn index (NBR / dNBR)", "Post-fire NDVI recovery"],
        ["soil_degradation"],
        [],
    ),
    Hotspot(
        "hrazdan", "Hrazdan River — Yerevan urban wastewater", 40.17, 44.49, "Yerevan",
        "water_pollution", 3, "uncertain",
        "River through the capital receiving insufficiently treated municipal and industrial "
        "wastewater, downstream of Sevan's outflow.",
        ["Wastewater discharge", "Urban runoff", "Reduced flow"],
        ["Turbidity", "Riparian vegetation", "Dissolved oxygen sampling"],
        [],
        [],
    ),
    Hotspot(
        "vayots_dzor_erosion", "Vayots Dzor — slope erosion & overgrazing", 39.75, 45.33, "Vayots Dzor",
        "soil_degradation", 4, "worsening",
        "Steep, dry slopes with thin soils; overgrazing near villages and gully erosion strip "
        "topsoil.",
        ["Overgrazing near settlements", "Steep slopes", "Loss of tree cover"],
        ["Bare soil index", "Gully detection", "Pasture NDVI"],
        ["drought"],
        [],
    ),
    Hotspot(
        "aragats_pastures", "Aragats & Aparan — degraded pastures", 40.55, 44.30, "Aragatsotn",
        "soil_degradation", 3, "stable",
        "Near-village pastures are overgrazed while remote summer pastures are under-used, "
        "concentrating erosion.",
        ["Uneven grazing pressure", "Tracks and trampling"],
        ["Pasture NDVI", "Bare patch fraction"],
        [],
        [],
    ),
    Hotspot(
        "armash", "Armash wetlands & fishponds", 39.77, 44.80, "Ararat",
        "drought", 3, "uncertain",
        "Important bird area dependent on fishpond water; vulnerable to groundwater decline.",
        ["Groundwater decline", "Pond drainage"],
        ["Surface water extent (JRC / MNDWI)", "Wetland NDVI"],
        ["water_pollution"],
        [],
    ),
]

# National indicators shown on the dashboard. Each carries its source.
NATIONAL_INDICATORS = [
    {"label": "Forest cover", "value": "9.3–11%", "note": "Varies by source (World Bank 2025: 9.3%; FAO/UN: ~11%)",
     "source": "https://www.worldbank.org/en/news/feature/2025/03/11/healing-armenia-s-land-a-strategic-approach-to-landscape-restoration"},
    {"label": "Land with signs of desertification", "value": "~80%", "note": "UN Armenia / FAO",
     "source": "https://armenia.un.org/en/node/315011"},
    {"label": "Ararat aquifer decline", "value": "≈ ⅔", "note": "Since the early 1980s",
     "source": "https://ebrdgeff.com/armenia/?p=4936"},
    {"label": "Lake Sevan drawdown", "value": "≈ 19 m", "note": "1916.2 → 1897.0 m during Soviet era",
     "source": "https://journals.ysu.am/index.php/proceedings-chem-biol/article/download/vol53_no3_2019_pp206-211/pdf/8338"},
    {"label": "Annual cost of forest & wetland degradation", "value": "> US$80M", "note": "≈ 0.4% of GDP",
     "source": "https://www.worldbank.org/en/news/feature/2025/03/11/healing-armenia-s-land-a-strategic-approach-to-landscape-restoration"},
    {"label": "Forest cover target (NDC)", "value": "12.9%", "note": "National climate commitment",
     "source": "https://www.worldbank.org/en/news/feature/2021/06/07/preserving-forests-is-key-to-armenia-s-climate-goals-and-economic-sustainability"},
]


def hotspots_as_records() -> list[dict]:
    return [h.to_dict() for h in HOTSPOTS]


def get_hotspot(hid: str) -> Hotspot | None:
    return next((h for h in HOTSPOTS if h.id == hid), None)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_hotspots(lat: float, lon: float, k: int = 3, max_km: float = 60.0) -> list[tuple[Hotspot, float]]:
    ranked = sorted(((h, haversine_km(lat, lon, h.lat, h.lon)) for h in HOTSPOTS), key=lambda t: t[1])
    return [(h, d) for h, d in ranked[:k] if d <= max_km]


def in_armenia(lat: float, lon: float) -> bool:
    a, b, c, d = ARMENIA_BBOX
    return a <= lat <= c and b <= lon <= d
