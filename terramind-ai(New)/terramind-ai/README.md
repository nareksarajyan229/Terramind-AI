# 🌍 TerraMind AI

> The app interface is fully in **Armenian** (Հայերեն). Code comments and this README are in English.

**AI-powered environmental monitoring for Armenia.** Upload satellite or drone imagery, or pick any point on the map.
TerraMind detects **water pollution, drought, deforestation, soil degradation, mining impacts and burn scars**. It then
scores the environmental risk and builds a phased **restoration plan** that uses native Armenian species.

Built entirely on free and open-source tools: **Python · PyTorch · YOLO · OpenCV · Streamlit · Google Earth Engine · Supabase**.

---

## Features

| Page | What it does |
|---|---|
| **Overview** | National indicators (forest cover, desertification, Ararat aquifer, Lake Sevan), with sources. Hotspot summary. |
| **🗺️ Interactive map** | 15 documented hotspots (Sevan blooms, Ararat aquifer, Debed/Alaverdi, Akhtala tailings, Teghut, Kajaran, Amulsar, Tavush forests, Khosrov fire, …) on satellite, terrain or street basemaps. Click any point to analyse it. Adds Earth Engine layers (Sentinel-2, NDVI, NDWI, NDCI algae, bare soil, Hansen forest loss) and a MODIS NDVI trend. |
| **🛰️ Image analysis** | Upload JPG/PNG or Sentinel-2/Landsat GeoTIFF. Shows an 8-class land-cover map, outlined problem areas with confidence, a severity heat map and hectare estimates. Exports PNG and JSON, and saves to the database. |
| **📊 Dashboard** | Statistics for every saved analysis: primary issues, risk levels, average risk by province, trend over time, CSV export. |
| **⚠️ Risk assessment** | Risk gauge, component radar and drivers. Compares image-only risk with risk in location context. Includes a likelihood × impact matrix for the national hotspots. |
| **🌱 Restoration** | Actions grouped into 0–6 months, 6–24 months and 2–10 years, each with cost tier, KPI and Armenian partners. Native species are listed by eco-zone. Exports Markdown. Also has a "what-if" scenario builder. |

## How the AI works

```
image / map point
      │
      ├─► Google Earth Engine ── Sentinel-2 median composite, NDVI/NDWI/MNDWI/NDCI/BSI/NBR,
      │                          Hansen forest loss, CHIRPS rainfall anomaly, MODIS NDVI trend
      ▼
 Land-cover classification (8 classes)
   1. PyTorch U-Net            (when trained weights are present)
   2. OpenCV spectral engine   (always available: multispectral indices or visible-band proxies
                                + texture, connected components, morphology)
      ▼
 Feature detection ── YOLO (forest clearings, logging roads, algal blooms, turbid plumes,
      │               tailings ponds, open pits, waste dumps, erosion gullies, burn scars, dry riverbeds)
      ▼
 Issue extraction ── clearings = bare patches ringed by canopy; pollution = turbid/bloom share of water;
      │              drought = stressed vegetation + exposed soil; erosion = bare soil with gully edges …
      ▼
 Risk engine ── weighted top-heavy score + proximity to documented Armenian hotspots
      ▼
 Restoration engine ── rule base → actions, KPIs, partners, native species by eco-zone
```

The app **always works offline**. If PyTorch, YOLO weights, Earth Engine or Supabase are missing, it falls back
gracefully. The sidebar shows which engines are active.

## Quick start

**No terminal:** double-click `START_TerraMind_Windows.bat` (Windows) or `START_TerraMind_Mac.command` (Mac).
The first start installs everything. After that it opens straight in your browser.

**Public link, nothing installed:** push the folder to GitHub and deploy it free on
[Streamlit Community Cloud](https://share.streamlit.io) with main file `app.py`. `requirements.txt` is kept light
for this. The heavy AI extras are in `requirements-ml.txt`.

**Terminal:**

```bash
git clone <your-repo> terramind-ai && cd terramind-ai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # add -r requirements-ml.txt for PyTorch / YOLO / GeoTIFF
cp .env.example .env                                    # optional: add Supabase / Earth Engine keys
streamlit run app.py
```

Open http://localhost:8501. Try **Image analysis → Demo sample**, or **Dashboard → Load demo data**.

Docker: `docker build -t terramind . && docker run -p 8501:8501 --env-file .env terramind`

Tests (engine only, no network or GPU needed): `python -m pytest -q`

## Connecting Google Earth Engine (free)

1. Sign up at https://earthengine.google.com/ and register a Google Cloud project for Earth Engine
   (noncommercial / research use is free).
2. `earthengine authenticate` on your machine. For servers, create a service account and set
   `GEE_SERVICE_ACCOUNT` and `GEE_KEY_FILE`.
3. Set `GEE_PROJECT=<your-project-id>` in `.env` and restart.

The map then runs **live analysis**. It uses Sentinel-2 indices for the latest growing season against a baseline
four years earlier, Hansen forest loss since 2000, a 12-month CHIRPS rainfall anomaly, and AI classification of the
Sentinel-2 true-colour image. `scripts/export_gee_tiles.py` exports multiband tiles for every hotspot so you can label
training data.

## Connecting Supabase (free tier)

1. Create a project at https://supabase.com.
2. Open the **SQL editor**, paste `supabase/schema.sql` and run it. This creates the `analyses` and
   `field_reports` tables, RLS policies, the `analysis-images` storage bucket and a summary view.
3. Put the project URL and anon key in `.env` as `SUPABASE_URL` and `SUPABASE_KEY`.

Without Supabase, analyses are stored in `data/terramind.db` (SQLite) through the same API.
For production, remove the demo `insert` policies and use Supabase Auth.

## Training your own models

The included spectral engine is explainable and needs no training data. For production-grade accuracy, train
the models on labelled Armenian imagery:

**YOLO detector**
```bash
# label 640×640 tiles in CVAT or Label Studio (YOLO format), classes listed in training/dataset.yaml
python training/train_yolo.py --data training/dataset.yaml --model yolov8s.pt --epochs 100
# → models/terramind_yolo.pt (picked up automatically)
```

**U-Net segmentation**
```bash
# datasets/seg/{train,val}/{images,masks}/*.png, mask values 0–7 (see script docstring)
python training/train_segmentation.py --data datasets/seg --epochs 60
# → models/terramind_unet.pt
```
Tip: bootstrap masks with `vision.classify_multispectral()` on Sentinel-2 tiles, then correct them by hand.
That is much faster than labelling from scratch. Open datasets such as LoveDA and DeepGlobe can be re-mapped to these classes.

## Project structure

```

app.py                     Navigation (Armenian page names)
views/                     Overview · Map · Image analysis · Dashboard · Risk · Restoration
terramind/
  vision.py                OpenCV pipeline, classes, issue extraction, overlays
  indices.py               NDVI, NDWI, MNDWI, NDCI, BSI, NBR, ExG, VARI, texture
  segmentation.py          PyTorch U-Net + tiled inference
  detector.py              Ultralytics YOLO wrapper → issue regions
  gee.py                   Earth Engine: composites, indices, forest loss, rainfall, tiles, thumbnails
  risk.py                  Risk scoring and hotspot risk matrix
  recommendations.py       Restoration rule base and native species by eco-zone
  armenia.py               Hotspot register and national indicators (with sources)
  services.py              End-to-end workflows used by the UI
  db.py                    Supabase / SQLite storage
  ui.py                    Theme, components, Plotly charts
supabase/schema.sql        Database schema, RLS and storage bucket
training/                  YOLO and U-Net training scripts, dataset.yaml
scripts/                   Demo-sample generator, Earth Engine tile export
tests/                     Engine tests (pytest)
```

## Armenia focus: key sources

- Lake Sevan cyanobacteria blooms and water level: Yerevan State University (2019, 2020), CivilNet
- Ararat Valley aquifer depletion and fish farms: EBRD GEFF, EVN Report, USGS
- Forest cover, desertification and degradation costs: World Bank (2021, 2025), UN Armenia / FAO
- Debed River, Alaverdi and Akhtala heavy metals: Arnika, Armenpress, Business & Human Rights Resource Centre

Links are in `terramind/armenia.py` and appear in the app. Hotspot severity ratings (1–5) are **indicative**,
for prioritisation only. Update them as new monitoring data comes in.

## Limitations

- **Without trained weights, detection is rule-based.** It uses spectral indices, texture and shape rules. That
  works well for clear cases but can confuse dry grassland with stress, or bright rock with tailings. Multispectral
  input (Sentinel-2 GeoTIFF) is much more reliable than RGB.
- The demo images in `assets/samples` are **synthetic illustrations**, not real imagery.
- Risk scores support prioritisation. Confirm AI detections with field inspection or lab sampling before any
  enforcement or investment decision.

## Licences

The project code is free to use. Some dependencies have their own licences: Ultralytics YOLO is **AGPL-3.0**;
Earth Engine is free for noncommercial use; Sentinel-2 data © ESA Copernicus; Hansen GFC © University of Maryland.
