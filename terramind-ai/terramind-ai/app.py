"""TerraMind AI — home / overview.  Run:  streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from terramind import armenia, ui

ui.setup("Overview", "🌍")
ui.hero("TerraMind AI",
        "AI-powered environmental monitoring for Armenia — detect water pollution, drought, deforestation "
        "and soil degradation from satellite imagery, assess risk, and plan restoration.")

st.markdown("#### Armenia at a glance")
cols = st.columns(len(armenia.NATIONAL_INDICATORS[:4]))
for c, ind in zip(cols, armenia.NATIONAL_INDICATORS[:4]):
    with c:
        ui.kpi(ind["label"], ind["value"], f'{ind["note"]} · <a href="{ind["source"]}" target="_blank">source</a>')

st.write("")
left, right = st.columns([1.15, 1])
with left:
    st.markdown("#### What you can do")
    a, b = st.columns(2)
    with a:
        st.page_link("pages/1_Interactive_Map.py", label="**Interactive map**", icon="🗺️")
        st.caption("Explore documented hotspots, Sentinel-2 layers and analyse any point in Armenia.")
        st.page_link("pages/3_Dashboard.py", label="**Analysis dashboard**", icon="📊")
        st.caption("Statistics across all saved analyses, marzer and issue types.")
        st.page_link("pages/5_Restoration.py", label="**Restoration planner**", icon="🌱")
        st.caption("Phased action plan with native species and monitoring KPIs.")
    with b:
        st.page_link("pages/2_Image_Analysis.py", label="**AI image analysis**", icon="🛰️")
        st.caption("Upload satellite or drone imagery for land-cover classification and issue detection.")
        st.page_link("pages/4_Risk_Assessment.py", label="**Risk assessment**", icon="⚠️")
        st.caption("Risk gauge, component breakdown and the national hotspot risk matrix.")

    st.markdown("#### How the AI works")
    st.markdown(
        "1. **Input** — an uploaded image (JPG/PNG, or Sentinel-2/Landsat GeoTIFF) or a map location pulled from "
        "Google Earth Engine.\n"
        "2. **Land cover** — a PyTorch U-Net segments 8 classes; without trained weights, an explainable OpenCV "
        "engine uses spectral indices (NDVI, NDWI, NDCI, BSI, NBR, or visible-band proxies).\n"
        "3. **Detection** — YOLO finds discrete features (clearings, tailings ponds, gullies, blooms); "
        "connected-component analysis outlines affected areas.\n"
        "4. **Risk** — issue severities are weighted and combined with proximity to documented Armenian hotspots.\n"
        "5. **Restoration** — rules map each problem to actions, partners, KPIs and native species by eco-zone.")

with right:
    st.markdown("#### Documented hotspots by category")
    df = pd.DataFrame(armenia.hotspots_as_records())
    counts = df.groupby("category").agg(n=("id", "count"), sev=("severity", "mean")).reset_index()
    counts["label"] = counts["category"].map(lambda c: armenia.CATEGORIES[c]["label"])
    counts = counts.sort_values("n")
    fig = go.Figure(go.Bar(
        x=counts["n"], y=counts["label"], orientation="h",
        marker=dict(color=[armenia.CATEGORIES[c]["color"] for c in counts["category"]], cornerradius=4,
                    line=dict(color=ui.SURFACE, width=2)),
        customdata=counts["sev"].round(1), text=counts["n"], textposition="outside",
        hovertemplate="%{y}<br>%{x} hotspots · mean severity %{customdata}/5<extra></extra>"))
    fig.update_xaxes(title="Number of hotspots", dtick=1)
    st.plotly_chart(ui._layout(fig, 300), use_container_width=True)

    st.markdown("#### Highest-severity hotspots")
    top = df.sort_values(["severity", "name"], ascending=[False, True]).head(6)
    st.dataframe(
        top[["name", "marz", "severity", "trend"]].rename(columns={"name": "Hotspot", "marz": "Marz",
                                                                    "severity": "Severity (1–5)", "trend": "Trend"}),
        hide_index=True, use_container_width=True)

st.divider()
st.caption("Severity ratings are indicative, for prioritisation only. Indicators link to their sources; "
           "AI detections should be confirmed by field inspection before enforcement decisions.")
