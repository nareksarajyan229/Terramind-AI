"""AI image analysis: upload or sample imagery -> land cover, issues, overlays, report."""
import json

import pandas as pd
import streamlit as st

from terramind import armenia, config, db, services, ui, vision

ui.setup("Image Analysis", "🛰️")
ui.hero("AI image analysis", "Upload satellite or drone imagery. TerraMind classifies land cover, detects environmental "
        "problems and outlines affected areas.")

SAMPLES = {
    "forest_clearing_tavush.png": ("Forest clearings (Tavush-style)", "tavush_forests"),
    "lake_sevan_bloom.png": ("Lake with algal bloom & turbid inflow (Sevan-style)", "sevan"),
    "ararat_drought_farmland.png": ("Dry farmland with gullies (Ararat-style)", "ararat_aquifer"),
    "mining_tailings_lori.png": ("Open pit & tailings pond (Lori-style)", "akhtala_nahatak"),
}

left, right = st.columns([1, 1.6])
with left:
    st.markdown("#### 1 · Input")
    src = st.radio("Source", ["Upload image", "Demo sample"], horizontal=True)
    data = name = None
    default_spot = "—"
    if src == "Upload image":
        f = st.file_uploader("Satellite / aerial image", type=["png", "jpg", "jpeg", "tif", "tiff"],
                             help="RGB images work everywhere. Multiband Sentinel-2 or 4-band (B,G,R,NIR) "
                                  "GeoTIFFs unlock NDVI/NDWI/NDCI analysis (requires rasterio).")
        if f:
            data, name = f.getvalue(), f.name
    else:
        pick = st.selectbox("Sample", list(SAMPLES), format_func=lambda k: SAMPLES[k][0])
        data, name = (config.SAMPLES_DIR / pick).read_bytes(), pick
        default_spot = SAMPLES[pick][1]
        st.caption("Synthetic illustration for demonstration — not real imagery.")

    st.markdown("#### 2 · Location (optional)")
    spots = ["—"] + [h.id for h in armenia.HOTSPOTS] + ["custom"]
    loc = st.selectbox("Where was this image taken?", spots, index=spots.index(default_spot),
                       format_func=lambda i: {"—": "Unknown", "custom": "Custom coordinates"}.get(
                           i, armenia.get_hotspot(i).name if armenia.get_hotspot(i) else i))
    lat = lon = None
    if loc == "custom":
        c1, c2 = st.columns(2)
        lat = c1.number_input("Lat", value=40.18, format="%.5f")
        lon = c2.number_input("Lon", value=44.51, format="%.5f")
    elif loc != "—":
        h = armenia.get_hotspot(loc)
        lat, lon = h.lat, h.lon

    st.markdown("#### 3 · Settings")
    gsd = st.number_input("Ground resolution (m / pixel)", 0.1, 100.0, 10.0, 0.5,
                          help="Sentinel-2 = 10 m, Landsat = 30 m, drone ≈ 0.05–0.5 m. Used for hectare estimates.")
    sens = st.slider("Detection sensitivity", 0.6, 1.5, 1.0, 0.05, help="Higher = flags more potential problems")
    engine = st.selectbox("AI engine", ["auto", "spectral", "segmentation", "yolo"],
                          format_func=lambda e: {"auto": "Auto (best available)", "spectral": "OpenCV spectral rules",
                                                 "segmentation": "PyTorch U-Net segmentation",
                                                 "yolo": "YOLO object detection + spectral"}[e])
    run = st.button("🚀 Run AI analysis", type="primary", use_container_width=True, disabled=data is None)

with right:
    if data is not None and not run and st.session_state.get("last_name") != name:
        st.image(data, caption=name, use_container_width=True)

if run and data is not None:
    with st.spinner("Running computer-vision pipeline…"):
        try:
            title = SAMPLES.get(name, (name,))[0]
            st.session_state.outcome = services.analyze_image(
                data, name, gsd_m=gsd, sensitivity=sens, engine=engine, lat=lat, lon=lon, title=title,
                source="sample" if src == "Demo sample" else "upload",
                segmenter=ui.get_segmenter(), detector=ui.get_detector())
            st.session_state.last_name = name
            st.session_state.saved = False
        except Exception as exc:
            st.error(f"Could not analyse this image: {exc}")

o = st.session_state.get("outcome")
if o is not None and o.vision is not None and o.source in ("upload", "sample"):
    v = o.vision
    with right:
        tab1, tab2, tab3 = st.tabs(["Detections", "Land cover", "Issue heat map"])
        show_boxes = True
        with tab1:
            show_boxes = st.toggle("Show bounding boxes", True)
            ov = vision.render_overlay(v.image, v.class_map, v.regions, show_boxes=show_boxes, issues=v.issues)
            st.image(ov, use_container_width=True)
        with tab2:
            st.image(vision.colorize(v.class_map), use_container_width=True)
        with tab3:
            st.image(vision.issue_heatmap(v.image, v.issues), use_container_width=True)
        st.markdown(ui.class_legend(), unsafe_allow_html=True)

    st.divider()
    st.markdown(f"### Results · {o.title}")
    for n in o.notes:
        st.info(n)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ui.kpi("Risk (image + location)", f"{o.risk.overall:.0f}/100",
               ui.level_badge(o.risk.level, o.risk.color))
    with k2:
        ui.kpi("Risk (image only)", f"{o.risk_image_only.overall:.0f}/100",
               ui.level_badge(o.risk_image_only.level, o.risk_image_only.color))
    with k3:
        ui.kpi("Vegetation cover", f"{v.metrics['vegetation_cover'] * 100:.0f}%",
               f"{v.metrics['vegetation_health'] * 100:.0f}% of it healthy")
    with k4:
        ui.kpi("Area analysed", f"{v.total_area_ha:,.0f} ha" if v.total_area_ha else "—", f"Engine: {v.engine}")

    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Issue severity")
        st.plotly_chart(ui.issue_bars(o.issues), use_container_width=True)
    with b:
        st.markdown("#### Land-cover composition")
        st.plotly_chart(ui.class_bars(v.fractions), use_container_width=True)

    st.markdown("#### Detected problems")
    det = [k for k in sorted(o.issues, key=lambda k: -o.issues[k]["severity"]) if o.issues[k]["detected"]]
    if not det:
        st.success("No significant environmental problems detected in this image.")
    for k in det:
        ui.issue_card(k, o.issues[k])

    with st.expander(f"Affected regions ({len(v.regions)})"):
        if v.regions:
            df = pd.DataFrame([r.to_dict() for r in v.regions])
            df["issue"] = df["issue"].map(lambda k: vision.ISSUES[k]["label"])
            st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown("#### Export & next steps")
    e1, e2, e3, e4 = st.columns(4)
    png = db.png_bytes(vision.render_overlay(v.image, v.class_map, v.regions, issues=v.issues))
    report = {**o.record(), "analysis": v.summary()}
    e1.download_button("⬇️ Overlay PNG", png, "terramind_overlay.png", "image/png", use_container_width=True)
    e2.download_button("⬇️ JSON report", json.dumps(report, indent=2, default=str), "terramind_report.json",
                       "application/json", use_container_width=True)
    if e3.button("💾 Save to database", disabled=st.session_state.get("saved", False), use_container_width=True):
        ui.get_store().save(o.record(), png)
        st.session_state.saved = True
        st.success(f"Saved to {ui.get_store().name}.")
    e4.page_link("pages/5_Restoration.py", label="Restoration plan →", icon="🌱")
