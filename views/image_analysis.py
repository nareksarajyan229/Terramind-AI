"""Պատկերի վերլուծություն․ հողածածկույթ, խնդիրներ, ծածկույթներ, հաշվետվություն։"""
import json

import pandas as pd
import streamlit as st

from terramind import armenia, config, db, services, ui, vision

ui.setup()
ui.hero("Պատկերի վերլուծություն", "Վերբեռնեք արբանյակային կամ դրոնային պատկեր։ TerraMind-ը կդասակարգի "
        "հողածածկույթը, կհայտնաբերի բնապահպանական խնդիրները և կնշի տուժած տարածքները։")

SAMPLES = {
    "forest_clearing_tavush.png": ("Անտառային բացատներ (Տավուշի օրինակ)", "tavush_forests"),
    "lake_sevan_bloom.png": ("Լիճ՝ ջրիմուռների ծաղկումով և պղտոր ներհոսքով (Սևանի օրինակ)", "sevan"),
    "ararat_drought_farmland.png": ("Չոր դաշտեր՝ առվակային էրոզիայով (Արարատի օրինակ)", "ararat_aquifer"),
    "mining_tailings_lori.png": ("Բաց հանքահոր և պոչամբար (Լոռու օրինակ)", "akhtala_nahatak"),
}
ENGINES = {"auto": "Ավտոմատ (լավագույն հասանելին)", "spectral": "OpenCV սպեկտրալ կանոններ",
           "segmentation": "PyTorch U-Net սեգմենտացիա", "yolo": "YOLO հայտնաբերում + սպեկտրալ"}
SOURCES = ["Վերբեռնել պատկեր", "Ցուցադրական օրինակ"]

left, right = st.columns([1, 1.6])
with left:
    st.markdown("#### 1 · Մուտքային տվյալներ")
    src = st.radio("Աղբյուր", SOURCES, horizontal=True)
    data = name = None
    default_spot = "—"
    if src == SOURCES[0]:
        f = st.file_uploader("Արբանյակային / օդային պատկեր", type=["png", "jpg", "jpeg", "tif", "tiff"],
                             help="RGB պատկերներն աշխատում են ամենուր։ Sentinel-2 կամ 4-շերտ (B,G,R,NIR) "
                                  "GeoTIFF-երը թույլ են տալիս NDVI/NDWI/NDCI վերլուծություն (պահանջում է rasterio)։")
        if f:
            data, name = f.getvalue(), f.name
    else:
        pick = st.selectbox("Օրինակ", list(SAMPLES), format_func=lambda k: SAMPLES[k][0])
        data, name = (config.SAMPLES_DIR / pick).read_bytes(), pick
        default_spot = SAMPLES[pick][1]
        st.caption("Ցուցադրական սինթետիկ պատկեր է, ոչ թե իրական արբանյակային լուսանկար։")

    st.markdown("#### 2 · Տեղանք (ըստ ցանկության)")
    spots = ["—"] + [h.id for h in armenia.HOTSPOTS] + ["custom"]
    loc = st.selectbox("Որտե՞ղ է արված պատկերը", spots, index=spots.index(default_spot),
                       format_func=lambda i: {"—": "Անհայտ", "custom": "Այլ կոորդինատներ"}.get(
                           i, armenia.get_hotspot(i).name if armenia.get_hotspot(i) else i))
    lat = lon = None
    if loc == "custom":
        c1, c2 = st.columns(2)
        lat = c1.number_input("Լայնություն", value=40.18, format="%.5f")
        lon = c2.number_input("Երկայնություն", value=44.51, format="%.5f")
    elif loc != "—":
        h = armenia.get_hotspot(loc)
        lat, lon = h.lat, h.lon

    st.markdown("#### 3 · Կարգավորումներ")
    gsd = st.number_input("Լուծաչափ (մ / պիքսել)", 0.1, 100.0, 10.0, 0.5,
                          help="Sentinel-2՝ 10 մ, Landsat՝ 30 մ, դրոն՝ ≈ 0.05–0.5 մ։ Օգտագործվում է հեկտարների հաշվարկի համար։")
    sens = st.slider("Հայտնաբերման զգայունություն", 0.6, 1.5, 1.0, 0.05,
                     help="Ավելի բարձր արժեքը նշում է ավելի շատ հնարավոր խնդիրներ")
    engine = st.selectbox("ԱԲ համակարգ", list(ENGINES), format_func=ENGINES.get)
    run = st.button("🚀 Սկսել ԱԲ վերլուծությունը", type="primary", use_container_width=True, disabled=data is None)

with right:
    if data is not None and not run and st.session_state.get("last_name") != name:
        st.image(data, caption=name, use_container_width=True)

if run and data is not None:
    with st.spinner("Համակարգչային տեսողության վերլուծություն…"):
        try:
            title = SAMPLES.get(name, (name,))[0]
            st.session_state.outcome = services.analyze_image(
                data, name, gsd_m=gsd, sensitivity=sens, engine=engine, lat=lat, lon=lon, title=title,
                source="sample" if src == SOURCES[1] else "upload",
                segmenter=ui.get_segmenter(), detector=ui.get_detector())
            st.session_state.last_name = name
            st.session_state.saved = False
        except Exception as exc:
            st.error(f"Չհաջողվեց վերլուծել այս պատկերը․ {exc}")

o = st.session_state.get("outcome")
if o is not None and o.vision is not None and o.source in ("upload", "sample"):
    v = o.vision
    with right:
        tab1, tab2, tab3 = st.tabs(["Հայտնաբերումներ", "Հողածածկույթ", "Խնդիրների ջերմային քարտեզ"])
        with tab1:
            show_boxes = st.toggle("Ցույց տալ շրջանակները", True)
            ov = vision.render_overlay(v.image, v.class_map, v.regions, show_boxes=show_boxes, issues=v.issues)
            st.image(ov, use_container_width=True)
            st.markdown(ui.box_legend(), unsafe_allow_html=True)
        with tab2:
            st.image(vision.colorize(v.class_map), use_container_width=True)
        with tab3:
            st.image(vision.issue_heatmap(v.image, v.issues), use_container_width=True)
        st.markdown(ui.class_legend(), unsafe_allow_html=True)

    st.divider()
    st.markdown(f"### Արդյունքներ · {o.title}")
    for n in o.notes:
        st.info(n)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ui.kpi("Ռիսկ (պատկեր + տեղանք)", f"{o.risk.overall:.0f}/100", ui.level_badge(o.risk.level, o.risk.color))
    with k2:
        ui.kpi("Ռիսկ (միայն պատկեր)", f"{o.risk_image_only.overall:.0f}/100",
               ui.level_badge(o.risk_image_only.level, o.risk_image_only.color))
    with k3:
        ui.kpi("Բուսածածկույթ", f"{v.metrics['vegetation_cover'] * 100:.0f}%",
               f"որից առողջ՝ {v.metrics['vegetation_health'] * 100:.0f}%")
    with k4:
        ui.kpi("Վերլուծված տարածք", f"{v.total_area_ha:,.0f} հա" if v.total_area_ha else "—", f"Համակարգ՝ {v.engine}")

    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Խնդիրների վտանգավորություն")
        st.plotly_chart(ui.issue_bars(o.issues), use_container_width=True)
    with b:
        st.markdown("#### Հողածածկույթի կազմը")
        st.plotly_chart(ui.class_bars(v.fractions), use_container_width=True)

    st.markdown("#### Հայտնաբերված խնդիրներ")
    det = [k for k in sorted(o.issues, key=lambda k: -o.issues[k]["severity"]) if o.issues[k]["detected"]]
    if not det:
        st.success("Այս պատկերում էական բնապահպանական խնդիրներ չեն հայտնաբերվել։")
    for k in det:
        ui.issue_card(k, o.issues[k])

    with st.expander(f"Տուժած տարածքներ ({len(v.regions)})"):
        if v.regions:
            df = pd.DataFrame([r.to_dict() for r in v.regions])
            df["issue"] = df["issue"].map(lambda k: vision.ISSUES[k]["label"])
            st.dataframe(df.rename(columns={"issue": "Խնդիր", "bbox": "Շրջանակ (x, y, w, h)", "area_px": "Պիքսել",
                                            "area_ha": "Հեկտար", "confidence": "Վստահություն", "source": "Աղբյուր"}),
                         hide_index=True, use_container_width=True)

    st.markdown("#### Արտահանում և հաջորդ քայլեր")
    e1, e2, e3, e4 = st.columns(4)
    png = db.png_bytes(vision.render_overlay(v.image, v.class_map, v.regions, issues=v.issues))
    report = {**o.record(), "analysis": v.summary()}
    e1.download_button("⬇️ Պատկեր (PNG)", png, "terramind_overlay.png", "image/png", use_container_width=True)
    e2.download_button("⬇️ Հաշվետվություն (JSON)", json.dumps(report, indent=2, ensure_ascii=False, default=str),
                       "terramind_report.json", "application/json", use_container_width=True)
    if e3.button("💾 Պահպանել", disabled=st.session_state.get("saved", False), use_container_width=True):
        ui.get_store().save(o.record(), png)
        st.session_state.saved = True
        st.success(f"Պահպանվեց՝ {ui.get_store().name}։")
    with e4:
        ui.link("restoration", "Վերականգնման ծրագիր →", "🌱")
