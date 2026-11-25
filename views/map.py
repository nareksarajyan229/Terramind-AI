"""Ինտերակտիվ քարտեզ․ թեժ կետեր, Earth Engine շերտեր և կետի վերլուծություն։"""
import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Fullscreen, MiniMap
from streamlit_folium import st_folium

from terramind import armenia, gee, services, ui, vision

ui.setup()
ui.hero("Ինտերակտիվ քարտեզ", "Հայաստանի փաստագրված բնապահպանական թեժ կետերը։ Սեղմեք քարտեզի ցանկացած "
        "կետի վրա՝ այն վերլուծելու համար։")

if "pick" not in st.session_state:
    st.session_state.pick = armenia.ARMENIA_CENTER

LAYERS = {"None": "Չկա", "true_color": "Sentinel-2 բնական գույներ", "ndvi": "NDVI (բուսականություն)",
          "ndwi": "NDWI (ջուր)", "ndci": "NDCI (ջրիմուռներ ջրում)", "bsi": "Մերկ հողի ինդեքս",
          "forest_loss": "Անտառի կորուստ 2000 թ.-ից"}

# --- կառավարում -----------------------------------------------------------------
c1, c2, c3, c4 = st.columns([1.4, 1.2, 1, 1])
cats = c1.multiselect("Խնդիրների կատեգորիաներ", list(armenia.CATEGORIES), default=list(armenia.CATEGORIES),
                      format_func=lambda k: armenia.CATEGORIES[k]["label"])
marz = c2.multiselect("Մարզ", armenia.MARZER, placeholder="Բոլոր մարզերը")
base = c3.selectbox("Հիմնաքարտեզ", ["Արբանյակային (Esri)", "Ռելիեֆ (OpenTopoMap)", "Փողոցներ (OSM)"])
ee_ok = ui.gee_ready()
layer = c4.selectbox("Earth Engine շերտ", list(LAYERS), format_func=LAYERS.get, disabled=not ee_ok,
                     help=None if ee_ok else "Շերտերի համար միացրեք Google Earth Engine-ը")

hs = [h for h in armenia.HOTSPOTS if h.category in cats and (not marz or h.marz in marz)]

# --- քարտեզ ----------------------------------------------------------------------
tiles = {
    "Արբանյակային (Esri)": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                            "Esri, Maxar, Earthstar Geographics"),
    "Ռելիեֆ (OpenTopoMap)": ("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", "© OpenTopoMap (CC-BY-SA)"),
    "Փողոցներ (OSM)": ("OpenStreetMap", None),
}
url, attr = tiles[base]
m = folium.Map(location=armenia.ARMENIA_CENTER, zoom_start=8, tiles=None, control_scale=True)
folium.TileLayer(url, attr=attr, name=base).add_to(m)

if layer != "None" and ee_ok:
    try:
        lat0, lon0 = st.session_state.pick
        folium.TileLayer(gee.tile_layer(layer, lat0, lon0, 20000), attr="Google Earth Engine",
                         name=f"GEE · {LAYERS[layer]}", overlay=True, opacity=0.85).add_to(m)
    except Exception as exc:
        st.warning(f"Չհաջողվեց բեռնել Earth Engine շերտը․ {exc}")

groups = {k: folium.FeatureGroup(name=v["label"]) for k, v in armenia.CATEGORIES.items() if k in cats}
for h in hs:
    cat = armenia.CATEGORIES[h.category]
    src = "".join(f'<li><a href="{s}" target="_blank">{s.split("/")[2]}</a></li>' for s in h.sources)
    html = (f"<div style='width:290px;font-family:sans-serif'><b>{h.name}</b><br>"
            f"<small>{h.marz} · {cat['label']} · վտանգավորություն {h.severity}/5 · {h.trend_hy}</small>"
            f"<p style='margin:6px 0'>{h.summary}</p><b>Պատճառներ․</b> {', '.join(h.drivers)}"
            f"{'<br><b>Աղբյուրներ․</b><ul style=margin:0>' + src + '</ul>' if src else ''}</div>")
    folium.CircleMarker(
        [h.lat, h.lon], radius=6 + h.severity * 2.4, color="#ffffff", weight=2, fill=True,
        fill_color=cat["color"], fill_opacity=0.85, tooltip=f"{h.name} — {cat['label']}",
        popup=folium.Popup(html, max_width=320)).add_to(groups[h.category])
for g in groups.values():
    g.add_to(m)

lat_p, lon_p = st.session_state.pick
folium.Marker([lat_p, lon_p], tooltip="Ընտրված կետ",
              icon=folium.Icon(color="darkgreen", icon="crosshairs", prefix="fa")).add_to(m)
folium.LayerControl(collapsed=True).add_to(m)
Fullscreen(title="Լիաէկրան", title_cancel="Դուրս գալ").add_to(m)
MiniMap(toggle_display=True).add_to(m)

mcol, side = st.columns([2.3, 1])
with mcol:
    out = st_folium(m, height=560, use_container_width=True, returned_objects=["last_clicked"])
    st.markdown(ui.legend_html([(v["label"], v["color"]) for k, v in armenia.CATEGORIES.items() if k in cats]),
                unsafe_allow_html=True)
    st.caption("Նշիչի չափը ցույց է տալիս ինդիկատիվ վտանգավորությունը։ Սեղմեք նշիչի վրա՝ մանրամասների "
               "և աղբյուրների համար։")
if out and out.get("last_clicked"):
    p = (round(out["last_clicked"]["lat"], 5), round(out["last_clicked"]["lng"], 5))
    if p != st.session_state.pick:
        st.session_state.pick = p
        st.rerun()

with side:
    st.markdown("#### Վերլուծել տեղանքը")
    jump = st.selectbox("Անցնել թեժ կետին", ["—"] + [h.id for h in armenia.HOTSPOTS],
                        format_func=lambda i: "—" if i == "—" else armenia.get_hotspot(i).name)
    if jump != "—":
        h = armenia.get_hotspot(jump)
        if (h.lat, h.lon) != st.session_state.pick:
            st.session_state.pick = (h.lat, h.lon)
            st.rerun()
    lat = st.number_input("Լայնություն", value=float(st.session_state.pick[0]), format="%.5f")
    lon = st.number_input("Երկայնություն", value=float(st.session_state.pick[1]), format="%.5f")
    radius = st.slider("Վերլուծության շառավիղ (կմ)", 1.0, 15.0, 3.0, 0.5)
    if not armenia.in_armenia(lat, lon):
        st.warning("Այս կետը Հայաստանի սահմաններից դուրս է։")
    mode = ("Sentinel-2 իրական տվյալներ + ԱԲ (Earth Engine)" if ee_ok
            else "Տեղեկատու պատկեր (Earth Engine-ը միացված չէ)")
    st.caption(f"Ռեժիմ՝ {mode}")
    if st.button("🔍 Վերլուծել այս տեղանքը", type="primary", use_container_width=True):
        with st.spinner("Արբանյակային տվյալների վերլուծություն…" if ee_ok else "Տեղեկատու պատկերի կառուցում…"):
            try:
                st.session_state.outcome = services.analyze_location(
                    lat, lon, radius * 1000, use_gee=ee_ok,
                    segmenter=ui.get_segmenter(), detector=ui.get_detector())
                st.session_state.saved = False
            except Exception as exc:
                st.error(f"Վերլուծությունը ձախողվեց․ {exc}")

    near = armenia.nearest_hotspots(lat, lon, k=3, max_km=50)
    if near:
        st.markdown("**Մոտակա փաստագրված թեժ կետեր**")
        for h, d in near:
            st.markdown(f"- {h.name} — {d:.0f} կմ")

# --- արդյունքներ ---------------------------------------------------------------
o = st.session_state.get("outcome")
if o is not None and o.source in ("gee", "hotspot"):
    st.divider()
    st.markdown(f"### Արդյունքներ · {o.title}" + (f" · {o.marz}" if o.marz else ""))
    for n in o.notes:
        st.info(n)
    r1, r2 = st.columns([1, 1.6])
    with r1:
        st.plotly_chart(ui.gauge(o.risk.overall, o.risk.level, o.risk.color), use_container_width=True)
        st.markdown(ui.level_badge(o.risk.level, o.risk.color, o.risk.overall), unsafe_allow_html=True)
        st.caption("Հաշվի է առնված փաստագրված թեժ կետերի մոտիկությունը։")
    with r2:
        st.plotly_chart(ui.issue_bars(o.issues), use_container_width=True)

    for k in sorted(o.issues, key=lambda k: -o.issues[k]["severity"]):
        if o.issues[k]["severity"] >= 15:
            ui.issue_card(k, o.issues[k])

    if o.vision is not None:
        st.markdown("#### Sentinel-2 պատկերի վերլուծություն")
        a, b = st.columns(2)
        a.image(o.vision.image, caption="Sentinel-2 բնական գույներ (վեգետացիոն շրջանի միջին)", use_container_width=True)
        b.image(vision.render_overlay(o.vision.image, o.vision.class_map, o.vision.regions, issues=o.vision.issues),
                caption="ԱԲ դասակարգում և հայտնաբերված տարածքներ", use_container_width=True)
        st.markdown(ui.class_legend(), unsafe_allow_html=True)
        st.markdown(ui.box_legend(), unsafe_allow_html=True)

    if o.gee_stats:
        with st.expander("Earth Engine չափումներ"):
            s = dict(o.gee_stats)
            st.dataframe(pd.DataFrame({"Ցուցանիշ": list(s), "Արժեք": [str(v) for v in s.values()]}),
                         hide_index=True, use_container_width=True)
        try:
            ts = gee.ndvi_timeseries(round(o.lat, 4), round(o.lon, 4), min(radius * 1000, 5000))
            if ts:
                df = pd.DataFrame(ts)
                fig = go.Figure(go.Scatter(x=pd.to_datetime(df["date"]), y=df["ndvi"], mode="lines",
                                           line=dict(color="#2a78d6", width=2),
                                           hovertemplate="%{x|%m.%Y}<br>NDVI %{y:.2f}<extra></extra>"))
                fig.update_layout(title="Բուսականության միտումը — MODIS NDVI (16-օրյա)")
                fig.update_yaxes(title="NDVI")
                st.plotly_chart(ui._layout(fig, 300), use_container_width=True)
        except Exception as exc:
            st.caption(f"NDVI-ի ժամանակային շարքը հասանելի չէ․ {exc}")

    s1, s2, s3 = st.columns(3)
    if s1.button("💾 Պահպանել վերլուծությունը", disabled=st.session_state.get("saved", False), use_container_width=True):
        ui.get_store().save(o.record())
        st.session_state.saved = True
        st.success(f"Պահպանվեց՝ {ui.get_store().name}։")
    with s2:
        ui.link("risk_assessment", "Ռիսկերի գնահատում →", "⚠️")
    with s3:
        ui.link("restoration", "Վերականգնման ծրագիր →", "🌱")
