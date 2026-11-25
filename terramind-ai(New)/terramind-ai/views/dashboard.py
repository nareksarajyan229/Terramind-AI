"""Վահանակ․ պահպանված վերլուծությունների վիճակագրություն և ազգային ցուցանիշներ։"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from terramind import armenia, config, risk, services, ui, vision

ui.setup()
ui.hero("Վերլուծությունների վահանակ", "Բոլոր պահպանված վերլուծությունների վիճակագրությունն ըստ խնդիրների, "
        "մարզերի և ժամանակի։")

SOURCE_HY = {"upload": "Վերբեռնում", "sample": "Օրինակ", "gee": "Earth Engine", "hotspot": "Տեղեկատու"}
store = ui.get_store()
rows = store.list()

if not rows:
    st.info("Դեռ պահպանված վերլուծություններ չկան։ Կատարեք վերլուծություն «Քարտեզ» կամ «Պատկերի "
            "վերլուծություն» էջում և պահպանեք այն, կամ բեռնեք ցուցադրական տվյալներ։")
    if st.button("Բեռնել ցուցադրական տվյալներ (4 պատկեր + 15 թեժ կետ)", type="primary"):
        with st.spinner("Օրինակների վերլուծություն…"):
            spots = {"forest_clearing_tavush.png": "tavush_forests", "lake_sevan_bloom.png": "sevan",
                     "ararat_drought_farmland.png": "ararat_aquifer", "mining_tailings_lori.png": "akhtala_nahatak"}
            for fn, hid in spots.items():
                h = armenia.get_hotspot(hid)
                o = services.analyze_image((config.SAMPLES_DIR / fn).read_bytes(), fn, gsd_m=10, sensitivity=1.0,
                                           engine="auto", lat=h.lat, lon=h.lon, title=f"Օրինակ · {h.name}",
                                           source="sample")
                store.save(o.record())
            for h in armenia.HOTSPOTS:
                o = services.analyze_location(h.lat, h.lon, use_gee=False)
                o.title = f"Տեղեկատու · {h.name}"
                store.save(o.record())
        st.rerun()
    st.stop()

df = pd.DataFrame(rows)
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
df["issue_label"] = df["top_issue"].map(lambda k: vision.ISSUES[k]["label"] if k in vision.ISSUES else "Չի հայտնաբերվել")

# --- ֆիլտրեր ---------------------------------------------------------------------
PERIODS = {7: "Վերջին 7 օր", 30: "Վերջին 30 օր", 90: "Վերջին 90 օր", 365: "Վերջին տարի", 0: "Ամբողջ ժամանակը"}
f1, f2, f3 = st.columns([1.2, 1.2, 1])
marz_f = f1.multiselect("Մարզ", sorted(df["marz"].dropna().unique()), placeholder="Բոլոր մարզերը")
src_f = f2.multiselect("Աղբյուր", sorted(df["source"].unique()), format_func=lambda s: SOURCE_HY.get(s, s),
                       placeholder="Բոլոր աղբյուրները")
days = f3.selectbox("Ժամանակահատված", list(PERIODS), index=4, format_func=PERIODS.get)
if marz_f:
    df = df[df["marz"].isin(marz_f)]
if src_f:
    df = df[df["source"].isin(src_f)]
if days:
    df = df[df["created_at"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)]
if df.empty:
    st.warning("Այս ֆիլտրերին համապատասխանող վերլուծություններ չկան։")
    st.stop()

high = df["risk_level"].isin(["High", "Critical"])
k1, k2, k3, k4 = st.columns(4)
with k1:
    ui.kpi("Վերլուծություններ", f"{len(df)}", f"Պահոց՝ {store.name}")
with k2:
    avg = df["risk_score"].mean()
    lvl, col = risk.level_for(avg)
    ui.kpi("Միջին ռիսկ", f"{avg:.0f}/100", ui.level_badge(lvl, col))
with k3:
    ui.kpi("Բարձր կամ կրիտիկական", f"{high.sum()}", f"վերլուծությունների {high.mean() * 100:.0f}%-ը")
with k4:
    area = df["area_ha"].dropna().sum()
    ui.kpi("Վերլուծված տարածք (պատկերներ)", f"{area:,.0f} հա" if area else "—", "Գնահատված՝ ըստ պատկերի լուծաչափի")

st.write("")
a, b = st.columns(2)
with a:
    st.markdown("#### Ամենահաճախ հանդիպող հիմնական խնդիրը")
    c = df.groupby("top_issue", dropna=False).size().reset_index(name="n")
    c["top_issue"] = c["top_issue"].fillna("none")
    c = c.sort_values("n")
    fig = go.Figure(go.Bar(
        x=c["n"], y=[vision.ISSUES[k]["label"] if k in vision.ISSUES else "Չի հայտնաբերվել" for k in c["top_issue"]],
        orientation="h", text=c["n"], textposition="outside",
        marker=dict(color=[ui.ISSUE_COLORS.get(k, "#b0aea8") for k in c["top_issue"]], cornerradius=4,
                    line=dict(color=ui.SURFACE, width=2)),
        hovertemplate="%{y}<br>%{x} վերլուծություն<extra></extra>"))
    fig.update_xaxes(title="Վերլուծություններ", dtick=1 if c["n"].max() < 12 else None)
    st.plotly_chart(ui._layout(fig, 300), use_container_width=True)
with b:
    st.markdown("#### Ռիսկի մակարդակների բաշխում")
    order = ["Low", "Moderate", "High", "Critical"]
    colors = {n: col for _, n, col in risk.LEVELS}
    cnt = df["risk_level"].value_counts().reindex(order, fill_value=0)
    fig = go.Figure(go.Bar(
        x=[f"{ui.LEVEL_ICONS[n]} {ui.level_name(n)}" for n in order], y=cnt.values,
        text=cnt.values, textposition="outside",
        marker=dict(color=[colors[n] for n in order], cornerradius=4, line=dict(color=ui.SURFACE, width=2)),
        hovertemplate="%{x}<br>%{y} վերլուծություն<extra></extra>"))
    fig.update_yaxes(title="Վերլուծություններ")
    st.plotly_chart(ui._layout(fig, 300), use_container_width=True)

a, b = st.columns(2)
with a:
    st.markdown("#### Խնդիրների միջին վտանգավորություն")
    sev = {}
    for iss in df["issues"]:
        for k, v in (iss or {}).items():
            sev.setdefault(k, []).append(v.get("severity", 0))
    mean_sev = {k: {"severity": sum(v) / len(v)} for k, v in sev.items() if k in vision.ISSUES}
    if mean_sev:
        st.plotly_chart(ui.issue_bars(mean_sev), use_container_width=True)
with b:
    st.markdown("#### Միջին ռիսկն ըստ մարզերի")
    g = df.dropna(subset=["marz"]).groupby("marz").agg(risk=("risk_score", "mean"), n=("id", "count")).reset_index()
    g = g.sort_values("risk")
    fig = go.Figure(go.Bar(
        x=g["risk"], y=g["marz"], orientation="h", customdata=g["n"], text=g["risk"].round(0), textposition="outside",
        marker=dict(color="#2a78d6", cornerradius=4, line=dict(color=ui.SURFACE, width=2)),
        hovertemplate="%{y}<br>Միջին ռիսկ՝ %{x:.0f}/100 · %{customdata} վերլուծություն<extra></extra>"))
    fig.update_xaxes(range=[0, 110], title="Միջին ռիսկ (0–100)")
    st.plotly_chart(ui._layout(fig, 300), use_container_width=True)

if df["created_at"].dt.date.nunique() > 1:
    st.markdown("#### Ռիսկի դինամիկան")
    t = df.set_index("created_at").resample("D")["risk_score"].mean().dropna().reset_index()
    fig = go.Figure(go.Scatter(x=t["created_at"], y=t["risk_score"], mode="lines+markers",
                               line=dict(color="#2a78d6", width=2), marker=dict(size=8, line=dict(color=ui.SURFACE, width=2)),
                               hovertemplate="%{x|%d.%m.%Y}<br>Միջին ռիսկ՝ %{y:.0f}<extra></extra>"))
    fig.update_yaxes(range=[0, 100], title="Միջին ռիսկ")
    st.plotly_chart(ui._layout(fig, 280), use_container_width=True)

st.markdown("#### Պահպանված վերլուծություններ")
show = df[["created_at", "title", "marz", "source", "engine", "risk_score", "risk_level", "issue_label"]].copy()
show["created_at"] = show["created_at"].dt.strftime("%d.%m.%Y %H:%M")
show["source"] = show["source"].map(lambda s: SOURCE_HY.get(s, s))
show["risk_level"] = show["risk_level"].map(ui.level_name)
show = show.rename(columns={"created_at": "Ամսաթիվ", "title": "Անվանում", "marz": "Մարզ", "source": "Աղբյուր",
                            "engine": "Համակարգ", "risk_score": "Ռիսկ", "risk_level": "Մակարդակ",
                            "issue_label": "Հիմնական խնդիր"})
st.dataframe(show, hide_index=True, use_container_width=True,
             column_config={"Ռիսկ": st.column_config.ProgressColumn("Ռիսկ", min_value=0, max_value=100, format="%.0f")})
st.download_button("⬇️ Արտահանել CSV", show.to_csv(index=False).encode("utf-8-sig"), "terramind_analyses.csv", "text/csv")

st.divider()
st.markdown("#### Բնապահպանական ազգային ցուցանիշներ")
cols = st.columns(3)
for i, ind in enumerate(armenia.NATIONAL_INDICATORS):
    with cols[i % 3]:
        ui.kpi(ind["label"], ind["value"], f'{ind["note"]} · <a href="{ind["source"]}" target="_blank">աղբյուր</a>')
        st.write("")
