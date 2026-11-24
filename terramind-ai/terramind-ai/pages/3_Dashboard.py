"""Analysis dashboard: statistics across saved analyses + national indicators."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from terramind import armenia, config, risk, services, ui, vision

ui.setup("Dashboard", "📊")
ui.hero("Analysis dashboard", "Statistics across every saved analysis, by issue, province and time.")

store = ui.get_store()
rows = store.list()

if not rows:
    st.info("No saved analyses yet. Run an analysis on the Map or Image Analysis page and save it — "
            "or load demo data to explore the dashboard.")
    if st.button("Load demo data (4 sample images + 15 hotspot profiles)", type="primary"):
        with st.spinner("Analysing samples…"):
            spots = {"forest_clearing_tavush.png": "tavush_forests", "lake_sevan_bloom.png": "sevan",
                     "ararat_drought_farmland.png": "ararat_aquifer", "mining_tailings_lori.png": "akhtala_nahatak"}
            for fn, hid in spots.items():
                h = armenia.get_hotspot(hid)
                o = services.analyze_image((config.SAMPLES_DIR / fn).read_bytes(), fn, gsd_m=10, sensitivity=1.0,
                                           engine="auto", lat=h.lat, lon=h.lon, title=f"Demo · {fn}", source="sample")
                store.save(o.record())
            for h in armenia.HOTSPOTS:
                o = services.analyze_location(h.lat, h.lon, use_gee=False)
                o.title = f"Reference · {h.name}"
                store.save(o.record())
        st.rerun()
    st.stop()

df = pd.DataFrame(rows)
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
df["issue_label"] = df["top_issue"].map(lambda k: vision.ISSUES[k]["label"] if k in vision.ISSUES else "None detected")

# --- filters (one row above the charts) -----------------------------------------
f1, f2, f3 = st.columns([1.2, 1.2, 1])
marz_f = f1.multiselect("Marz", sorted(df["marz"].dropna().unique()), placeholder="All provinces")
src_f = f2.multiselect("Source", sorted(df["source"].unique()), placeholder="All sources")
days = f3.selectbox("Period", [7, 30, 90, 365, 0], index=4, format_func=lambda d: "All time" if d == 0 else f"Last {d} days")
if marz_f:
    df = df[df["marz"].isin(marz_f)]
if src_f:
    df = df[df["source"].isin(src_f)]
if days:
    df = df[df["created_at"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)]
if df.empty:
    st.warning("No analyses match these filters.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
with k1:
    ui.kpi("Analyses", f"{len(df)}", f"Storage: {store.name}")
with k2:
    avg = df["risk_score"].mean()
    lvl, col = risk.level_for(avg)
    ui.kpi("Average risk", f"{avg:.0f}/100", ui.level_badge(lvl, col))
with k3:
    ui.kpi("High or critical", f"{(df['risk_level'].isin(['High', 'Critical'])).sum()}",
           f"{(df['risk_level'].isin(['High', 'Critical'])).mean() * 100:.0f}% of analyses")
with k4:
    area = df["area_ha"].dropna().sum()
    ui.kpi("Area analysed (images)", f"{area:,.0f} ha" if area else "—", "Estimated from image resolution")

st.write("")
a, b = st.columns(2)
with a:
    st.markdown("#### Most common primary issue")
    c = df.groupby("top_issue", dropna=False).size().reset_index(name="n")
    c["top_issue"] = c["top_issue"].fillna("none")
    c = c.sort_values("n")
    fig = go.Figure(go.Bar(
        x=c["n"], y=[vision.ISSUES[k]["label"] if k in vision.ISSUES else "None detected" for k in c["top_issue"]],
        orientation="h", text=c["n"], textposition="outside",
        marker=dict(color=[ui.ISSUE_COLORS.get(k, "#b0aea8") for k in c["top_issue"]], cornerradius=4,
                    line=dict(color=ui.SURFACE, width=2)),
        hovertemplate="%{y}<br>%{x} analyses<extra></extra>"))
    fig.update_xaxes(title="Analyses", dtick=1 if c["n"].max() < 12 else None)
    st.plotly_chart(ui._layout(fig, 300), use_container_width=True)
with b:
    st.markdown("#### Risk level distribution")
    order = ["Low", "Moderate", "High", "Critical"]
    ICONS = {"Low": "●", "Moderate": "▲", "High": "◆", "Critical": "■"}
    colors = {n: col for _, n, col in risk.LEVELS}
    cnt = df["risk_level"].value_counts().reindex(order, fill_value=0)
    fig = go.Figure(go.Bar(
        x=[f"{ICONS[n]} {n}" for n in order], y=cnt.values,
        text=cnt.values, textposition="outside",
        marker=dict(color=[colors[n] for n in order], cornerradius=4, line=dict(color=ui.SURFACE, width=2)),
        hovertemplate="%{x}<br>%{y} analyses<extra></extra>"))
    fig.update_yaxes(title="Analyses")
    st.plotly_chart(ui._layout(fig, 300), use_container_width=True)

a, b = st.columns(2)
with a:
    st.markdown("#### Average issue severity")
    sev = {}
    for iss in df["issues"]:
        for k, v in (iss or {}).items():
            sev.setdefault(k, []).append(v.get("severity", 0))
    mean_sev = {k: {"severity": sum(v) / len(v)} for k, v in sev.items() if k in vision.ISSUES}
    if mean_sev:
        st.plotly_chart(ui.issue_bars(mean_sev), use_container_width=True)
with b:
    st.markdown("#### Average risk by province")
    g = df.dropna(subset=["marz"]).groupby("marz").agg(risk=("risk_score", "mean"), n=("id", "count")).reset_index()
    g = g.sort_values("risk")
    fig = go.Figure(go.Bar(
        x=g["risk"], y=g["marz"], orientation="h", customdata=g["n"], text=g["risk"].round(0), textposition="outside",
        marker=dict(color="#2a78d6", cornerradius=4, line=dict(color=ui.SURFACE, width=2)),
        hovertemplate="%{y}<br>Average risk %{x:.0f}/100 · %{customdata} analyses<extra></extra>"))
    fig.update_xaxes(range=[0, 110], title="Average risk (0–100)")
    st.plotly_chart(ui._layout(fig, 300), use_container_width=True)

if df["created_at"].dt.date.nunique() > 1:
    st.markdown("#### Risk over time")
    t = df.set_index("created_at").resample("D")["risk_score"].mean().dropna().reset_index()
    fig = go.Figure(go.Scatter(x=t["created_at"], y=t["risk_score"], mode="lines+markers",
                               line=dict(color="#2a78d6", width=2), marker=dict(size=8, line=dict(color=ui.SURFACE, width=2)),
                               hovertemplate="%{x|%d %b %Y}<br>Average risk %{y:.0f}<extra></extra>"))
    fig.update_yaxes(range=[0, 100], title="Average risk")
    st.plotly_chart(ui._layout(fig, 280), use_container_width=True)

st.markdown("#### Saved analyses")
show = df[["created_at", "title", "marz", "source", "engine", "risk_score", "risk_level", "issue_label"]].copy()
show["created_at"] = show["created_at"].dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(show.rename(columns={"created_at": "Date", "title": "Title", "marz": "Marz", "source": "Source",
                                  "engine": "Engine", "risk_score": "Risk", "risk_level": "Level",
                                  "issue_label": "Primary issue"}),
             hide_index=True, use_container_width=True,
             column_config={"Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f")})
st.download_button("⬇️ Export CSV", show.to_csv(index=False), "terramind_analyses.csv", "text/csv")

st.divider()
st.markdown("#### National environmental indicators")
cols = st.columns(3)
for i, ind in enumerate(armenia.NATIONAL_INDICATORS):
    with cols[i % 3]:
        ui.kpi(ind["label"], ind["value"], f'{ind["note"]} · <a href="{ind["source"]}" target="_blank">source</a>')
        st.write("")
