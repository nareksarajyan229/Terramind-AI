"""Environmental risk assessment for the current analysis and the national hotspot register."""
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from terramind import armenia, risk, services, ui, vision

ui.setup("Risk Assessment", "⚠️")
ui.hero("Environmental risk assessment", "How serious is it, what drives it, and where in Armenia is it worst.")

o = st.session_state.get("outcome")
st.markdown("### Current analysis")
if o is None:
    st.info("No analysis in this session yet. Pick a hotspot to assess from reference data, or run an analysis "
            "on the Map or Image Analysis page.")
    hid = st.selectbox("Assess a documented hotspot", [h.id for h in armenia.HOTSPOTS],
                       format_func=lambda i: armenia.get_hotspot(i).name)
    if st.button("Assess", type="primary"):
        h = armenia.get_hotspot(hid)
        st.session_state.outcome = services.analyze_location(h.lat, h.lon, use_gee=ui.gee_ready())
        st.rerun()
else:
    st.markdown(f"**{o.title}**" + (f" · {o.marz}" if o.marz else "") +
                (f" · {o.lat:.4f}, {o.lon:.4f}" if o.lat is not None else ""))
    a, b, c = st.columns([1, 1.2, 1])
    with a:
        st.plotly_chart(ui.gauge(o.risk.overall, o.risk.level, o.risk.color), use_container_width=True)
        st.markdown(ui.level_badge(o.risk.level, o.risk.color, o.risk.overall), unsafe_allow_html=True)
        if o.risk_image_only is not o.risk:
            st.caption(f"Without location context: {o.risk_image_only.overall:.0f}/100 ({o.risk_image_only.level})")
    with b:
        st.plotly_chart(ui.radar(o.risk.components), use_container_width=True)
    with c:
        st.markdown("**Main risk drivers**")
        if o.risk.drivers:
            for d in o.risk.drivers:
                st.markdown(f"- {d}")
        else:
            st.markdown("- No component above 25/100")
        if o.risk.context:
            st.markdown("**Nearby documented hotspots**")
            for ctx in o.risk.context:
                st.markdown(f"- {ctx['hotspot']} · {ctx['distance_km']} km · severity {ctx['severity']}/5")

    comp = pd.DataFrame([{"Issue": vision.ISSUES[k]["label"], "Score": v,
                          "Level": risk.level_for(v)[0], "Weight": risk.WEIGHTS[k],
                          "Evidence": o.issues.get(k, {}).get("evidence", "")}
                         for k, v in sorted(o.risk.components.items(), key=lambda kv: -kv[1])])
    st.dataframe(comp, hide_index=True, use_container_width=True,
                 column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f")})
    with st.expander("How the score is calculated"):
        st.markdown(
            "- Each issue gets a **severity 0–100** from the AI analysis (share and pattern of affected area, "
            "or Earth Engine indices).\n"
            "- Severity is raised toward the rating of any **documented hotspot within 40 km**, decaying with distance.\n"
            "- The overall score is a **top-weighted mean**: the worst issue counts fully, the 2nd half, the 3rd a "
            "quarter and so on, so one severe problem cannot be diluted by several minor ones.\n"
            "- Levels: **Low** < 25 ≤ **Moderate** < 50 ≤ **High** < 75 ≤ **Critical**.")

st.divider()
st.markdown("### National hotspot risk matrix")
mx = pd.DataFrame(risk.hotspot_risk_matrix())
mx["L"] = mx["likelihood"].apply(lambda v: int(min(5, max(1, math.ceil(v)))))
mx["I"] = mx["impact"].astype(int)

z, txt, hov = [], [], []
for impact in range(1, 6):
    zr, tr, hr = [], [], []
    for like in range(1, 6):
        cell = mx[(mx["L"] == like) & (mx["I"] == impact)]
        score = like * impact * 4
        zr.append(score)
        tr.append(str(len(cell)) if len(cell) else "")
        hr.append(f"Likelihood {like} × impact {impact} = {score}<br>" +
                  ("<br>".join("• " + n for n in cell["name"]) if len(cell) else "No hotspots"))
    z.append(zr); txt.append(tr); hov.append(hr)

colorscale = [[0, "#0ca30c"], [0.25, "#0ca30c"], [0.25, "#fab219"], [0.5, "#fab219"],
              [0.5, "#ec835a"], [0.75, "#ec835a"], [0.75, "#d03b3b"], [1, "#d03b3b"]]
fig = go.Figure(go.Heatmap(z=z, x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5], text=txt, texttemplate="%{text}",
                           textfont=dict(size=18, color="#0b0b0b"), customdata=hov,
                           hovertemplate="%{customdata}<extra></extra>", colorscale=colorscale, zmin=0, zmax=100,
                           showscale=False, xgap=2, ygap=2, opacity=0.75))
fig.update_xaxes(title="Likelihood (trend + severity)", dtick=1)
fig.update_yaxes(title="Impact (severity)", dtick=1)
m1, m2 = st.columns([1, 1.25])
with m1:
    st.plotly_chart(ui._layout(fig, 400), use_container_width=True)
    st.markdown(ui.legend_html([("● Low", "#0ca30c"), ("▲ Moderate", "#fab219"), ("◆ High", "#ec835a"),
                                ("■ Critical", "#d03b3b")]), unsafe_allow_html=True)
    st.caption("Numbers = hotspots in each cell. Hover a cell for names.")
with m2:
    show = mx[["name", "category", "marz", "trend", "score"]].rename(
        columns={"name": "Hotspot", "category": "Category", "marz": "Marz", "trend": "Trend", "score": "Risk"})
    st.dataframe(show, hide_index=True, use_container_width=True, height=400,
                 column_config={"Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f")})
st.caption("Hotspot ratings are indicative editorial assessments based on the cited sources, for prioritisation only.")
