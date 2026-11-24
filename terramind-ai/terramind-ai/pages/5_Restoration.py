"""Restoration planner: phased actions, native species and KPIs."""
import pandas as pd
import streamlit as st

from terramind import recommendations as rec
from terramind import ui, vision

ui.setup("Restoration", "🌱")
ui.hero("Restoration planner", "Concrete, phased actions for the problems found — with native Armenian species, "
        "partners and the indicators to track.")

o = st.session_state.get("outcome")
mode = st.radio("Plan for", ["Current analysis", "Custom scenario"], horizontal=True,
                index=0 if o is not None else 1, disabled=o is None)

if mode == "Current analysis" and o is not None:
    plan, title = o.plan, o.title
    st.caption(f"Based on: **{o.title}**" + (f" · {o.marz}" if o.marz else "") +
               f" · risk {o.risk.overall:.0f}/100 ({o.risk.level})")
else:
    st.markdown("Set the severity of each problem to build a plan without an analysis.")
    cols = st.columns(3)
    issues = {}
    for i, (k, meta) in enumerate(vision.ISSUES.items()):
        issues[k] = {"severity": cols[i % 3].slider(meta["label"], 0, 100, 0, 5)}
    zone = st.selectbox("Eco-zone", list(rec.SPECIES_BY_ZONE), format_func=lambda z: rec.SPECIES_BY_ZONE[z]["label"])
    marz = {"humid_forest": "Tavush", "arid_south": "Ararat", "highland": "Gegharkunik"}[zone]
    plan, title = rec.recommend(issues, marz=marz), "Custom scenario"

st.info(plan["summary"])
if not plan["actions"]:
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    ui.kpi("Actions", str(len(plan["actions"])), f"{sum(a['priority'] == 'Immediate' for a in plan['actions'])} immediate")
with c2:
    ui.kpi("Issues addressed", str(len({a['issue'] for a in plan['actions']})), "")
with c3:
    ui.kpi("Eco-zone", plan["zone"]["label"].split(" (")[0], plan["zone"]["label"].split("(")[-1].rstrip(")"))

st.write("")
for phase, icon in [("0–6 months", "⚡"), ("6–24 months", "🛠️"), ("2–10 years", "🌳")]:
    items = [a for a in plan["actions"] if a["phase"] == phase]
    if not items:
        continue
    st.markdown(f"### {icon} {phase}")
    for a in items:
        color = ui.ISSUE_COLORS.get(a["issue"], "#999")
        st.markdown(
            f'<div class="tm-issue" style="border-left-color:{color}"><h4>{a["title"]} '
            f'<span class="tm-muted">· {a["priority"]} · cost {a["cost"]} · {vision.ISSUES[a["issue"]]["label"]}</span></h4>'
            f'<div>{a["detail"]}</div><div class="tm-muted" style="margin-top:6px">'
            f'<b>KPI:</b> {a["kpi"]} &nbsp;·&nbsp; <b>Partners:</b> {a["partners"]}</div></div>',
            unsafe_allow_html=True)

st.markdown(f"### 🌲 Recommended native species — {plan['zone']['label']}")
st.dataframe(pd.DataFrame(plan["zone"]["species"], columns=["Common name", "Latin name", "Use"]),
             hide_index=True, use_container_width=True)
st.caption("Use local seed sources and nurseries. Confirm species choice with a forester for the exact site, "
           "elevation and soil.")

st.download_button("⬇️ Download plan (Markdown)", rec.to_markdown(plan, f"TerraMind AI — Restoration plan: {title}"),
                   "terramind_restoration_plan.md", "text/markdown")
