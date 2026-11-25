"""Վերականգնման պլանավորում։"""
import pandas as pd
import streamlit as st

from terramind import recommendations as rec
from terramind import ui, vision

ui.setup()
ui.hero("Վերականգնման պլանավորում", "Կոնկրետ, փուլային գործողություններ հայտնաբերված խնդիրների համար՝ "
        "հայաստանյան տեղական ծառատեսակներով, գործընկերներով և հետևելու ցուցանիշներով։")

MODES = ["Ընթացիկ վերլուծություն", "Սեփական սցենար"]
o = st.session_state.get("outcome")
mode = st.radio("Ծրագիր՝", MODES, horizontal=True, index=0 if o is not None else 1, disabled=o is None)

if mode == MODES[0] and o is not None:
    plan, title = o.plan, o.title
    st.caption(f"Հիմքում՝ **{o.title}**" + (f" · {o.marz}" if o.marz else "") +
               f" · ռիսկ {o.risk.overall:.0f}/100 ({ui.level_name(o.risk.level)})")
else:
    st.markdown("Նշեք յուրաքանչյուր խնդրի վտանգավորությունը՝ առանց վերլուծության ծրագիր կազմելու համար։")
    cols = st.columns(3)
    issues = {}
    for i, (k, meta) in enumerate(vision.ISSUES.items()):
        issues[k] = {"severity": cols[i % 3].slider(meta["label"], 0, 100, 0, 5)}
    zone = st.selectbox("Էկոգոտի", list(rec.SPECIES_BY_ZONE), format_func=lambda z: rec.SPECIES_BY_ZONE[z]["label"])
    marz = {"humid_forest": "Տավուշ", "arid_south": "Արարատ", "highland": "Գեղարքունիք"}[zone]
    plan, title = rec.recommend(issues, marz=marz), "Սեփական սցենար"

st.info(plan["summary"])
if not plan["actions"]:
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    n_now = sum(a["priority"] == rec.PRIORITIES[0] for a in plan["actions"])
    ui.kpi("Գործողություններ", str(len(plan["actions"])), f"որից անհապաղ՝ {n_now}")
with c2:
    ui.kpi("Լուծվող խնդիրներ", str(len({a['issue'] for a in plan['actions']})), "")
with c3:
    ui.kpi("Էկոգոտի", plan["zone"]["label"].split(" (")[0], plan["zone"]["label"].split("(")[-1].rstrip(")"))

st.write("")
for phase, icon in zip(rec.PHASES, ["⚡", "🛠️", "🌳"]):
    items = [a for a in plan["actions"] if a["phase"] == phase]
    if not items:
        continue
    st.markdown(f"### {icon} {phase}")
    for a in items:
        color = ui.ISSUE_COLORS.get(a["issue"], "#999")
        st.markdown(
            f'<div class="tm-issue" style="border-left-color:{color}"><h4>{a["title"]} '
            f'<span class="tm-muted">· {a["priority"]} · արժեք {a["cost"]} · {vision.ISSUES[a["issue"]]["label"]}</span></h4>'
            f'<div>{a["detail"]}</div><div class="tm-muted" style="margin-top:6px">'
            f'<b>Ցուցանիշ․</b> {a["kpi"]} &nbsp;·&nbsp; <b>Գործընկերներ․</b> {a["partners"]}</div></div>',
            unsafe_allow_html=True)

st.markdown(f"### 🌲 Առաջարկվող տեղական տեսակներ — {plan['zone']['label']}")
st.dataframe(pd.DataFrame(plan["zone"]["species"], columns=["Անվանում", "Լատիներեն անվանում", "Կիրառում"]),
             hide_index=True, use_container_width=True)
st.caption("Օգտագործեք տեղական սերմեր և տնկարաններ։ Կոնկրետ տարածքի, բարձրության և հողի համար տեսակների "
           "ընտրությունը համաձայնեցրեք անտառագետի հետ։")

st.download_button("⬇️ Ներբեռնել ծրագիրը (Markdown)",
                   rec.to_markdown(plan, f"TerraMind AI — Վերականգնման ծրագիր․ {title}"),
                   "terramind_restoration_plan.md", "text/markdown")
