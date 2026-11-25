"""Բնապահպանական ռիսկերի գնահատում։"""
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from terramind import armenia, risk, services, ui, vision

ui.setup()
ui.hero("Բնապահպանական ռիսկերի գնահատում", "Որքա՞ն լուրջ է իրավիճակը, ինչն է դրա պատճառը և որտեղ է "
        "այն ամենավատը Հայաստանում։")

o = st.session_state.get("outcome")
st.markdown("### Ընթացիկ վերլուծություն")
if o is None:
    st.info("Այս աշխատաշրջանում դեռ վերլուծություն չկա։ Ընտրեք թեժ կետ՝ տեղեկատու տվյալներով գնահատելու "
            "համար, կամ կատարեք վերլուծություն «Քարտեզ» կամ «Պատկերի վերլուծություն» էջում։")
    hid = st.selectbox("Գնահատել փաստագրված թեժ կետը", [h.id for h in armenia.HOTSPOTS],
                       format_func=lambda i: armenia.get_hotspot(i).name)
    if st.button("Գնահատել", type="primary"):
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
            st.caption(f"Առանց տեղանքի համատեքստի՝ {o.risk_image_only.overall:.0f}/100 "
                       f"({ui.level_name(o.risk_image_only.level)})")
    with b:
        st.plotly_chart(ui.radar(o.risk.components), use_container_width=True)
    with c:
        st.markdown("**Ռիսկի հիմնական գործոններ**")
        if o.risk.drivers:
            for d in o.risk.drivers:
                st.markdown(f"- {d}")
        else:
            st.markdown("- Ոչ մի բաղադրիչ 25/100-ից բարձր չէ")
        if o.risk.context:
            st.markdown("**Մոտակա փաստագրված թեժ կետեր**")
            for ctx in o.risk.context:
                st.markdown(f"- {ctx['hotspot']} · {ctx['distance_km']} կմ · վտանգավորություն {ctx['severity']}/5")

    comp = pd.DataFrame([{"Խնդիր": vision.ISSUES[k]["label"], "Գնահատական": v,
                          "Մակարդակ": ui.level_name(risk.level_for(v)[0]), "Կշիռ": risk.WEIGHTS[k],
                          "Հիմնավորում": o.issues.get(k, {}).get("evidence", "")}
                         for k, v in sorted(o.risk.components.items(), key=lambda kv: -kv[1])])
    st.dataframe(comp, hide_index=True, use_container_width=True,
                 column_config={"Գնահատական": st.column_config.ProgressColumn("Գնահատական", min_value=0,
                                                                              max_value=100, format="%.0f")})
    with st.expander("Ինչպես է հաշվարկվում գնահատականը"):
        st.markdown(
            "- Յուրաքանչյուր խնդիր ստանում է **0–100 վտանգավորություն** ԱԲ վերլուծությունից (տուժած տարածքի "
            "չափն ու բնույթը կամ Earth Engine-ի ինդեքսները)։\n"
            "- Վտանգավորությունը բարձրանում է **40 կմ շառավղում գտնվող փաստագրված թեժ կետի** գնահատականի "
            "ուղղությամբ՝ նվազելով հեռավորության հետ։\n"
            "- Ընդհանուր գնահատականը **կշռված միջին է**․ ամենալուրջ խնդիրը հաշվվում է ամբողջությամբ, "
            "երկրորդը՝ կեսով, երրորդը՝ քառորդով և այլն, որպեսզի մեկ լուրջ խնդիրը չ«նոսրացվի» մի քանի "
            "փոքրերով։\n"
            "- Մակարդակներ․ **Ցածր** < 25 ≤ **Չափավոր** < 50 ≤ **Բարձր** < 75 ≤ **Կրիտիկական**։")

st.divider()
st.markdown("### Թեժ կետերի ռիսկերի ազգային մատրից")
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
        hr.append(f"Հավանականություն {like} × ազդեցություն {impact} = {score}<br>" +
                  ("<br>".join("• " + n for n in cell["name"]) if len(cell) else "Թեժ կետեր չկան"))
    z.append(zr); txt.append(tr); hov.append(hr)

colorscale = [[0, "#0ca30c"], [0.25, "#0ca30c"], [0.25, "#fab219"], [0.5, "#fab219"],
              [0.5, "#ec835a"], [0.75, "#ec835a"], [0.75, "#d03b3b"], [1, "#d03b3b"]]
fig = go.Figure(go.Heatmap(z=z, x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5], text=txt, texttemplate="%{text}",
                           textfont=dict(size=18, color="#0b0b0b"), customdata=hov,
                           hovertemplate="%{customdata}<extra></extra>", colorscale=colorscale, zmin=0, zmax=100,
                           showscale=False, xgap=2, ygap=2, opacity=0.75))
fig.update_xaxes(title="Հավանականություն (միտում + վտանգավորություն)", dtick=1)
fig.update_yaxes(title="Ազդեցություն (վտանգավորություն)", dtick=1)
m1, m2 = st.columns([1, 1.25])
with m1:
    st.plotly_chart(ui._layout(fig, 400), use_container_width=True)
    st.markdown(ui.legend_html([(f"{ui.LEVEL_ICONS[n]} {ui.level_name(n)}", c) for _, n, c in risk.LEVELS]),
                unsafe_allow_html=True)
    st.caption("Թվերը ցույց են տալիս յուրաքանչյուր վանդակում գտնվող թեժ կետերի քանակը։ Անունները տեսնելու "
               "համար պահեք մկնիկը վանդակի վրա։")
with m2:
    show = mx[["name", "category", "marz", "trend", "score"]].rename(
        columns={"name": "Թեժ կետ", "category": "Կատեգորիա", "marz": "Մարզ", "trend": "Միտում", "score": "Ռիսկ"})
    st.dataframe(show, hide_index=True, use_container_width=True, height=400,
                 column_config={"Ռիսկ": st.column_config.ProgressColumn("Ռիսկ", min_value=0, max_value=100, format="%.0f")})
st.caption("Թեժ կետերի գնահատականները ինդիկատիվ են և հիմնված են նշված աղբյուրների վրա․ նախատեսված են միայն "
           "առաջնահերթությունների որոշման համար։")
