"""Ընդհանուր ակնարկ։"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from terramind import armenia, ui

ui.setup()
ui.hero("TerraMind AI",
        "Արհեստական բանականությամբ բնապահպանական մշտադիտարկում Հայաստանի համար․ արբանյակային "
        "պատկերներից հայտնաբերեք ջրի աղտոտումը, երաշտը, անտառահատումը և հողի դեգրադացումը, "
        "գնահատեք ռիսկերը և պլանավորեք վերականգնումը։")

st.markdown("#### Հայաստանը թվերով")
cols = st.columns(4)
for c, ind in zip(cols, armenia.NATIONAL_INDICATORS[:4]):
    with c:
        ui.kpi(ind["label"], ind["value"], f'{ind["note"]} · <a href="{ind["source"]}" target="_blank">աղբյուր</a>')

st.write("")
left, right = st.columns([1.15, 1])
with left:
    st.markdown("#### Ինչ կարող եք անել")
    a, b = st.columns(2)
    with a:
        ui.link("map", "**Ինտերակտիվ քարտեզ**", "🗺️")
        st.caption("Ուսումնասիրեք փաստագրված թեժ կետերը, Sentinel-2 շերտերը և վերլուծեք Հայաստանի ցանկացած կետ։")
        ui.link("dashboard", "**Վերլուծությունների վահանակ**", "📊")
        st.caption("Վիճակագրություն բոլոր պահպանված վերլուծությունների, մարզերի և խնդիրների վերաբերյալ։")
        ui.link("restoration", "**Վերականգնման պլանավորում**", "🌱")
        st.caption("Փուլային գործողությունների ծրագիր՝ տեղական ծառատեսակներով և ցուցանիշներով։")
    with b:
        ui.link("image_analysis", "**Պատկերի վերլուծություն**", "🛰️")
        st.caption("Վերբեռնեք արբանյակային կամ դրոնային պատկեր՝ հողածածկույթի դասակարգման և խնդիրների հայտնաբերման համար։")
        ui.link("risk_assessment", "**Ռիսկերի գնահատում**", "⚠️")
        st.caption("Ռիսկի ցուցիչ, բաղադրիչների վերլուծություն և թեժ կետերի ռիսկերի մատրից։")

    st.markdown("#### Ինչպես է աշխատում արհեստական բանականությունը")
    st.markdown(
        "1. **Մուտքային տվյալներ** — վերբեռնված պատկեր (JPG/PNG կամ Sentinel-2/Landsat GeoTIFF) կամ "
        "քարտեզի կետ՝ Google Earth Engine-ից։\n"
        "2. **Հողածածկույթ** — PyTorch U-Net-ը բաժանում է պատկերը 8 դասի։ Մարզված մոդելի բացակայության "
        "դեպքում OpenCV-ն օգտագործում է սպեկտրալ ինդեքսներ (NDVI, NDWI, NDCI, BSI, NBR)։\n"
        "3. **Հայտնաբերում** — YOLO-ն գտնում է առանձին օբյեկտներ (բացատներ, պոչամբարներ, առվակներ, "
        "ջրիմուռների ծաղկում), իսկ կապակցված տարածքների վերլուծությունը նշում է տուժած հատվածները։\n"
        "4. **Ռիսկ** — խնդիրների վտանգավորությունը կշռվում և համադրվում է Հայաստանի փաստագրված թեժ "
        "կետերի մոտիկության հետ։\n"
        "5. **Վերականգնում** — կանոնները յուրաքանչյուր խնդրի համար առաջարկում են գործողություններ, "
        "գործընկերներ, ցուցանիշներ և տեղական ծառատեսակներ՝ ըստ էկոգոտու։")

with right:
    st.markdown("#### Թեժ կետերն ըստ կատեգորիայի")
    df = pd.DataFrame(armenia.hotspots_as_records())
    counts = df.groupby("category").agg(n=("id", "count"), sev=("severity", "mean")).reset_index()
    counts["label"] = counts["category"].map(lambda c: armenia.CATEGORIES[c]["label"])
    counts = counts.sort_values("n")
    fig = go.Figure(go.Bar(
        x=counts["n"], y=counts["label"], orientation="h",
        marker=dict(color=[armenia.CATEGORIES[c]["color"] for c in counts["category"]], cornerradius=4,
                    line=dict(color=ui.SURFACE, width=2)),
        customdata=counts["sev"].round(1), text=counts["n"], textposition="outside",
        hovertemplate="%{y}<br>%{x} թեժ կետ · միջին վտանգավորություն՝ %{customdata}/5<extra></extra>"))
    fig.update_xaxes(title="Թեժ կետերի քանակ", dtick=1)
    st.plotly_chart(ui._layout(fig, 300), use_container_width=True)

    st.markdown("#### Ամենավտանգավոր թեժ կետերը")
    top = df.sort_values(["severity", "name"], ascending=[False, True]).head(6)
    top["trend"] = top["trend"].map(armenia.TRENDS)
    st.dataframe(
        top[["name", "marz", "severity", "trend"]].rename(columns={
            "name": "Թեժ կետ", "marz": "Մարզ", "severity": "Վտանգավորություն (1–5)", "trend": "Միտում"}),
        hide_index=True, use_container_width=True)

st.divider()
st.caption("Վտանգավորության գնահատականները ինդիկատիվ են և նախատեսված են միայն առաջնահերթությունների համար։ "
           "Ցուցանիշները հղում են իրենց աղբյուրներին։ Արհեստական բանականության հայտնաբերումները պետք է "
           "հաստատվեն տեղում ստուգմամբ՝ նախքան որևէ որոշում կայացնելը։")
