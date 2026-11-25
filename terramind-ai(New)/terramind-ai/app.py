"""TerraMind AI — գործարկում․  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="TerraMind AI", page_icon="🌍", layout="wide")

PAGES = {
    "overview": st.Page("views/overview.py", title="Ընդհանուր ակնարկ", icon="🌍", default=True),
    "map": st.Page("views/map.py", title="Ինտերակտիվ քարտեզ", icon="🗺️"),
    "image": st.Page("views/image_analysis.py", title="Պատկերի վերլուծություն", icon="🛰️"),
    "dashboard": st.Page("views/dashboard.py", title="Վահանակ", icon="📊"),
    "risk": st.Page("views/risk_assessment.py", title="Ռիսկերի գնահատում", icon="⚠️"),
    "restoration": st.Page("views/restoration.py", title="Վերականգնում", icon="🌱"),
}
st.session_state["_pages"] = PAGES
st.navigation(list(PAGES.values())).run()
