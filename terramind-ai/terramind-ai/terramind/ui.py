"""Shared Streamlit UI: theme, layout pieces, cached resources and chart helpers."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from . import armenia, db, gee, vision
from .config import settings
from .detector import Detector
from .segmentation import TORCH_OK, Segmenter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e4df"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ISSUE_COLORS = {k: armenia.CATEGORIES[k]["color"] for k in armenia.CATEGORIES}

CSS = """
<style>
:root { --tm-primary:#0f5e45; --tm-ink:#0b0b0b; --tm-ink2:#52514e; --tm-line:#e6e4df; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #0e2a20; }
[data-testid="stSidebar"] * { color: #e8efe9 !important; }
[data-testid="stSidebar"] .stAlert * { color: inherit !important; }
.tm-hero { background: linear-gradient(120deg,#0e2a20 0%,#0f5e45 60%,#2f7d5b 100%);
  border-radius: 14px; padding: 26px 30px; color: #fff; margin-bottom: 18px; }
.tm-hero h1 { color:#fff; font-size: 2.0rem; margin: 0 0 4px 0; letter-spacing: -.5px; }
.tm-hero p { color:#d7eadf; margin: 0; font-size: 1.02rem; }
.tm-card { background:#fff; border:1px solid var(--tm-line); border-radius:12px; padding:16px 18px; height:100%; }
.tm-kpi-label { color: var(--tm-ink2); font-size: .82rem; text-transform: uppercase; letter-spacing: .4px; }
.tm-kpi-value { color: var(--tm-ink); font-size: 1.7rem; font-weight: 700; line-height: 1.2; margin-top: 2px; }
.tm-kpi-note { color: var(--tm-ink2); font-size: .8rem; margin-top: 4px; }
.tm-badge { display:inline-block; padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: .82rem;
  color: #0b0b0b; border: 1px solid rgba(0,0,0,.12); }
.tm-issue { border-left: 4px solid #999; padding: 10px 14px; background:#fff; border-radius: 8px;
  border-top:1px solid var(--tm-line); border-right:1px solid var(--tm-line); border-bottom:1px solid var(--tm-line);
  margin-bottom: 10px; }
.tm-issue h4 { margin: 0 0 2px 0; font-size: 1rem; }
.tm-muted { color: var(--tm-ink2); font-size: .86rem; }
.tm-legend { display:flex; flex-wrap:wrap; gap: 8px 16px; font-size:.85rem; color: var(--tm-ink2); }
.tm-legend span.sw { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
</style>
"""


def setup(title: str, icon: str = "🌍") -> None:
    st.set_page_config(page_title=f"{title} · TerraMind AI", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    sidebar()


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="tm-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "") -> None:
    st.markdown(f'<div class="tm-card"><div class="tm-kpi-label">{label}</div>'
                f'<div class="tm-kpi-value">{value}</div><div class="tm-kpi-note">{note}</div></div>',
                unsafe_allow_html=True)


def level_badge(level: str, color: str, score: float | None = None) -> str:
    icon = {"Low": "●", "Moderate": "▲", "High": "◆", "Critical": "■"}.get(level, "●")
    txt = f"{icon} {level}" + (f" · {score:.0f}/100" if score is not None else "")
    return f'<span class="tm-badge" style="background:{color}33;border-color:{color}">{txt}</span>'


def legend_html(items: list[tuple[str, str]]) -> str:
    return '<div class="tm-legend">' + "".join(
        f'<span><span class="sw" style="background:{c}"></span>{l}</span>' for l, c in items) + "</div>"


def class_legend() -> str:
    return legend_html([(lbl, "#%02x%02x%02x" % col) for _, lbl, col in vision.CLASSES])


# --------------------------------------------------------------------------------
# Sidebar & resources
# --------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading segmentation model…")
def get_segmenter() -> Segmenter:
    return Segmenter(settings.seg_weights)


@st.cache_resource(show_spinner="Loading YOLO detector…")
def get_detector() -> Detector:
    return Detector(settings.yolo_weights)


@st.cache_resource
def get_store():
    return db.get_store()


@st.cache_resource
def gee_ready() -> bool:
    return gee.init()


def sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🌍 TerraMind AI")
        st.caption("Environmental intelligence for Armenia")
        st.divider()
        seg, det = get_segmenter(), get_detector()
        ok_gee = gee_ready()
        store = get_store()
        rows = [
            ("OpenCV spectral engine", True, "always on"),
            ("PyTorch U-Net", seg.ready, seg.reason or "weights loaded" if TORCH_OK else "PyTorch not installed"),
            ("YOLO detector", det.ready, det.reason or "weights loaded"),
            ("Google Earth Engine", ok_gee, gee.status()[1] or "connected"),
            (f"Storage: {store.name}", True, db.store_error() or "connected"),
        ]
        st.markdown("**System status**")
        for name, ok, why in rows:
            st.markdown(f"{'🟢' if ok else '⚪'} {name}  \n<small>{why}</small>", unsafe_allow_html=True)
        st.divider()
        st.caption("Open source · Sentinel-2 © ESA Copernicus · Hansen GFC © UMD · CHIRPS © UCSB")


# --------------------------------------------------------------------------------
# Charts (Plotly; built-in hover tooltips)
# --------------------------------------------------------------------------------

def _layout(fig: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=30, b=8), paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, font=dict(color=INK, size=13), showlegend=legend,
                      hoverlabel=dict(bgcolor="#fff", font_color=INK, bordercolor=GRID))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont_color=INK_2)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont_color=INK_2)
    return fig


def gauge(score: float, level: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "/100", "font": {"size": 34, "color": INK}},
        title={"text": f"Overall risk — {level}", "font": {"size": 15, "color": INK_2}},
        gauge={"axis": {"range": [0, 100], "tickcolor": INK_2}, "bar": {"color": color, "thickness": 0.3},
               "bgcolor": SURFACE, "borderwidth": 0,
               "steps": [{"range": [0, 25], "color": "#0ca30c22"}, {"range": [25, 50], "color": "#fab21922"},
                         {"range": [50, 75], "color": "#ec835a22"}, {"range": [75, 100], "color": "#d03b3b22"}]}))
    return _layout(fig, 260)


def issue_bars(issues: dict, title: str = "") -> go.Figure:
    keys = sorted(issues, key=lambda k: issues[k]["severity"])
    fig = go.Figure(go.Bar(
        x=[issues[k]["severity"] for k in keys], y=[vision.ISSUES[k]["label"] for k in keys], orientation="h",
        marker=dict(color=[ISSUE_COLORS[k] for k in keys], cornerradius=4, line=dict(color=SURFACE, width=2)),
        text=[f"{issues[k]['severity']:.0f}" for k in keys], textposition="outside", textfont_color=INK,
        hovertemplate="%{y}<br>Severity %{x:.0f}/100<extra></extra>"))
    fig.update_xaxes(range=[0, 110], title="Severity (0–100)")
    fig.update_layout(title=title)
    return _layout(fig, 300)


def class_bars(fractions: dict) -> go.Figure:
    keys = [k for k in vision.CLASS_KEYS if fractions.get(k, 0) > 0.0005]
    cols = ["#%02x%02x%02x" % vision.CLASS_COLORS[k] for k in keys]
    fig = go.Figure(go.Bar(
        x=[vision.CLASS_LABELS[k] for k in keys], y=[fractions[k] * 100 for k in keys],
        marker=dict(color=cols, cornerradius=4, line=dict(color="#52514e", width=0.5)),
        hovertemplate="%{x}<br>%{y:.1f}% of image<extra></extra>"))
    fig.update_yaxes(title="% of area", ticksuffix="%")
    return _layout(fig, 300)


def radar(components: dict) -> go.Figure:
    keys = list(components)
    labels = [vision.ISSUES[k]["label"].split(" /")[0] for k in keys]
    vals = [components[k] for k in keys]
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], fill="toself",
                                    line=dict(color="#2a78d6", width=2), fillcolor="rgba(42,120,214,.18)",
                                    hovertemplate="%{theta}: %{r:.0f}/100<extra></extra>"))
    fig.update_layout(polar=dict(bgcolor=SURFACE, radialaxis=dict(range=[0, 100], gridcolor=GRID, tickfont_color=INK_2),
                                 angularaxis=dict(gridcolor=GRID, tickfont_color=INK)))
    return _layout(fig, 340)


def issue_card(key: str, info: dict) -> None:
    color = ISSUE_COLORS.get(key, "#999")
    from .risk import level_for

    lvl, lc = level_for(info["severity"])
    extra = []
    if info.get("area_ha") is not None:
        extra.append(f"{info['area_ha']:,.1f} ha")
    if info.get("area_fraction") is not None:
        extra.append(f"{info['area_fraction'] * 100:.1f}% of image")
    if info.get("n_regions"):
        extra.append(f"{info['n_regions']} region(s)")
    if info.get("evidence"):
        extra.append(info["evidence"])
    st.markdown(
        f'<div class="tm-issue" style="border-left-color:{color}"><h4>{vision.ISSUES[key]["label"]} '
        f'{level_badge(lvl, lc, info["severity"])}</h4><div class="tm-muted">{" · ".join(extra)}</div></div>',
        unsafe_allow_html=True)
