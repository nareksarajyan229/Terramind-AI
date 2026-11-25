"""Streamlit-ի ընդհանուր ինտերֆեյս․ թեմա, բաղադրիչներ, պահված ռեսուրսներ և գրաֆիկներ։"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from . import armenia, db, gee, vision
from .config import settings
from .detector import Detector
from .risk import LEVEL_HY, level_for
from .segmentation import TORCH_OK, Segmenter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e4df"
ISSUE_COLORS = {k: armenia.CATEGORIES[k]["color"] for k in armenia.CATEGORIES}
LEVEL_ICONS = {"Low": "●", "Moderate": "▲", "High": "◆", "Critical": "■"}

CSS = """
<style>
:root { --tm-primary:#0f5e45; --tm-ink:#0b0b0b; --tm-ink2:#52514e; --tm-line:#e6e4df; }
html, body, [class*="css"] { font-family: "Noto Sans Armenian", "Segoe UI", "Arial", sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #0e2a20; }
[data-testid="stSidebar"] * { color: #e8efe9 !important; }
.tm-hero { background: linear-gradient(120deg,#0e2a20 0%,#0f5e45 60%,#2f7d5b 100%);
  border-radius: 14px; padding: 26px 30px; color: #fff; margin-bottom: 18px; }
.tm-hero h1 { color:#fff; font-size: 2.0rem; margin: 0 0 4px 0; }
.tm-hero p { color:#d7eadf; margin: 0; font-size: 1.02rem; }
.tm-card { background:#fff; border:1px solid var(--tm-line); border-radius:12px; padding:16px 18px; height:100%; }
.tm-kpi-label { color: var(--tm-ink2); font-size: .85rem; }
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
.tm-code { display:inline-block; min-width:18px; text-align:center; font-weight:700; border-radius:3px; margin-right:6px;
  padding:0 4px; color:#0b0b0b; font-size:.78rem; }
[data-testid="stFileUploaderDropzoneInstructions"] > div > span { display: none; }
[data-testid="stFileUploaderDropzoneInstructions"] > div::before { content: "Քաշեք ֆայլն այստեղ"; font-weight: 600; }
</style>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Armenian:wght@400;600;700&display=swap" rel="stylesheet">
"""


def setup() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    sidebar()


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="tm-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "") -> None:
    st.markdown(f'<div class="tm-card"><div class="tm-kpi-label">{label}</div>'
                f'<div class="tm-kpi-value">{value}</div><div class="tm-kpi-note">{note}</div></div>',
                unsafe_allow_html=True)


def level_name(level: str) -> str:
    return LEVEL_HY.get(level, level)


def level_badge(level: str, color: str, score: float | None = None) -> str:
    txt = f"{LEVEL_ICONS.get(level, '●')} {level_name(level)}" + (f" · {score:.0f}/100" if score is not None else "")
    return f'<span class="tm-badge" style="background:{color}33;border-color:{color}">{txt}</span>'


def legend_html(items: list[tuple[str, str]]) -> str:
    return '<div class="tm-legend">' + "".join(
        f'<span><span class="sw" style="background:{c}"></span>{l}</span>' for l, c in items) + "</div>"


def class_legend() -> str:
    return legend_html([(lbl, "#%02x%02x%02x" % col) for _, lbl, col in vision.CLASSES])


def box_legend() -> str:
    """Explains the letter codes drawn on detection boxes."""
    items = "".join(
        f'<span><span class="tm-code" style="background:rgb{vision.ISSUES[k]["color"]}">{code}</span>'
        f'{vision.ISSUES[k]["label"]}</span>' for k, code in vision.ISSUE_CODES.items())
    return f'<div class="tm-legend">{items}</div>'


def link(target: str, label: str, icon: str) -> None:
    st.page_link(f"views/{target}.py", label=label, icon=icon)


# --------------------------------------------------------------------------------
# Կողային վահանակ և ռեսուրսներ
# --------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Սեգմենտացիայի մոդելի բեռնում…")
def get_segmenter() -> Segmenter:
    return Segmenter(settings.seg_weights)


@st.cache_resource(show_spinner="YOLO դետեկտորի բեռնում…")
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
        st.caption("Բնապահպանական արհեստական բանականություն Հայաստանի համար")
        st.divider()
        seg, det = get_segmenter(), get_detector()
        ok_gee = gee_ready()
        store = get_store()
        rows = [
            ("OpenCV սպեկտրալ համակարգ", True, "միշտ միացված է"),
            ("PyTorch U-Net", seg.ready, (seg.reason or "մոդելը բեռնված է") if TORCH_OK else "PyTorch-ը տեղադրված չէ"),
            ("YOLO դետեկտոր", det.ready, det.reason or "մոդելը բեռնված է"),
            ("Google Earth Engine", ok_gee, gee.status()[1] or "միացված է"),
            (f"Պահոց՝ {store.name}", True, db.store_error() or "միացված է"),
        ]
        st.markdown("**Համակարգի կարգավիճակ**")
        for name, ok, why in rows:
            st.markdown(f"{'🟢' if ok else '⚪'} {name}  \n<small>{why}</small>", unsafe_allow_html=True)
        st.divider()
        st.caption("Բաց կոդ · Sentinel-2 © ESA Copernicus · Hansen GFC © UMD · CHIRPS © UCSB")


# --------------------------------------------------------------------------------
# Գրաֆիկներ (Plotly, ներկառուցված հուշումներով)
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
        title={"text": f"Ընդհանուր ռիսկ — {level_name(level)}", "font": {"size": 15, "color": INK_2}},
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
        hovertemplate="%{y}<br>Վտանգավորություն՝ %{x:.0f}/100<extra></extra>"))
    fig.update_xaxes(range=[0, 110], title="Վտանգավորություն (0–100)")
    fig.update_layout(title=title)
    return _layout(fig, 300)


def class_bars(fractions: dict) -> go.Figure:
    keys = [k for k in vision.CLASS_KEYS if fractions.get(k, 0) > 0.0005]
    cols = ["#%02x%02x%02x" % vision.CLASS_COLORS[k] for k in keys]
    fig = go.Figure(go.Bar(
        x=[vision.CLASS_LABELS[k] for k in keys], y=[fractions[k] * 100 for k in keys],
        marker=dict(color=cols, cornerradius=4, line=dict(color="#52514e", width=0.5)),
        hovertemplate="%{x}<br>Պատկերի %{y:.1f}%-ը<extra></extra>"))
    fig.update_yaxes(title="Տարածքի %", ticksuffix="%")
    return _layout(fig, 300)


def radar(components: dict) -> go.Figure:
    keys = list(components)
    labels = [vision.ISSUES[k]["label"].split(" /")[0] for k in keys]
    vals = [components[k] for k in keys]
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], fill="toself",
                                    line=dict(color="#2a78d6", width=2), fillcolor="rgba(42,120,214,.18)",
                                    hovertemplate="%{theta}՝ %{r:.0f}/100<extra></extra>"))
    fig.update_layout(polar=dict(bgcolor=SURFACE, radialaxis=dict(range=[0, 100], gridcolor=GRID, tickfont_color=INK_2),
                                 angularaxis=dict(gridcolor=GRID, tickfont_color=INK)))
    return _layout(fig, 340)


def issue_card(key: str, info: dict) -> None:
    color = ISSUE_COLORS.get(key, "#999")
    lvl, lc = level_for(info["severity"])
    extra = []
    if info.get("area_ha") is not None:
        extra.append(f"{info['area_ha']:,.1f} հա")
    if info.get("area_fraction") is not None:
        extra.append(f"պատկերի {info['area_fraction'] * 100:.1f}%-ը")
    if info.get("n_regions"):
        extra.append(f"{info['n_regions']} տարածք")
    if info.get("evidence"):
        extra.append(info["evidence"])
    st.markdown(
        f'<div class="tm-issue" style="border-left-color:{color}"><h4>{vision.ISSUES[key]["label"]} '
        f'{level_badge(lvl, lc, info["severity"])}</h4><div class="tm-muted">{" · ".join(extra)}</div></div>',
        unsafe_allow_html=True)
