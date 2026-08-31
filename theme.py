"""
Design system for the Damp & Mould Risk Platform.

Visual language is drawn from building surveying and thermographic instrumentation:
cold-wall slates and blues, condensation teal, ochre and brick for elevated risk.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
INK = "#14202B"          # deep slate — cold surface
INK_SOFT = "#4A5C6A"     # secondary text
LINE = "#D4DDE3"         # hairline
SURFACE = "#F1F5F7"      # plaster
COLD = "#2E5C7E"         # cold wall blue
TEAL = "#0E7C86"         # condensation / primary accent
OCHRE = "#C4762B"        # elevated risk
BRICK = "#A8332A"        # severe risk
MOSS = "#3D7A55"         # safe

RISK_COLORS = {
    "LOW": MOSS,
    "MODERATE": "#B8942F",
    "HIGH": OCHRE,
    "SEVERE": BRICK,
}

SERIES = [COLD, TEAL, OCHRE, BRICK, MOSS, "#6B7F92"]


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Sans+3:wght@400;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,300,0,0&display=swap');

/* Outline icons — FILL 0 keeps every glyph a stroke, never a solid shape */
.material-symbols-outlined {{
    font-family: 'Material Symbols Outlined';
    font-variation-settings: 'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24;
    font-weight: normal;
    font-style: normal;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
    vertical-align: middle;
}}

html, body, [class*="css"] {{
    font-family: 'Source Sans 3', system-ui, sans-serif;
    color: {INK};
}}

h1, h2, h3, h4 {{
    font-family: 'Archivo', system-ui, sans-serif !important;
    letter-spacing: -0.015em;
    color: {INK} !important;
}}

h1 {{ font-weight: 700 !important; font-size: 2.1rem !important; }}
h2 {{ font-weight: 600 !important; font-size: 1.35rem !important; margin-top: 1.6rem !important; }}
h3 {{ font-weight: 600 !important; font-size: 1.08rem !important; }}

.block-container {{ padding-top: 2.4rem; max-width: 1180px; }}

/* Page masthead ---------------------------------------------------------- */
.masthead {{
    border-left: 3px solid {TEAL};
    padding: 0.1rem 0 0.1rem 1rem;
    margin-bottom: 1.6rem;
}}
.masthead .title {{
    font-family: 'Archivo', sans-serif;
    font-weight: 700;
    font-size: 2.0rem;
    line-height: 1.15;
    color: {INK};
    display: flex;
    align-items: center;
    gap: 0.6rem;
}}
.masthead .title .material-symbols-outlined {{
    font-size: 1.9rem;
    color: {TEAL};
    font-variation-settings: 'FILL' 0, 'wght' 250, 'GRAD' 0, 'opsz' 24;
}}
.masthead .subtitle {{
    font-size: 1.0rem;
    color: {INK_SOFT};
    margin-top: 0.3rem;
    max-width: 62ch;
}}

/* Instrument readouts ---------------------------------------------------- */
.readout-row {{ display: flex; gap: 0.9rem; flex-wrap: wrap; margin: 0.4rem 0 1.4rem; }}
.readout {{
    flex: 1 1 150px;
    background: {SURFACE};
    border-top: 2px solid {COLD};
    padding: 0.75rem 0.9rem 0.8rem;
}}
.readout .label {{
    font-size: 0.78rem;
    color: {INK_SOFT};
    line-height: 1.25;
}}
.readout .value {{
    font-family: 'Archivo', sans-serif;
    font-weight: 700;
    font-size: 1.7rem;
    line-height: 1.1;
    margin-top: 0.25rem;
    color: {INK};
}}
.readout .unit {{ font-size: 0.95rem; font-weight: 500; color: {INK_SOFT}; }}

/* Risk meter — signature element ----------------------------------------- */
.meter-wrap {{ margin: 0.5rem 0 1.6rem; }}
.meter-head {{
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 0.45rem;
}}
.meter-head .caption {{ font-size: 0.85rem; color: {INK_SOFT}; }}
.meter-head .verdict {{
    font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.05rem;
}}
.meter-track {{
    position: relative; height: 15px; border-radius: 2px;
    background: linear-gradient(90deg,
        {MOSS} 0%, {MOSS} 5%,
        #B8942F 20%, {OCHRE} 50%, {BRICK} 85%, #7E241D 100%);
}}
.meter-needle {{
    position: absolute; top: -5px; width: 3px; height: 25px;
    background: {INK}; border-radius: 1px;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.9);
}}
.meter-scale {{
    display: flex; justify-content: space-between;
    font-size: 0.72rem; color: {INK_SOFT}; margin-top: 0.35rem;
}}

/* Notes ------------------------------------------------------------------ */
.note {{
    border-left: 3px solid {COLD};
    background: {SURFACE};
    padding: 0.8rem 1rem;
    margin: 0.9rem 0;
    font-size: 0.94rem;
    line-height: 1.55;
}}
.note.warn {{ border-left-color: {OCHRE}; background: #FBF4EA; }}
.note.alert {{ border-left-color: {BRICK}; background: #FAEEEC; }}
.note.good {{ border-left-color: {MOSS}; background: #EDF5F0; }}
.note strong {{ color: {INK}; }}

/* Sidebar ---------------------------------------------------------------- */
section[data-testid="stSidebar"] {{
    background: {INK};
    border-right: none;
}}
section[data-testid="stSidebar"] * {{ color: #E6EDF2 !important; }}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: #FFFFFF !important;
    font-family: 'Archivo', sans-serif !important;
}}
.sidebar-brand {{
    font-family: 'Archivo', sans-serif; font-weight: 700;
    font-size: 1.12rem; line-height: 1.25; color: #FFFFFF;
    padding-bottom: 0.15rem;
}}
.sidebar-sub {{
    font-size: 0.83rem; color: #9DB2C0 !important;
    border-bottom: 1px solid #2B3B49; padding-bottom: 0.9rem; margin-bottom: 0.9rem;
}}
.sidebar-meta {{ font-size: 0.8rem; color: #9DB2C0 !important; line-height: 1.6; }}

/* Sidebar navigation — icon buttons behaving as nav items */
section[data-testid="stSidebar"] .stButton {{ margin-bottom: 0.05rem; }}
section[data-testid="stSidebar"] .stButton > button {{
    justify-content: flex-start;
    text-align: left;
    padding: 0.42rem 0.7rem;
    border-radius: 3px;
    border: none;
    background: transparent;
    color: #9DB2C0 !important;
    font-size: 0.93rem;
    font-weight: 400;
    transition: none;
}}
section[data-testid="stSidebar"] .stButton > button p {{
    color: inherit !important;
    font-size: 0.93rem;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: #1E2E3B;
    color: #FFFFFF !important;
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: #1E2E3B;
    color: #FFFFFF !important;
    box-shadow: inset 2px 0 0 {TEAL};
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] p {{
    color: #FFFFFF !important;
    font-weight: 600;
}}
section[data-testid="stSidebar"] .stButton > button:focus-visible {{
    outline: 2px solid {TEAL};
    outline-offset: 1px;
}}

/* Tables ----------------------------------------------------------------- */
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; }}

/* Sliders ---------------------------------------------------------------- */
[data-testid="stSlider"] label {{ font-size: 0.86rem !important; color: {INK_SOFT} !important; }}

/* Dividers --------------------------------------------------------------- */
hr {{ border-color: {LINE}; margin: 1.8rem 0; }}
</style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Matplotlib theme
# ---------------------------------------------------------------------------
def apply_chart_theme():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": LINE,
        "axes.linewidth": 0.9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "600",
        "axes.titlecolor": INK,
        "axes.titlepad": 9,
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": LINE,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.7,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def icon(name, size=None, color=None):
    """Inline outline icon (Material Symbols, FILL 0)."""
    style = ""
    if size:
        style += f"font-size:{size};"
    if color:
        style += f"color:{color};"
    attr = f' style="{style}"' if style else ""
    return f'<span class="material-symbols-outlined"{attr}>{name}</span>'


def masthead(title, subtitle="", icon_name=None):
    glyph = icon(icon_name) if icon_name else ""
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="masthead"><div class="title">{glyph}<span>{title}</span></div>{sub}</div>',
        unsafe_allow_html=True,
    )


def readouts(items):
    """items: list of (label, value, unit)."""
    cells = "".join(
        f'<div class="readout"><div class="label">{lab}</div>'
        f'<div class="value">{val}<span class="unit">{unit}</span></div></div>'
        for lab, val, unit in items
    )
    st.markdown(f'<div class="readout-row">{cells}</div>', unsafe_allow_html=True)


def risk_meter(pct, caption="Share of monitored period at mould risk"):
    """Damp-meter style scale. pct is 0-100."""
    if pct < 5:
        band = "LOW"
    elif pct < 20:
        band = "MODERATE"
    elif pct < 50:
        band = "HIGH"
    else:
        band = "SEVERE"
    pos = max(0.6, min(99.4, pct))
    st.markdown(
        f"""
<div class="meter-wrap">
  <div class="meter-head">
    <span class="caption">{caption}</span>
    <span class="verdict" style="color:{RISK_COLORS[band]}">{band} &middot; {pct:.1f}%</span>
  </div>
  <div class="meter-track"><div class="meter-needle" style="left:{pos}%"></div></div>
  <div class="meter-scale"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
</div>
        """,
        unsafe_allow_html=True,
    )
    return band


def note(text, kind="plain"):
    cls = {"plain": "", "warn": " warn", "alert": " alert", "good": " good"}[kind]
    st.markdown(f'<div class="note{cls}">{text}</div>', unsafe_allow_html=True)
