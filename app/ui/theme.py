"""Design system for the Clean.ai UI.

One place for every token and shared visual component. The categorical
chart order is fixed and CVD-validated on white - keep the order.
"""

# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------

PRIMARY = "#4F46E5"
PRIMARY_DARK = "#4338CA"
VIOLET = "#7C3AED"
SECONDARY = "#0891B2"
ACCENT = "#DB2777"
SUCCESS = "#059669"
WARNING = "#B45309"
DANGER = "#DC2626"

BACKGROUND = "#F6F7FB"
SURFACE = "#FFFFFF"
SIDEBAR_BG = "#101322"
TEXT = "#0F172A"
TEXT_MUTED = "#5B6478"
BORDER = "#E7EAF3"

GRADIENT = f"linear-gradient(135deg, {PRIMARY} 0%, {VIOLET} 100%)"

CATEGORICAL = [PRIMARY, ACCENT, SECONDARY, "#D97706", "#047857"]
BEFORE_COLOR = "#94A3B8"
AFTER_COLOR = PRIMARY

FONT_STACK = "'Inter', -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"

# ---------------------------------------------------------------------------
# global stylesheet
# ---------------------------------------------------------------------------

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {{
    font-family: {FONT_STACK};
    background:
        radial-gradient(1100px 420px at 18% -8%, rgba(99,102,241,0.08), transparent 70%),
        radial-gradient(900px 380px at 95% 0%, rgba(219,39,119,0.045), transparent 70%),
        {BACKGROUND};
    color: {TEXT};
}}
.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1180px; }}

h1, h2, h3, h4 {{ font-family: {FONT_STACK}; color: {TEXT}; letter-spacing: -0.015em; }}
[data-testid="stMarkdownContainer"] p {{ line-height: 1.55; }}
[data-testid="stCaptionContainer"] {{ color: {TEXT_MUTED}; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG};
    border-right: none;
}}
section[data-testid="stSidebar"] * {{ color: #E8EAF6 !important; }}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] small {{ color: #8A91B4 !important; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.08); }}
[data-testid="stSidebarNav"] a, [data-testid="stNavSectionHeader"] {{
    border-radius: 10px;
}}
[data-testid="stSidebarNav"] a:hover {{ background: rgba(255,255,255,0.07) !important; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
    background: rgba(99,102,241,0.28) !important;
}}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ font-weight: 600; }}
section[data-testid="stSidebar"] .stButton > button {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    color: #E8EAF6 !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,255,255,0.12);
    border-color: rgba(255,255,255,0.25);
}}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
    border-radius: 10px;
    font-weight: 600;
    font-family: {FONT_STACK};
    border: 1px solid {BORDER};
    box-shadow: 0 1px 2px rgba(16,24,40,0.05);
    transition: all .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(16,24,40,0.10);
}}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
    background: {GRADIENT} !important;
    border: none !important;
    color: #fff !important;
}}

/* ---------- cards: metrics, expanders ---------- */
div[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1rem 1.15rem;
    box-shadow: 0 1px 2px rgba(16,24,40,0.05);
}}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {{
    color: {TEXT_MUTED};
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    font-weight: 700;
    letter-spacing: -0.02em;
}}
div[data-testid="stExpander"] details {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}}

/* ---------- tabs ---------- */
button[data-baseweb="tab"] {{
    font-family: {FONT_STACK};
    font-weight: 600;
    color: {TEXT_MUTED};
}}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {PRIMARY}; }}
div[data-baseweb="tab-highlight"] {{ background-color: {PRIMARY}; }}

/* ---------- inputs ---------- */
div[data-baseweb="input"], div[data-baseweb="textarea"] {{
    border-radius: 10px;
}}
[data-testid="stFileUploaderDropzone"] {{
    background: {SURFACE};
    border: 1.5px dashed #C7CCE2;
    border-radius: 14px;
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: {PRIMARY}; }}

/* ---------- dataframes / editor ---------- */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
}}

/* ---------- alerts ---------- */
div[data-testid="stAlert"] {{
    border-radius: 12px;
    border: 1px solid {BORDER};
}}

/* ---------- custom components ---------- */
.ds-stepper {{
    display: flex; align-items: center; gap: 0;
    margin: 0 0 1.6rem 0; padding: 1.0rem 1.2rem;
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 14px; box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}}
.ds-step {{ display: flex; align-items: center; gap: 9px; }}
.ds-step-circle {{
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12.5px; font-weight: 700; flex: none;
}}
.ds-step-label {{ font-size: 0.86rem; font-weight: 600; white-space: nowrap; }}
.ds-step.done .ds-step-circle {{ background: #E3E6FB; color: {PRIMARY}; }}
.ds-step.done .ds-step-label {{ color: {TEXT_MUTED}; }}
.ds-step.active .ds-step-circle {{ background: {GRADIENT}; color: #fff; box-shadow: 0 0 0 4px rgba(79,70,229,0.15); }}
.ds-step.active .ds-step-label {{ color: {TEXT}; }}
.ds-step.pending .ds-step-circle {{ background: #EEF0F7; color: #9AA1BC; }}
.ds-step.pending .ds-step-label {{ color: #9AA1BC; }}
.ds-step-line {{ flex: 1; height: 2px; background: #E7EAF3; margin: 0 12px; min-width: 18px; }}
.ds-step-line.done {{ background: #C9CDF5; }}

.ds-section {{ display: flex; align-items: baseline; gap: 12px; margin: 2.1rem 0 0.4rem 0; }}
.ds-section-num {{
    width: 26px; height: 26px; border-radius: 8px; flex: none;
    background: {GRADIENT}; color: #fff;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; transform: translateY(3px);
}}
.ds-section-title {{ font-size: 1.32rem; font-weight: 700; letter-spacing: -0.015em; }}
.ds-section-sub {{ color: {TEXT_MUTED}; font-size: 0.92rem; margin: 0.1rem 0 0.9rem 38px; }}

.ds-hero {{
    background: {GRADIENT};
    border-radius: 16px; padding: 1.5rem 1.8rem; color: #fff;
    display: flex; align-items: center; gap: 2.2rem; flex-wrap: wrap;
    box-shadow: 0 8px 24px rgba(79,70,229,0.25);
    margin: 0.4rem 0 1.1rem 0;
}}
.ds-hero-score {{ font-size: 3.2rem; font-weight: 800; line-height: 1; letter-spacing: -0.03em; }}
.ds-hero-label {{ font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.85; }}
.ds-hero-chip {{
    display: inline-block; padding: 5px 14px; border-radius: 999px;
    background: rgba(255,255,255,0.18); font-size: 0.85rem; font-weight: 600;
    backdrop-filter: blur(4px);
}}
.ds-hero-note {{ font-size: 0.95rem; opacity: 0.92; max-width: 520px; }}

.ds-brand {{ font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; }}
.ds-brand .dot {{ color: #A5B4FC; }}

.ds-feature {{ display: flex; gap: 14px; margin-bottom: 1.05rem; align-items: flex-start; }}
.ds-feature-icon {{
    width: 38px; height: 38px; border-radius: 11px; flex: none;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.18);
    display: flex; align-items: center; justify-content: center;
}}
.ds-feature-title {{ font-weight: 700; font-size: 0.98rem; color: #fff; }}
.ds-feature-desc {{ font-size: 0.86rem; color: rgba(255,255,255,0.78); line-height: 1.45; }}

.ds-panel {{
    background: {GRADIENT};
    border-radius: 20px; padding: 2.4rem 2.2rem;
    box-shadow: 0 16px 40px rgba(79,70,229,0.28);
}}
</style>
"""

# ---------------------------------------------------------------------------
# shared HTML components
# ---------------------------------------------------------------------------

STAGES = ["Upload", "Explore", "Review", "Execute", "Results"]


def stepper(active_index: int) -> str:
    parts = ['<div class="ds-stepper">']
    for index, name in enumerate(STAGES):
        state = "done" if index < active_index else "active" if index == active_index else "pending"
        circle = "&#10003;" if state == "done" else str(index + 1)
        parts.append(
            f'<div class="ds-step {state}">'
            f'<div class="ds-step-circle">{circle}</div>'
            f'<div class="ds-step-label">{name}</div></div>'
        )
        if index < len(STAGES) - 1:
            line_state = "done" if index < active_index else ""
            parts.append(f'<div class="ds-step-line {line_state}"></div>')
    parts.append("</div>")
    return "".join(parts)


def section(number: int, title: str, subtitle: str = "") -> str:
    html = (
        f'<div class="ds-section"><span class="ds-section-num">{number}</span>'
        f'<span class="ds-section-title">{title}</span></div>'
    )
    if subtitle:
        html += f'<div class="ds-section-sub">{subtitle}</div>'
    return html


def hero(score: int, lift: int, status: str, note: str) -> str:
    chip_text = {
        "clean": "All checks passed",
        "partial": "Finished with warnings",
        "stopped": "Stopped early",
    }.get(status, status)
    lift_text = f"+{lift}" if lift >= 0 else str(lift)
    return f"""
<div class="ds-hero">
  <div>
    <div class="ds-hero-label">Quality score</div>
    <div class="ds-hero-score">{score}<span style="font-size:1.3rem;font-weight:600;opacity:.8"> /100</span></div>
    <div style="margin-top:6px;font-weight:700">{lift_text} vs upload</div>
  </div>
  <div style="flex:1;min-width:260px">
    <span class="ds-hero-chip">{chip_text}</span>
    <div class="ds-hero-note" style="margin-top:10px">{note}</div>
  </div>
</div>
"""


def brand_html(size: str = "1.5rem", light: bool = False) -> str:
    color = "#FFFFFF" if light else TEXT
    return (
        f'<div class="ds-brand" style="font-size:{size};color:{color}">'
        f'Clean<span class="dot">.ai</span></div>'
    )
