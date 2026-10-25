"""Palette and shared styling for the Clean.ai UI.

The categorical order is fixed and was validated for colorblind
separation on a white surface - keep the order, don't substitute hues.
"""

PRIMARY = "#4F46E5"      # indigo - buttons, links, "after" series
SECONDARY = "#0891B2"    # cyan - info accents
ACCENT = "#DB2777"       # pink - highlights, deltas
SUCCESS = "#059669"
WARNING = "#D97706"
DANGER = "#DC2626"
BACKGROUND = "#F8FAFC"
SURFACE = "#FFFFFF"
TEXT = "#0F172A"
TEXT_MUTED = "#64748B"
BORDER = "#E2E8F0"

# fixed categorical order for charts; "before" is always neutral gray,
# "after" is always primary so the entity keeps its color everywhere
CATEGORICAL = [PRIMARY, ACCENT, SECONDARY, WARNING, "#047857"]
BEFORE_COLOR = "#94A3B8"
AFTER_COLOR = PRIMARY

GLOBAL_CSS = f"""
<style>
    .block-container {{ padding-top: 2.2rem; }}
    h1, h2, h3 {{ color: {TEXT}; }}
    .stButton > button[kind="primary"] {{ border-radius: 10px; font-weight: 600; }}
    .stButton > button {{ border-radius: 10px; }}
    div[data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }}
    div[data-testid="stExpander"] {{
        background: {SURFACE};
        border-radius: 12px;
    }}
    .cleanai-chip {{
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        font-size: 0.82rem; font-weight: 600; margin-right: 6px;
    }}
    .chip-done {{ background: #D1FAE5; color: {SUCCESS}; }}
    .chip-active {{ background: {PRIMARY}; color: white; }}
    .chip-pending {{ background: {BORDER}; color: {TEXT_MUTED}; }}
</style>
"""

STAGES = ["Upload", "Context", "Analysis", "Plan", "Execute", "Results"]


def stage_chips(active_index: int) -> str:
    parts = []
    for index, name in enumerate(STAGES):
        css = "chip-done" if index < active_index else "chip-active" if index == active_index else "chip-pending"
        parts.append(f'<span class="cleanai-chip {css}">{name}</span>')
    return '<div style="margin-bottom:1rem">' + "".join(parts) + "</div>"
