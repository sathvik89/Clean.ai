import streamlit as st

from app.ui.components import user_store
from cleanai.services.auth import AuthError

ICONS = {
    "profile": '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.5" y2="16.5"/></svg>',
    "plan": '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round"><path d="M9 6h11M9 12h11M9 18h11"/><path d="M4 6l1 1 2-2M4 12l1 1 2-2M4 18l1 1 2-2"/></svg>',
    "shield": '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 3v5c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6l8-3z"/><path d="M9 12l2 2 4-4"/></svg>',
    "chart": '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round"><path d="M4 20V10M10 20V4M16 20v-7M21 20H3"/></svg>',
    "recipe": '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2h-2"/><path d="M9 14l2 2 4-5"/><rect x="8" y="1.5" width="8" height="4" rx="1"/></svg>',
}

FEATURES = [
    ("profile", "Deep profiling", "Finds the real problems — even 'N/A' and '-' hiding as text, bad emails, impossible ages."),
    ("plan", "A plan you control", "Review every step with a dry-run impact preview. Edit cells or just say \"don't drop rows\"."),
    ("shield", "Safe by design", "Sandboxed code, row-deletion caps, duplicate-call guards, and a full action ledger."),
    ("chart", "Proof it worked", "Before/after quality scores and charts, saved to your run history."),
    ("recipe", "A recipe to keep", "A standalone pandas script that re-runs the cleaning anywhere — no API key needed."),
]


def _features_html() -> str:
    rows = []
    for icon, title, desc in FEATURES:
        rows.append(
            f'<div class="ds-feature"><div class="ds-feature-icon">{ICONS[icon]}</div>'
            f'<div><div class="ds-feature-title">{title}</div>'
            f'<div class="ds-feature-desc">{desc}</div></div></div>'
        )
    return "".join(rows)


def login_page() -> None:
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    left, gap, right = st.columns([1.15, 0.08, 1], gap="small")

    with left:
        st.markdown(
            f"""
<div class="ds-panel">
  <div class="ds-brand" style="color:#fff;font-size:1.9rem;margin-bottom:6px">Clean<span class="dot">.ai</span></div>
  <div style="color:rgba(255,255,255,.92);font-size:1.12rem;font-weight:600;margin-bottom:1.6rem">
    An agent that cleans your messy data — and shows its work.
  </div>
  {_features_html()}
  <div style="color:rgba(255,255,255,.6);font-size:.82rem;margin-top:1.4rem">
    It learns, too: every finished run teaches the planner better strategies for similar datasets.
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown("### Welcome")
        st.caption("Sign in to clean your first dataset.")
        sign_in_tab, register_tab = st.tabs(["Sign in", "Create account"])

        with sign_in_tab:
            with st.form("sign_in", border=False):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                    user = user_store().authenticate(email, password)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Wrong email or password.")

        with register_tab:
            with st.form("register", border=False):
                name = st.text_input("Name (optional)")
                email = st.text_input("Email", key="reg_email")
                password = st.text_input("Password (8+ characters)", type="password", key="reg_password")
                if st.form_submit_button("Create account", type="primary", use_container_width=True):
                    try:
                        user = user_store().register(email, password, name)
                        st.session_state.user = user
                        st.toast("Account created — welcome!", icon=":material/verified:")
                        st.rerun()
                    except AuthError as exc:
                        st.error(str(exc))
