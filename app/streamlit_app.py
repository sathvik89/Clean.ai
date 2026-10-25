"""Clean.ai entry point.

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import streamlit as st

from app.ui.components import current_user, logout
from app.ui.theme import GLOBAL_CSS, PRIMARY
from app.views.history import history_page
from app.views.insights import insights_page
from app.views.login import login_page
from app.views.run_detail import run_detail_page
from app.views.workspace import workspace_page

st.set_page_config(
    page_title="Clean.ai", page_icon="🧼",
    layout="wide", initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

if current_user():
    workspace = st.Page(workspace_page, title="Workspace", icon=":material/cleaning_services:", default=True)
    run_detail = st.Page(run_detail_page, title="Run detail", icon=":material/receipt_long:")
    history = st.Page(history_page, title="History", icon=":material/history:")
    insights = st.Page(insights_page, title="Insights", icon=":material/monitoring:")
    st.session_state.run_detail_page = run_detail

    navigation = st.navigation({"Clean": [workspace, run_detail], "Review": [history, insights]})

    with st.sidebar:
        st.markdown(
            f"<h2 style='margin-bottom:0'>🧼 Clean<span style='color:{PRIMARY}'>.ai</span></h2>",
            unsafe_allow_html=True,
        )
        st.caption("agentic data cleaning")
        st.divider()
        user = current_user()
        st.markdown(f"**{user.get('display_name') or user['email']}**")
        if st.button("Log out", use_container_width=True):
            logout()
else:
    navigation = st.navigation([st.Page(login_page, title="Sign in", icon=":material/login:")])

navigation.run()
