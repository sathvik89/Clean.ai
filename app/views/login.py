import streamlit as st

from app.ui.components import user_store
from app.ui.theme import PRIMARY, SECONDARY, TEXT_MUTED
from cleanai.services.auth import AuthError


def login_page() -> None:
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.markdown(f"<h1 style='font-size:2.6rem'>🧼 Clean<span style='color:{PRIMARY}'>.ai</span></h1>",
                    unsafe_allow_html=True)
        st.markdown(
            "#### An agent that cleans your messy data — and shows its work.\n\n"
            "Upload a messy CSV, tell it what the data is for, and Clean.ai will:\n"
            "- 🔍 **Profile** every column and find the real problems (even `'N/A'` hiding as text)\n"
            "- 📋 **Propose a plan** you can inspect, edit, or rewrite in plain English\n"
            "- 🛡️ **Execute safely** — sandboxed code, row-deletion guardrails, full action ledger\n"
            "- 📈 **Prove it worked** with before/after quality scores and charts\n"
            "- 🧾 **Hand you a recipe** — a standalone python script that re-runs the cleaning anywhere\n"
        )
        st.markdown(
            f"<span style='color:{TEXT_MUTED}'>It remembers what worked: every finished run teaches "
            "the planner better strategies for similar datasets.</span>",
            unsafe_allow_html=True,
        )

    with right:
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
                        st.toast("Account created — welcome!", icon="🎉")
                        st.rerun()
                    except AuthError as exc:
                        st.error(str(exc))
