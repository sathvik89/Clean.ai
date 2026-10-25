"""Shared UI helpers: stores, auth gate, small render utilities."""

from __future__ import annotations

import streamlit as st

from cleanai.config import get_settings
from cleanai.services.auth import UserStore
from cleanai.services.persistence import RunStore


@st.cache_resource
def user_store() -> UserStore:
    return UserStore(get_settings().users_db_path)


@st.cache_resource
def run_store() -> RunStore:
    return RunStore(get_settings().runs_db_path)


def current_user() -> dict | None:
    return st.session_state.get("user")


def require_user() -> dict:
    user = current_user()
    if not user:
        st.warning("Please sign in first.")
        st.stop()
    return user


def logout() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def step_icon(status: str) -> str:
    icons = {
        "done": ":material/check_circle:",
        "running": ":material/autorenew:",
        "blocked": ":material/block:",
        "skipped": ":material/skip_next:",
    }
    return icons.get(status, ":material/radio_button_unchecked:")


def render_step_timeline(steps: list[dict], container) -> None:
    lines = []
    for step in steps:
        icon = step_icon(step.get("status", "pending"))
        target = ", ".join(step.get("columns") or []) or "whole table"
        result = f" — {step['result']}" if step.get("result") else ""
        lines.append(f"{icon} **{step['id']} · {step['tool'].replace('_', ' ')}** on {target}{result}")
    container.markdown("\n\n".join(lines) if lines else "_no steps_")


def reset_workspace() -> None:
    for key in ("ws_stage", "df", "file_name", "context_text", "cleaning_session",
                "analysis_ready", "summary", "saved_run_id", "plan_impacts"):
        st.session_state.pop(key, None)
    for key in [k for k in st.session_state if str(k).startswith("sugg_")]:
        st.session_state.pop(key, None)
