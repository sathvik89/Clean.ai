import os
import re
import traceback
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from graph import app
from tools import format_log_details


load_dotenv()


APP_NAME = "Clean.ai"
SAMPLE_DATASET_NAME = "sample_messy_dataset.csv"
SAMPLE_DATASET_PATH = os.path.join(os.path.dirname(__file__), SAMPLE_DATASET_NAME)
SAMPLE_DATASET_DESCRIPTION = (
    "This is a messy employee operations dataset prepared for HR analytics and compliance reporting. "
    "Please clean invalid ages, salaries, dates, emails, duplicate employee identifiers, missing values, "
    "inconsistent department labels, and mixed formatting so the final dataset is reliable for downstream analysis."
)


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css');

    :root {
        --bg: #f6f1e8;
        --panel: rgba(255, 251, 245, 0.82);
        --panel-strong: rgba(255, 248, 238, 0.95);
        --text: #24170f;
        --muted: #695448;
        --accent: #1f7a5c;
        --accent-dark: #14523e;
        --accent-soft: rgba(31, 122, 92, 0.10);
        --border: rgba(36, 23, 15, 0.10);
        --shadow: 0 18px 50px rgba(44, 29, 19, 0.10);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(31, 122, 92, 0.12), transparent 30%),
            radial-gradient(circle at top right, rgba(204, 151, 71, 0.14), transparent 28%),
            linear-gradient(180deg, #fbf7f1 0%, #f4ede3 100%);
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(23, 34, 30, 0.98) 0%, rgba(17, 24, 21, 0.98) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f7f1e7;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.04);
        color: #f7f1e7;
        font-weight: 600;
        min-height: 3rem;
        box-shadow: none;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(255, 255, 255, 0.24);
        background: rgba(255, 255, 255, 0.08);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
        color: var(--text) !important;
        letter-spacing: 0.01em;
    }

    .brand-shell {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        padding: 1rem 0 1.2rem 0;
    }

    .brand-mark {
        width: 52px;
        height: 52px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #f3cf8e 0%, #2ea37a 100%);
        color: #112019;
        font-size: 1.35rem;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.22);
    }

    .brand-copy h1 {
        margin: 0;
        font-size: 2rem;
        color: #fff7eb !important;
    }

    .brand-copy p {
        margin: 0.1rem 0 0 0;
        color: rgba(247, 241, 231, 0.72);
        font-size: 0.95rem;
    }

    .sidebar-label {
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(247, 241, 231, 0.58);
        margin: 1rem 0 0.55rem 0;
        font-weight: 700;
    }

    .hero-card, .section-card, .metric-card, .node-card, .plan-card, .log-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 24px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
    }

    .hero-card {
        padding: 1.55rem 1.7rem;
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent-dark);
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 1rem 0 0.7rem 0;
        font-size: clamp(2.4rem, 5vw, 4.4rem);
        line-height: 1.02;
        max-width: 16ch;
    }

    .hero-subtitle {
        color: var(--muted);
        max-width: 60rem;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    .highlight-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }

    .section-card {
        padding: 1.4rem;
        height: 100%;
    }

    .section-card h3 {
        margin-top: 0.3rem;
        margin-bottom: 0.65rem;
        font-size: 1.7rem;
    }

    .section-card p {
        color: var(--muted);
        line-height: 1.7;
        margin-bottom: 0;
    }

    .metric-card {
        padding: 0.95rem 1.1rem;
        min-height: 104px;
        margin-bottom: 1rem;
    }

    .metric-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        font-size: 0.78rem;
    }

    .metric-value {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.15rem;
        line-height: 1;
        margin: 0.45rem 0 0.25rem 0;
        color: var(--text);
    }

    .metric-help {
        color: var(--muted);
        margin: 0;
    }

    .workspace-card {
        background: var(--panel-strong);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.1rem 1.2rem;
        box-shadow: var(--shadow);
        margin-bottom: 0.9rem;
    }

    .node-card, .plan-card, .log-card {
        padding: 0.8rem 0.95rem;
        margin-bottom: 0.55rem;
        border-radius: 18px;
    }

    .node-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.25rem;
    }

    .node-name {
        font-weight: 700;
        color: var(--text);
        text-transform: capitalize;
    }

    .node-status {
        border-radius: 999px;
        padding: 0.18rem 0.55rem;
        background: var(--accent-soft);
        color: var(--accent-dark);
        font-size: 0.74rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .node-log {
        color: var(--muted);
        margin: 0.1rem 0 0 0;
        line-height: 1.45;
        font-size: 0.94rem;
    }

    .plan-editor-label {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 0.45rem;
    }

    .activity-list {
        margin-top: 0.6rem;
    }

    .activity-row {
        display: flex;
        align-items: flex-start;
        gap: 0.65rem;
        padding: 0.32rem 0;
        border-bottom: 1px solid rgba(36, 23, 15, 0.06);
    }

    .activity-row:last-child {
        border-bottom: none;
    }

    .activity-bullet {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--accent);
        margin-top: 0.48rem;
        flex-shrink: 0;
    }

    .activity-copy {
        line-height: 1.5;
        color: var(--muted);
        font-size: 0.96rem;
    }

    .activity-copy strong {
        color: var(--text);
        font-weight: 700;
    }

    .activity-running {
        color: var(--accent-dark);
        font-weight: 700;
        font-size: 0.86rem;
        margin-left: 0.35rem;
    }

    .step-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.9rem;
    }

    .step-chip {
        border-radius: 999px;
        padding: 0.5rem 0.85rem;
        background: rgba(255, 255, 255, 0.68);
        border: 1px solid var(--border);
        color: var(--text);
        font-size: 0.88rem;
        font-weight: 600;
    }

    .stTextArea textarea, .stFileUploader {
        border-radius: 18px !important;
    }

    .stTextArea textarea {
        min-height: 140px !important;
        padding-top: 0.9rem !important;
        padding-bottom: 0.9rem !important;
    }

    .stButton > button {
        border-radius: 16px;
        border: none;
        background: linear-gradient(135deg, var(--accent-dark) 0%, var(--accent) 100%);
        color: #f9f4eb;
        font-weight: 700;
        min-height: 3rem;
        box-shadow: 0 10px 22px rgba(31, 122, 92, 0.18);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #103d2f 0%, #216d52 100%);
        color: #fffaf2;
    }

    .logs-empty {
        color: var(--muted);
        text-align: center;
        padding: 2rem 1rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        padding: 0.7rem 0.9rem !important;
        border-radius: 18px !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        padding: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialise_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "step" not in st.session_state:
        st.session_state.step = "idle"
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "node_updates" not in st.session_state:
        st.session_state.node_updates = []
    if "nav" not in st.session_state:
        st.session_state.nav = "landing"
    if "uploaded_name" not in st.session_state:
        st.session_state.uploaded_name = ""
    if "run_error" not in st.session_state:
        st.session_state.run_error = ""
    if "current_activity" not in st.session_state:
        st.session_state.current_activity = None
    if "source_df" not in st.session_state:
        st.session_state.source_df = None
    if "source_name" not in st.session_state:
        st.session_state.source_name = ""
    if "final_df" not in st.session_state:
        st.session_state.final_df = None
    if "final_csv_bytes" not in st.session_state:
        st.session_state.final_csv_bytes = None
    if "sample_csv_bytes" not in st.session_state:
        st.session_state.sample_csv_bytes = None
    if "user_context_text" not in st.session_state:
        st.session_state.user_context_text = ""


def reset_workspace() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.step = "idle"
    st.session_state.logs = []
    st.session_state.node_updates = []
    st.session_state.uploaded_name = ""
    st.session_state.run_error = ""
    st.session_state.current_activity = None
    st.session_state.source_df = None
    st.session_state.source_name = ""
    st.session_state.final_df = None
    st.session_state.final_csv_bytes = None
    st.session_state.sample_csv_bytes = None
    st.session_state.user_context_text = ""


def set_nav(target: str) -> None:
    st.session_state.nav = target


def load_sample_dataset() -> None:
    sample_bytes = open(SAMPLE_DATASET_PATH, "rb").read()
    sample_df = pd.read_csv(SAMPLE_DATASET_PATH)
    st.session_state.source_df = sample_df
    st.session_state.source_name = SAMPLE_DATASET_NAME
    st.session_state.uploaded_name = SAMPLE_DATASET_NAME
    st.session_state.user_context_text = SAMPLE_DATASET_DESCRIPTION
    st.session_state.final_df = None
    st.session_state.final_csv_bytes = None
    st.session_state.sample_csv_bytes = sample_bytes
    st.session_state.step = "idle"


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand-shell">
                <div class="brand-mark"><i class="bi bi-stars"></i></div>
                <div class="brand-copy">
                    <h1>{APP_NAME}</h1>
                    <p>agentic data cleaning workspace</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">Navigate</div>', unsafe_allow_html=True)
        if st.button("Landing Page", use_container_width=True):
            set_nav("landing")
        if st.button("Clean Your Data", use_container_width=True):
            set_nav("clean")

        st.markdown('<div class="sidebar-label">Review</div>', unsafe_allow_html=True)
        if st.button("Open Logs", use_container_width=True):
            set_nav("logs")

        st.markdown('<div class="sidebar-label">Product</div>', unsafe_allow_html=True)
        st.markdown(
            """
            Clean.ai analyzes messy CSV datasets, plans a cleaning strategy,
            waits for your approval, executes step by step with tools, and
            validates the result before delivery.
            """
        )

        st.markdown('<div class="sidebar-label">Backend</div>', unsafe_allow_html=True)
        if os.getenv("GROQ_API_KEY"):
            st.success("groq connection detected")
        else:
            st.warning("groq api key is not configured in the environment")


def sync_uploaded_dataset(uploaded_file) -> None:
    if uploaded_file is None:
        return
    st.session_state.source_df = pd.read_csv(uploaded_file)
    st.session_state.source_name = uploaded_file.name
    st.session_state.uploaded_name = uploaded_file.name
    st.session_state.final_df = None
    st.session_state.final_csv_bytes = None


def get_active_dataset():
    if st.session_state.source_df is None:
        return None, ""
    return st.session_state.source_df.copy(), st.session_state.source_name


def build_readable_logs(logs: list[str]) -> str:
    if not logs:
        return "No logs available yet."
    lines = []
    for index, log in enumerate(logs, start=1):
        lines.append(f"{index}. {format_log_details(log)}")
    return "\n\n".join(lines)


def add_node_update(node_name: str, output: dict) -> None:
    if isinstance(output, tuple):
        if len(output) >= 2 and isinstance(output[1], dict):
            output = output[1]
        else:
            output = {}
    elif output is None:
        output = {}

    label = node_name.replace("_", " ")
    logs = output.get("logs", [])
    message = logs[-1] if logs else "node completed"
    st.session_state.node_updates.append(
        {
            "node": label,
            "message": prettify_message(label, message),
        }
    )


def normalize_output(output):
    if isinstance(output, tuple):
        if len(output) >= 2 and isinstance(output[1], dict):
            return output[1]
        return {}
    if output is None:
        return {}
    return output


def prettify_message(node_name: str, message: str) -> str:
    if not message:
        if "executor" in node_name.lower():
            return "choosing the next tool action for the cleaning workflow"
        return "processing"

    clean_message = message.strip()
    replacements = {
        "initial analysis complete": "analyzing the dataset structure, null patterns, and column types",
        "strategic cleaning plan generated": "building the suggested cleaning plan for your review",
        "awaiting human confirmation of the plan": "waiting for your approval before execution starts",
        "data quality audit complete": "validating the cleaned dataset against the requested goal",
    }
    if clean_message in replacements:
        return replacements[clean_message]
    if clean_message.startswith("handled missing values"):
        return "handling missing values using the selected columns and strategy"
    if clean_message.startswith("removed duplicates"):
        return "removing duplicate rows based on the planned rule"
    if clean_message.startswith("converted types"):
        return "converting column types to match the intended schema"
    if clean_message.startswith("analyzed outliers"):
        return "checking numeric columns for outliers before finalizing the plan"
    if clean_message.startswith("cleaned text"):
        return "standardizing text formatting and cleaning string columns"
    if clean_message.startswith("executed custom pandas"):
        return "running a custom pandas cleanup step for a complex transformation"
    if clean_message.startswith("handled categorical encoding"):
        return "encoding categorical columns for downstream use"
    if clean_message.startswith("data quality audit complete: all validation checks passed"):
        return "validation passed and the dataset is ready to finish"
    if clean_message.startswith("data quality audit complete: remaining issues:"):
        return clean_message.replace("data quality audit complete: ", "")
    return clean_message


def format_plan_for_editor(plan: str) -> str:
    if not plan:
        return ""

    formatted_lines = []
    step_number = 1
    for raw_line in plan.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line = re.sub(r"^#{1,6}\s*", "", line)
        if re.match(r"^[-*]\s+", line):
            content = re.sub(r"^[-*]\s+", "", line)
            formatted_lines.append(f"{step_number}. {content}")
            step_number += 1
            continue

        if re.match(r"^\d+[\.\)]\s+", line):
            content = re.sub(r"^\d+[\.\)]\s+", "", line)
            formatted_lines.append(f"{step_number}. {content}")
            step_number += 1
            continue

        formatted_lines.append(line)

    return "\n".join(formatted_lines)


def render_node_progress(container=None, active_node: str | None = None, active_message: str | None = None) -> None:
    target = container if container is not None else st
    with target.container():
        st.subheader("Current Process")
        st.caption("Live progress so the user can follow what the system is doing.")

        activity_lines = []
        if active_node and active_message:
            activity_lines.append(f"- **{active_node}**: {active_message} _(running)_")

        if st.session_state.node_updates:
            for item in st.session_state.node_updates:
                activity_lines.append(f"- **{item['node']}**: {item['message']}")

        if not activity_lines:
            st.info("No workflow activity yet. Start analysis to see live node progress here.")
        else:
            with st.expander("View Process Details", expanded=True):
                st.markdown("\n".join(activity_lines))


def run_graph_stream(initial_state=None, progress_placeholder=None) -> None:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    print(
        f"[clean.ai][app] starting graph stream thread_id={st.session_state.thread_id} "
        f"resume={initial_state is None}",
        flush=True,
    )
    try:
        for event in app.stream(initial_state, config):
            print(f"[clean.ai][app] stream event keys={list(event.keys())}", flush=True)
            for node_name, output in event.items():
                normalized_output = normalize_output(output)
                preview_logs = normalized_output.get("logs", [])
                current_message = prettify_message(node_name, preview_logs[-1] if preview_logs else "")
                st.session_state.current_activity = {
                    "node": node_name.replace("_", " "),
                    "message": current_message,
                }
                if progress_placeholder is not None:
                    render_node_progress(
                        progress_placeholder,
                        active_node=st.session_state.current_activity["node"],
                        active_message=st.session_state.current_activity["message"],
                    )

                add_node_update(node_name, normalized_output)

                if "logs" in normalized_output:
                    for log in normalized_output["logs"]:
                        print(f"[clean.ai][app] log={log}", flush=True)
                        st.session_state.logs.append(log)
                if progress_placeholder is not None:
                    render_node_progress(progress_placeholder)
        st.session_state.current_activity = None
    except Exception as error:
        error_text = str(error)
        print(f"[clean.ai][app] graph stream failed: {error_text}", flush=True)
        traceback.print_exc()
        st.session_state.run_error = error_text
        st.session_state.logs.append(f"run failed: {error_text}")
        st.session_state.current_activity = None
        raise


def render_landing_page() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow"><i class="bi bi-magic"></i> goal-aware cleaning</div>
            <h1 class="hero-title">Reliable data cleaning, with approval in the loop.</h1>
            <p class="hero-subtitle">
                Clean.ai combines agent reasoning, tool-based execution, and validation loops so messy
                CSV files become usable for machine learning, BI, analytics, auditing, or custom workflows.
                The system analyzes the dataset, creates a cleaning plan, waits for human approval, then
                executes step by step and verifies the result before finishing.
            </p>
            <div class="step-chip-row">
                <div class="step-chip">analyze structure</div>
                <div class="step-chip">build cleaning plan</div>
                <div class="step-chip">human review</div>
                <div class="step-chip">tool-based execution</div>
                <div class="step-chip">validation loop</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="section-card">
                <div class="eyebrow"><i class="bi bi-cpu"></i> what it does</div>
                <h3>Understands before it acts</h3>
                <p>
                    The agent studies dataset shape, types, missing values, duplicates, and structure before
                    it proposes any cleaning action. Intent from the user is part of the reasoning loop.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="section-card">
                <div class="eyebrow"><i class="bi bi-list-check"></i> how it works</div>
                <h3>Plans, then executes with tools</h3>
                <p>
                    No hidden dataframe mutations inside the model. The llm decides the next action and pandas
                    tools perform the actual changes in a traceable workflow.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="section-card">
                <div class="eyebrow"><i class="bi bi-shield-check"></i> why it matters</div>
                <h3>Validation is built in</h3>
                <p>
                    After execution, the system audits the cleaned data again and loops when needed, helping
                    ensure the result is actually ready for the intended use case.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        """
        <div class="workspace-card">
            <div class="eyebrow"><i class="bi bi-compass"></i> how to use it</div>
            <h2 style="margin-top:0.8rem;">Simple workflow</h2>
            <div class="highlight-grid">
                <div class="section-card">
                    <h3>1. Upload</h3>
                    <p>Open the clean your data workspace and upload a CSV dataset.</p>
                </div>
                <div class="section-card">
                    <h3>2. Describe intent</h3>
                    <p>Explain what the dataset is about and the cleaning goal or downstream use case.</p>
                </div>
                <div class="section-card">
                    <h3>3. Review the plan</h3>
                    <p>Edit the suggested cleaning steps, accept them, or reject and restart.</p>
                </div>
                <div class="section-card">
                    <h3>4. Track progress</h3>
                    <p>Watch node-by-node execution and open the logs page for a dedicated review view.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_overview(df: pd.DataFrame, file_name: str) -> None:
    st.markdown('<div class="workspace-card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="eyebrow"><i class="bi bi-database"></i> dataset loaded</div>
        <h2 style="margin-top:0.55rem;">{file_name}</h2>
        <p style="color:var(--muted); margin-top:-0.4rem;">
            Quick overview
        </p>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">rows</div>
                <div class="metric-value">{df.shape[0]:,}</div>
                <p class="metric-help">total records in the uploaded dataset</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">columns</div>
                <div class="metric-value">{df.shape[1]:,}</div>
                <p class="metric-help">fields available for analysis and cleaning</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_plan_editor(config: dict) -> None:
    state = app.get_state(config)
    st.markdown(
        """
        <div class="workspace-card">
            <div class="eyebrow"><i class="bi bi-clipboard2-check"></i> plan review</div>
            <h2 style="margin-top:0.45rem;">Review the suggested cleaning plan</h2>
            <p style="color:var(--muted);">
                Edit the plan if needed, then accept it to continue. Rejecting the plan will reset this run.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="plan-editor-label">editable plan text</div>', unsafe_allow_html=True)
    plan = st.text_area(
        "Suggested cleaning plan",
        value=format_plan_for_editor(state.values["cleaning_plan"]),
        height=320,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Accept Plan And Continue", use_container_width=True):
            app.update_state(config, {"cleaning_plan": plan})
            st.session_state.step = "clean"
            st.rerun()
    with col2:
        if st.button("Reject Plan", use_container_width=True):
            reset_workspace()
            st.session_state.nav = "clean"
            st.rerun()


def render_cleaning_workspace() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow"><i class="bi bi-sliders2"></i> clean your data</div>
            <h1 class="hero-title">Upload, review, approve, and validate.</h1>
            <p class="hero-subtitle">
                This workspace keeps the flow focused: dataset summary, intent capture, plan review,
                visible node progress, and a separate logs area for full trace review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    upload_col, sample_col = st.columns([3, 1])
    with upload_col:
        uploaded_file = st.file_uploader("upload your messy dataset", type=["csv"])
        if st.session_state.source_name == SAMPLE_DATASET_NAME:
            st.caption(f"Using sample dataset: `{SAMPLE_DATASET_NAME}`")
    with sample_col:
        st.markdown("<div style='height: 1.95rem;'></div>", unsafe_allow_html=True)
        if st.button("Use Sample Dataset", use_container_width=True):
            load_sample_dataset()
            st.rerun()
        if st.session_state.sample_csv_bytes is not None:
            st.download_button(
                "Download Sample CSV",
                data=st.session_state.sample_csv_bytes,
                file_name=SAMPLE_DATASET_NAME,
                mime="text/csv",
                use_container_width=True,
            )

    sync_uploaded_dataset(uploaded_file)
    df, dataset_name = get_active_dataset()
    if df is None:
        st.info("Upload a CSV file or use the sample dataset to start exploring the system.")
        render_node_progress()
        return

    render_dataset_overview(df, dataset_name)
    progress_placeholder = st.empty()
    render_node_progress(progress_placeholder)

    st.markdown('<div class="workspace-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="eyebrow"><i class="bi bi-chat-square-text"></i> cleaning intent</div>
        <h2 style="margin-top:0.8rem;">Tell the agent what this dataset is for</h2>
        """,
        unsafe_allow_html=True,
    )
    user_context = st.text_area(
        "Describe the dataset and intended use case",
        placeholder="for example: customer churn modeling, finance audit prep, dashboard-ready sales analytics, or compliance review",
        height=140,
        value=st.session_state.user_context_text,
    )
    st.session_state.user_context_text = user_context

    if st.session_state.step == "idle":
        if st.button("Analyze And Generate Plan", use_container_width=True):
            st.session_state.logs = []
            st.session_state.node_updates = []
            st.session_state.run_error = ""
            initial_state = {
                "file_path": dataset_name,
                "user_context": user_context or "general cleaning",
                "current_df": df,
                "df_history": [df],
                "analysis_report": "",
                "cleaning_plan": "",
                "audit_report": {},
                "validation_profile": {},
                "work_summary": "",
                "validation_feedback": "",
                "run_error": "",
                "is_clean": False,
                "tool_rounds": 0,
                "max_tool_rounds": 12,
                "validation_rounds": 0,
                "max_validation_rounds": 6,
                "validation_signature": "",
                "stagnation_rounds": 0,
                "max_stagnation_rounds": 2,
                "logs": [],
                "messages": [],
            }

            try:
                with st.status("analyzing dataset and preparing a cleaning plan...", expanded=True) as status:
                    run_graph_stream(initial_state=initial_state, progress_placeholder=progress_placeholder)
                    status.update(label="plan is ready for review", state="complete")
            except Exception as error:
                error_text = str(error)
                if "rate_limit" in error_text.lower() or "429" in error_text:
                    st.error("Groq rate limit reached. The current account has exhausted its token quota for now. Please wait and try again later.")
                else:
                    st.error(f"Run failed during analysis/planning: {error_text}")
                return

            st.session_state.step = "plan"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.step == "plan":
        render_plan_editor(config)

    if st.session_state.step == "clean":
        st.markdown(
            """
            <div class="workspace-card">
                <div class="eyebrow"><i class="bi bi-gear-wide-connected"></i> execution</div>
                <h2 style="margin-top:0.45rem;">Agent is cleaning and validating the dataset</h2>
                <p style="color:var(--muted);">
                    The workflow below updates as the graph moves through the execution and validation nodes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            with st.status("executing plan and validating results...", expanded=True) as status:
                run_graph_stream(progress_placeholder=progress_placeholder)
                status.update(label="cleaning run finished", state="complete")
        except Exception as error:
            error_text = str(error)
            if "rate_limit" in error_text.lower() or "429" in error_text:
                st.error("Groq rate limit reached during execution. The run has been paused because the account token quota is exhausted.")
            else:
                st.error(f"Run failed during execution: {error_text}")
            render_node_progress(progress_placeholder)
            return

        st.session_state.step = "finished"
        st.rerun()

    if st.session_state.step == "finished":
        state = app.get_state(config)
        final_df = state.values["current_df"]
        run_error = state.values.get("run_error", "")
        st.session_state.final_df = final_df.copy()
        st.session_state.final_csv_bytes = final_df.to_csv(index=False).encode("utf-8")

        st.markdown(
            """
            <div class="workspace-card">
                <div class="eyebrow"><i class="bi bi-check2-circle"></i> ready</div>
                <h2 style="margin-top:0.8rem;">Cleaning and validation complete</h2>
                <p style="color:var(--muted);">
                    Your cleaned dataset is ready. You can download it now or open the dedicated logs page for a full review.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.download_button(
                "Download Cleaned CSV",
                data=st.session_state.final_csv_bytes,
                file_name=f"cleaned_{dataset_name}",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            if st.button("Start New Run", use_container_width=True):
                reset_workspace()
                st.session_state.nav = "clean"
                st.rerun()

        if run_error:
            st.warning(f"Run finished with a guardrail message: {run_error}")

    render_node_progress(progress_placeholder)


def render_logs_page() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow"><i class="bi bi-journal-text"></i> logs and trace</div>
            <h1 class="hero-title" style="max-width:11ch;">Review everything the agent did.</h1>
            <p class="hero-subtitle">
                This page keeps the detailed execution trace separate from the main cleaning workspace so users can
                inspect the run without cluttering the core flow.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.logs and not st.session_state.node_updates:
        st.markdown(
            '<div class="workspace-card"><p class="logs-empty">No logs available yet. Run a cleaning workflow first.</p></div>',
            unsafe_allow_html=True,
        )
        return

    st.subheader("Full Process Log")
    st.code(build_readable_logs(st.session_state.logs), language="text")

    if st.session_state.final_csv_bytes is not None and st.session_state.source_name:
        st.download_button(
            "Download Latest Cleaned CSV",
            data=st.session_state.final_csv_bytes,
            file_name=f"cleaned_{st.session_state.source_name}",
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state.run_error:
        st.warning(f"latest run error: {st.session_state.run_error}")


initialise_session()
render_sidebar()

if st.session_state.nav == "landing":
    render_landing_page()
elif st.session_state.nav == "clean":
    render_cleaning_workspace()
else:
    render_logs_page()
