# Agentic Data Cleaning

An enterprise-grade data cleaning system powered by agentic AI. This application leverages LangGraph and large language models to automate data quality assessment, strategy generation, and iterative data transformation on tabular datasets.

##Live Link - LINK[https://clean-ai-v1.streamlit.app/]

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Tool Reference](#tool-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

This system implements a **reasoning-driven agentic workflow** that automates data cleaning with human oversight. Instead of static scripts, the agent:

1. Analyzes raw data and identifies quality issues
2. Understands business context and cleaning purpose
3. Generates a tailored cleaning strategy
4. Executes transformations with human approval
5. Validates results iteratively until quality standards are met

Every action is logged, every transformation is reversible, and human review gates prevent unintended data loss.

---

## System Architecture

### Workflow Diagram

```
                              START
                                ↓
                    ┌──────────────────────┐
                    │  ANALYZER NODE       │
                    │  (Data Audit)        │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  PLANNER NODE        │
                    │  (Strategy Design)   │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  HUMAN REVIEW NODE   │
                    │  (Approval Gate)     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  EXECUTOR NODE       │
                    │  (Execute Tools)     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  TOOLS NODE          │
                    │  (Pandas Operations) │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  VALIDATOR NODE      │
                    │  (Quality Check)     │
                    └─────┬────────────┬───┘
                          │            │
                   Issues Found    Data Clean
                          ↓            ↓
                      LOOP BACK    OUTPUT
                      TO EXECUTOR  GENERATION
                                     ↓
                                   END
```

### Node Overview

**Analyzer Node**: Performs comprehensive data audit. Identifies null patterns, duplicates, type inconsistencies, and column roles. Generates validation profile for quality checks.

**Planner Node**: Interprets analysis and generates prioritized cleaning strategy based on user context (ML training, BI analytics, compliance, etc.). Recommends specific tools and order of operations.

**Human Review Node**: Safety checkpoint where user approves the cleaning plan. Allows modifications before any transformations execute.

**Executor Node**: Orchestrates tool execution. Reads current data state and plan, selects appropriate tools, invokes transformations, logs all actions.

**Tools Node**: Execution environment housing all data transformation utilities (imputation, deduplication, type conversion, etc.).

**Validator Node**: Quality assurance layer. Checks for remaining nulls, duplicates, type consistency, and semantic validity. Routes back to executor if issues persist or stops if data is clean.

---

## How It Works

1. **Upload Data**: User uploads CSV or Excel file and provides context (purpose, constraints, rules)

2. **Automated Audit**: Analyzer examines data without modification, identifies issues, infers column roles

3. **Generate Plan**: Planner designs a step-by-step cleaning strategy based on findings and user context

4. **User Approval**: Human reviews and approves plan; modifications allowed before execution

5. **Execute Cleaning**: Executor runs approved steps using available tools, logs each action

6. **Iterative Validation**: Validator checks cleaned data. If issues remain, executor re-runs targeted cleaning. Loop continues until data is clean or max rounds reached

7. **Download Results**: User receives cleaned CSV, detailed logs, and validation report

---

## Key Features

**Automated Deep Audit**  
Instantly identifies data quality issues: nulls, duplicates, type inconsistencies, and semantic anomalies without manual inspection.

**Context-Aware Planning**  
Agent understands cleaning purpose (ML training, BI analytics, compliance auditing) and adjusts strategy accordingly.

**Human-in-the-Loop Safety**  
No transformations execute without explicit user approval. Review phase allows plan modification before execution.

**Comprehensive Tool Ecosystem**  
Seven specialized tools cover 90%+ of real-world cleaning scenarios: missing value imputation, deduplication, type conversion, outlier detection, text normalization, categorical encoding, custom pandas code execution.

**Iterative Validation**  
Built-in feedback loop validates cleaned data against original requirements. Agent automatically re-executes targeted cleaning if issues remain.

**Complete Audit Trail**  
Every action logged with timestamp, tool name, parameters, rows affected, and outcome. Transparent tracking of all transformations.

**Deterministic Quality Checks**  
Validation is rule-based: null detection, duplicate detection, type consistency, email format validation, range enforcement, key uniqueness.

**Enterprise Safety**  
Stagnation detection (stops if no progress), round limits (max 6 tool rounds, max 6 validation rounds), error recovery, complete state history for rollback.

---

## Technology Stack

- **LangGraph**: Agentic workflow orchestration and state management
- **LangChain**: LLM integration and tool abstraction
- **Groq API**: High-performance LLM inference (Llama 3.3-70b)
- **Pandas & NumPy**: Data manipulation and validation
- **Streamlit**: Interactive web interface
- **Python 3.9+**

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Groq API key (free tier available at [console.groq.com](https://console.groq.com))

### Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/sathvik89/agentic-data-cleaning.git
   cd agentic-data-cleaning
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python3 -c "import pandas, langchain, langgraph, groq; print('✓ All dependencies installed')"
   ```

### requirements.txt
```
pandas>=2.0.0
numpy>=1.24.0
langchain>=0.1.0
langgraph>=0.0.1
langchain-groq>=0.1.0
streamlit>=1.28.0
python-dotenv>=1.0.0
```

---

## Configuration

### Environment Variables

Create a `.env` file in project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile
```

### Obtaining Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Navigate to "API Keys" section
4. Create and copy your API key
5. Paste into `.env` file

### .gitignore
```
.env
__pycache__/
*.pyc
venv/
.DS_Store
```

---

## Usage

### Starting the Application

```bash
streamlit run app.py
```

Access the interface at `http://localhost:8501`

### Basic Workflow

1. **Upload File**: Select CSV or Excel file (max 200MB)
2. **Provide Context**: Describe the purpose and any business rules
3. **Review Analysis**: Examine identified data quality issues
4. **Approve Plan**: Review and approve the cleaning strategy
5. **Monitor Execution**: Watch real-time progress logs
6. **Download Results**: Get cleaned data, logs, and validation report

### Example Context

```
I need this customer data for an ML pipeline to predict churn.
Keep only customers with age >= 18.
Email and phone must be valid.
Preserve all other columns.
```

---

## Tool Reference

### smart_handle_missing()
Handles missing values with multiple strategies: drop rows, fill with constant, mean, median, mode, or interpolation. Supports single or multiple columns.

### rigorous_remove_duplicates()
Removes exact duplicate rows or duplicates based on specific columns. Keeps first occurrence.

### smart_type_conversion()
Converts columns to specified types (int, float, string, datetime, categorical) with error handling. Invalid values coerced to null.

### detect_outliers_report()
Identifies statistical outliers using IQR method without modifying data. Useful for analysis before removal decisions.

### perform_text_cleaning()
Normalizes text data: lowercase, strip whitespace, remove special characters. Chainable actions.

### handle_categorical()
Encodes categorical variables with label encoding (integer codes) or one-hot encoding (dummy variables) for ML pipelines.

### execute_custom_pandas()
Executes arbitrary pandas code in a safe sandbox for complex transformations not covered by standard tools.

---

## Troubleshooting

### API Key Not Found
**Error**: `GROQ_API_KEY environment variable not set`

**Solution**: Create `.env` file with your API key and restart the application.

---

### File Too Large (> 200MB)
**Error**: `Uploaded file exceeds 200MB limit`

**Solution**: Split file into smaller chunks or preprocess externally.

---

### Agent Reaches Max Iterations
**Error**: `Validation Report (Round 6): Issues still detected but max_validation_rounds reached`

**Solution**: Review remaining issues in validation_feedback. Some issues may require manual intervention or adjustment of cleaning strategy.

---
 
## Project Structure
 
```
clean_ai/
├── .env                      # Environment variables (API keys, config)
├── .gitignore               # Git ignore rules
├── app.py                   # Streamlit UI entry point
├── graph.py                 # LangGraph workflow and state management
├── main.py                  # Main execution logic
├── nodes.py                 # Agent node implementations
├── prompts.py               # LLM prompts for analyzer, planner, executor
├── state.py                 # AgentState TypedDict definition
├── tools.py                 # Data transformation and validation tools
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── sample_messy_dataset.csv # Example dataset for testing
```
 
---

## Support & Resources

- **GitHub Issues**: [Report bugs and request features](https://github.com/sathvik89/agentic-data-cleaning/issues)
- **Groq API Docs**: [console.groq.com/docs](https://console.groq.com/docs)
- **LangGraph Docs**: [langchain.com/docs/langgraph](https://langchain.com/docs/langgraph)

---

**Version**: 1.0.0  
**Status**: Production Ready  
**License**: MIT
