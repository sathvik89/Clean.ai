# Agentic Data Cleaning

An enterprise-grade, stateful data cleaning system powered by agentic artificial intelligence. This application leverages LangGraph, advanced LLMs, and deterministic data validation to automate the intricate process of identifying, planning, and executing data cleaning operations on tabular datasets.

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
- [State Management](#state-management)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Traditional data cleaning approaches rely on static scripts or rule engines that lack contextual awareness. This system implements a **reasoning-driven agentic workflow** that mirrors how expert data scientists approach data preparation:

1. Analyze the dataset holistically
2. Understand the business context and cleaning objectives
3. Develop a prioritized, tailored cleaning strategy
4. Execute transformations systematically
5. Validate results iteratively until quality thresholds are met

The agent operates within a controlled, stateful environment where each decision is logged, every transformation is reversible, and human oversight is integrated at critical checkpoints.

### Core Principles

- **Goal-Aware Cleaning**: The agent understands why the data is being cleaned (ML training, BI analytics, compliance auditing, custom requirements) and adjusts its approach accordingly.
- **Iterative Refinement**: A feedback loop validates results and re-executes cleaning steps until the dataset meets quality standards or reaches iteration limits.
- **Transparency**: Every action is logged with timestamps and rationale, providing a complete audit trail.
- **Safety**: Human review gates prevent destructive operations and allow plan modification before execution.

---

## System Architecture

### Agent Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AGENTIC DATA CLEANING FLOW                         │
└─────────────────────────────────────────────────────────────────────────────┘

                                  ┌──────────────┐
                                  │    START     │
                                  └──────┬───────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │ User uploads CSV/Excel file + context   │
                    └────────────────────┬────────────────────┘
                                         │
                         ┌───────────────▼────────────────┐
                         │   ANALYZER NODE                │
                         │  (Deep Data Audit)             │
                         │                                │
                         │ • Extract head/tail            │
                         │ • Identify null patterns       │
                         │ • Detect duplicates            │
                         │ • Infer column roles           │
                         │ • Sample type consistency      │
                         │ • Build validation profile     │
                         └───────────────┬────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │ Analysis Report + Validation Profile    │
                    └────────────────────┬────────────────────┘
                                         │
                         ┌───────────────▼────────────────┐
                         │   PLANNER NODE                 │
                         │  (Strategic Planning)          │
                         │                                │
                         │ • Parse analysis findings      │
                         │ • Incorporate user context     │
                         │ • Prioritize issues            │
                         │ • Generate multi-step plan     │
                         │ • Recommend tools & order      │
                         └───────────────┬────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │      Cleaning Plan (Steps 1..N)        │
                    └────────────────────┬────────────────────┘
                                         │
                         ┌───────────────▼──────────────────┐
                         │  HUMAN REVIEW NODE               │
                         │  (Safety Checkpoint)             │
                         │                                  │
                         │ ⚠ User reviews proposed plan    │
                         │ ⚠ Approve / Modify / Reject     │
                         │ ⚠ Add custom instructions       │
                         └───────────────┬──────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │ Approved Plan + User Feedback           │
                    └────────────────────┬────────────────────┘
                                         │
          ┌──────────────────────────────▼──────────────────────────────┐
          │                    EXECUTION LOOP                           │
          │         (Runs up to max_tool_rounds iterations)             │
          │                                                             │
          │  ┌────────────────────────────────────────────────────┐   │
          │  │  EXECUTOR NODE                                     │   │
          │  │  (Hands-On Cleaning)                               │   │
          │  │                                                    │   │
          │  │  • Read current state & plan                       │   │
          │  │  • Select next cleaning step                       │   │
          │  │  • Choose appropriate tool(s)                      │   │
          │  │  • Generate tool parameters                        │   │
          │  │  • Log action rationale                            │   │
          │  └────────────────┬─────────────────────────────────┘   │
          │                   │                                       │
          │  ┌────────────────▼─────────────────────────────────┐   │
          │  │  TOOLS NODE                                      │   │
          │  │  (Pandas Execution Environment)                  │   │
          │  │                                                  │   │
          │  │  Tools Available:                               │   │
          │  │  • smart_handle_missing()    [Imputation]       │   │
          │  │  • rigorous_remove_duplicates() [Dedup]         │   │
          │  │  • smart_type_conversion()   [Type Casting]     │   │
          │  │  • detect_outliers_report()  [Outlier Analysis] │   │
          │  │  • perform_text_cleaning()   [Text Normalize]   │   │
          │  │  • handle_categorical()      [Encoding]         │   │
          │  │  • execute_custom_pandas()   [Custom Code]      │   │
          │  │                                                  │   │
          │  │  ✓ Apply transformation to current_df            │   │
          │  │  ✓ Update work_summary with details              │   │
          │  │  ✓ Return result to executor                     │   │
          │  └────────────────┬─────────────────────────────────┘   │
          │                   │                                       │
          │  ┌────────────────▼─────────────────────────────────┐   │
          │  │  Check: More tools needed?                       │   │
          │  │  • YES → Loop to Executor (repeat cycle)         │   │
          │  │  • NO  → Proceed to Validator                    │   │
          │  └────────────────┬─────────────────────────────────┘   │
          │                   │                                       │
          └───────────────────┼──────────────────────────────────────┘
                              │
                ┌─────────────▼──────────────┐
                │   VALIDATOR NODE           │
                │  (Quality Assurance)       │
                │                            │
                │ • Re-audit cleaned data    │
                │ • Check remaining nulls    │
                │ • Detect remaining dups    │
                │ • Validate column roles    │
                │ • Check semantic rules     │
                │ • Validate key uniqueness  │
                │ • Generate issue report    │
                └────────┬──────────────┬────┘
                         │              │
        ┌────────────────┘              └────────────────┐
        │                                                │
        │ Issues Remain + Rounds < Max               No Issues OR Max Rounds
        │                                                │
    ┌───▼──────┐                                  ┌─────▼──────┐
    │LOOP BACK │  (Re-execute cleaning)          │   FINALIZE │
    │TO         │  (Increment validation_rounds)  │   OUTPUT   │
    │EXECUTOR  │                                  │            │
    └──────────┘                                  └─────┬──────┘
                                                        │
                                    ┌───────────────────▼──────────────┐
                                    │  OUTPUT GENERATION NODE          │
                                    │  (Prepare Deliverables)          │
                                    │                                  │
                                    │  • Cleaned CSV file              │
                                    │  • Cleaning logs (TXT)           │
                                    │  • Validation report (JSON)      │
                                    │  • Transformation summary (TXT)  │
                                    │  • Data quality metrics          │
                                    └───────────────────┬──────────────┘
                                                        │
                                    ┌───────────────────▼──────────────┐
                                    │   Download to User               │
                                    │   • All artifacts in ZIP         │
                                    │   • Audit trail included         │
                                    │   • Ready for downstream use     │
                                    └───────────────────┬──────────────┘
                                                        │
                                    ┌───────────────────▼──────────────┐
                                    │        END                       │
                                    └──────────────────────────────────┘
```

### Node Descriptions

#### **Analyzer Node**
Performs a comprehensive audit of the raw dataset without modification.

**Responsibilities:**
- Extract dataset dimensions and column schema
- Identify null value patterns and hotspots
- Detect duplicate rows and key column duplicates
- Infer column roles (numeric, categorical, datetime, email, text)
- Build a validation profile for later quality checks
- Generate a compact, token-efficient data summary

**Output:** `analysis_report`, `validation_profile`, `audit_report`

---

#### **Planner Node**
Interprets the analysis and generates a tailored, prioritized cleaning strategy.

**Responsibilities:**
- Parse findings from the analyzer
- Incorporate user-provided context (purpose: ML training, BI analytics, compliance, custom)
- Identify root causes of data quality issues
- Prioritize issues by impact and complexity
- Design a multi-step cleaning plan with specific tool recommendations
- Include rationale for each step to guide the executor

**Output:** `cleaning_plan` (structured steps with tool names and parameters)

---

#### **Human Review Node**
A mandatory safety checkpoint where users approve or refine the cleaning strategy.

**Responsibilities:**
- Present the analysis report and cleaning plan in a readable format
- Allow users to approve the plan as-is
- Permit users to modify, add, or remove specific cleaning steps
- Collect custom instructions or business rules to incorporate
- Validate user feedback before proceeding to execution

**Output:** `cleaning_plan` (finalized), `user_feedback` (optional modifications)

---

#### **Executor Node**
The "hands" of the agent, orchestrating tool invocations based on the approved plan.

**Responsibilities:**
- Maintain execution state and track completed steps
- Select the next cleaning step from the plan
- Determine required tool(s) and parameter values based on current data state
- Invoke the Tools Node to apply transformations
- Update `work_summary` with action descriptions and rationale
- Log each tool invocation with timestamp and outcome
- Decide whether additional tools are needed or validation should proceed

**Output:** `current_df` (transformed), `work_summary` (appended logs), `logs` (detailed records)

---

#### **Tools Node**
A specialized execution environment housing all data transformation utilities.

**Available Tools:**

1. **smart_handle_missing()**
   - Handles null/missing values
   - Strategies: drop rows, fill with constant, mean, median, mode, interpolate
   - Supports single or multiple columns
   - Handles numeric, datetime, and categorical data intelligently

2. **rigorous_remove_duplicates()**
   - Removes exact duplicate rows
   - Optional: remove duplicates based on specific columns
   - Preserves first occurrence by default
   - Logs rows removed

3. **smart_type_conversion()**
   - Converts columns to specified data types
   - Supports: int, float, string, datetime, categorical
   - Uses coercion with error handling (invalid values become null)
   - Preserves data integrity during conversion

4. **detect_outliers_report()**
   - Identifies outliers using IQR (Interquartile Range) method
   - Threshold: values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
   - Returns summary report without modifying data
   - Useful for analysis before removal decisions

5. **perform_text_cleaning()**
   - Applies text transformations to specified columns
   - Actions: lowercase, strip whitespace, remove special characters
   - Chainable: multiple actions per column
   - Handles non-string data gracefully

6. **handle_categorical()**
   - Encodes categorical variables for modeling
   - Methods: label encoding (int codes), one-hot encoding (dummy variables)
   - Configurable per column
   - Suitable for ML pipeline integration

7. **execute_custom_pandas()**
   - Runs arbitrary pandas code in a safe sandbox environment
   - Useful for complex transformations not covered by standard tools
   - User-provided code executed with df, pd, and np available
   - Error messages returned if code fails

**Tool Selection Logic:**
- Executor analyzes current data state and plan
- Selects tools based on issue type and priority
- May invoke multiple tools in sequence (e.g., convert type, then impute, then validate)
- Each tool call is logged with parameters and results

---

#### **Validator Node**
Quality assurance layer that checks cleaned data against original requirements.

**Validation Checks:**

1. **Null Value Validation**
   - Detects remaining missing values
   - Reports hotspots by column
   - Compares against acceptable thresholds

2. **Duplicate Validation**
   - Checks for exact duplicate rows
   - Validates key column uniqueness
   - Reports duplicate counts per column

3. **Type Consistency Validation**
   - Detects mixed types within single columns
   - Validates semantic types (numeric, datetime, email, categorical)
   - Flags inconsistencies for re-processing

4. **Profile-Based Validation**
   - Validates each column against inferred role
   - Numeric columns: checks for invalid/out-of-range values
   - Datetime columns: validates format and range
   - Email columns: validates format using regex
   - Age columns: checks 0-120 range
   - Non-negative numeric: flags negative values
   - Past datetime: flags future dates

5. **Stagnation Detection**
   - Compares current state to previous validation signature
   - If no progress detected: stops early to avoid infinite loops
   - Logs stagnation and returns current best state

**Decision Logic:**
- If no issues remain → data is clean, proceed to output generation
- If issues remain AND validation_rounds < max → re-route to executor
- If issues remain AND validation_rounds ≥ max → return best-effort cleaned data

**Output:** `is_clean` (boolean), `validation_feedback` (issue list)

---

### State Management

The system uses a **TypedDict-based AgentState** for complete workflow tracking.

**Key State Fields:**

```python
# Input & Context
file_path: str                          # Original uploaded file path
user_context: str                       # User's purpose/instructions

# Data Tracking
current_df: pd.DataFrame                # Active working dataframe
df_history: List[pd.DataFrame]          # Append-only history of states

# Analysis & Planning
analysis_report: str                    # Analyzer's findings
cleaning_plan: str                      # Planner's strategy
audit_report: Dict[str, Any]            # Structured audit results
validation_profile: Dict[str, Any]      # Column role definitions

# Execution Tracking
work_summary: str                       # Rolling log of all actions
validation_feedback: str                # Validator's issue report
run_error: str                          # Error messages if any

# Flow Control
is_clean: bool                          # Data quality flag
tool_rounds: int                        # Current execution iteration
max_tool_rounds: int                    # Limit (default: 6)
validation_rounds: int                  # Current validation iteration
max_validation_rounds: int              # Limit (default: 6)
validation_signature: str               # Hash for stagnation detection
stagnation_rounds: int                  # Stagnation counter
max_stagnation_rounds: int              # Limit (default: 2)

# Logging
logs: List[str]                         # Append-only action log
messages: List                          # Conversation history
```

**State Updates:**
- Append-only logs maintain immutable audit trail
- DataFrame history tracks all intermediate states
- Custom reducers (operator.add, custom update_logs) enable safe list concatenation
- No destructive edits—rollback is always possible

---

## How It Works

### End-to-End Flow

#### **Step 1: Data Ingestion**
User uploads a CSV or Excel file through the Streamlit interface and provides context:
- **Purpose**: "I need this data for machine learning model training"
- **Constraints**: "Keep rows where age > 18"
- **Business Rules**: "Email must be valid and unique"

#### **Step 2: Automated Audit (Analyzer Node)**
The agent examines the raw data:
```
Dataset: rows=150000, columns=24
Columns with nulls:
  - email: 342 missing (0.2%)
  - phone: 12500 missing (8.3%)
  - age: 1250 missing (0.8%)
Duplicate rows: 3200
Key column duplicates:
  - email: 450 duplicates
  - user_id: 0 duplicates
Mixed-type columns detected:
  - salary (int|float|str): likely parsing errors
  - join_date (str): inconsistent formats
```

#### **Step 3: Strategic Planning (Planner Node)**
Based on analysis and context, the agent designs a plan:
```
Cleaning Plan:
1. Remove duplicate rows (3200 total duplicates)
2. Handle missing email (342): Drop rows (unique key constraint)
3. Handle missing phone (12500): Fill with "Unknown"
4. Handle missing age (1250): Fill with median value
5. Convert salary to float (fix mixed types)
6. Parse join_date to datetime (standardize format)
7. Validate email format and uniqueness
8. Final deduplication on email column
9. Export cleaned dataset
```

#### **Step 4: Human Review (Human Review Node)**
User sees the plan and can:
- ✓ Approve as-is
- ✓ Modify steps (e.g., "Don't drop phone, impute with mode instead")
- ✓ Add rules (e.g., "Remove rows where salary < 0")

#### **Step 5: Execution with Iteration (Executor + Tools Loop)**
Agent executes approved plan:

**Round 1:**
```
Tool: rigorous_remove_duplicates() on entire dataframe
  Result: Removed 3200 rows → 146800 rows remain
  
Tool: smart_handle_missing('drop', columns=['email'])
  Result: Dropped 342 rows → 146458 rows remain
  
Tool: smart_handle_missing('constant', columns=['phone'], fill_value='Unknown')
  Result: Filled 12500 null cells with 'Unknown'
  
Tool: smart_handle_missing('median', columns=['age'])
  Result: Filled 1250 cells with median age 35
```

**Round 2 (if needed):**
```
Tool: smart_type_conversion({
  'salary': 'float',
  'join_date': 'datetime'
})
  Result: Converted types, coerced invalid values to null
  
Tool: detect_outliers_report(['salary', 'age'])
  Report: salary has 1200 outliers; age has 50 outliers
  (No removal, just reporting for user awareness)
```

#### **Step 6: Validation & Iteration (Validator Loop)**
After each execution round, validator checks:
```
Validation Report (Round 1):
✓ No duplicate rows
✗ Email column still has 0 nulls (good)
✓ All rows have valid emails
✗ Datetime column has 340 parsing errors
  → Re-route to Executor for another cleaning round

Validation Report (Round 2):
✓ No duplicate rows
✓ No null values
✓ All datetimes valid
✓ All emails valid & unique
✓ Data is CLEAN
```

#### **Step 7: Output Generation & Download**
System packages cleaned data and artifacts:
```
cleaned_data.csv          ← Final cleaned dataset
cleaning_logs.txt         ← Detailed action log
data_quality_report.json  ← Metrics and validation results
transformation_summary.txt ← Human-readable summary
```

---

## Key Features

### **Automated Deep Audit**
Instant comprehensive assessment of data quality without manual inspection scripts. Identifies null patterns, duplicates, type inconsistencies, and semantic anomalies in seconds.

### **Context-Aware Planning**
The agent understands the purpose of cleaning and adjusts strategy accordingly:
- **ML Training Data**: Prioritizes consistency and removes null-heavy rows
- **BI/Analytics**: Preserves as much data as possible, uses aggregation
- **Compliance Auditing**: Enforces strict rules and documents everything
- **Custom Rules**: Incorporates user-provided business constraints

### **Human-in-the-Loop Safeguard**
No transformation occurs without explicit user approval. Review phase allows plan modification before execution, preventing unintended data loss.

### **Comprehensive Tool Ecosystem**
Seven specialized tools cover 90%+ of real-world cleaning scenarios:
- Missing value imputation with 6 strategies
- Deduplication with column-level control
- Type conversion with coercion safety
- Outlier analysis and reporting
- Text normalization with multiple actions
- Categorical encoding (label and one-hot)
- Custom pandas code execution for edge cases

### **Iterative Validation & Auto-Correction**
Built-in feedback loop validates cleaned data against original requirements. If issues remain, the agent re-executes targeted cleaning steps without human intervention (up to max iterations).

### **Complete Audit Trail**
Every action is logged with:
- Timestamp of execution
- Tool name and parameters
- Rows/cells affected
- Rationale and outcome
- Pre/post state comparison

Users can always understand exactly what happened to their data.

### **Deterministic Quality Checks**
Validation is rule-based, not heuristic:
- Null detection is exact
- Duplicate detection is exact
- Type consistency is verified per semantic role
- Email format validated with regex
- Age/date ranges enforced per business logic
- Key uniqueness verified by definition

### **Enterprise Safety Features**
- **Stagnation Detection**: Stops if no progress after 2 validation rounds
- **Round Limits**: Tool execution capped at 6 rounds, validation at 6 rounds
- **Error Recovery**: Exceptions logged; best-effort state returned
- **Reversibility**: Complete df_history allows rollback if needed
- **File Size Limits**: Supports up to 200MB files

---

## Technology Stack

### **Core Libraries**
- **LangGraph**: Agentic workflow orchestration, state management, and node routing
- **LangChain**: Integration layer for LLM communication and tool abstraction
- **Groq API**: High-performance LLM inference using Llama 3.3-70b
- **Pandas**: Efficient data manipulation, type handling, and validation
- **NumPy**: Numerical operations (outlier detection, statistics)

### **User Interface**
- **Streamlit**: Interactive web interface for file upload, progress monitoring, and output download

### **Python Version**
- Python 3.9+

### **Deployment**
- Streamlit Cloud, AWS, GCP, or self-hosted server

---

## Installation

### **Prerequisites**
- Python 3.9 or higher
- pip package manager
- Groq API key (free tier available)

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/agentic-data-cleaning.git
cd agentic-data-cleaning
```

### **Step 2: Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Verify Installation**
```bash
python -c "import pandas, langchain, langraph, groq; print('✓ All dependencies installed')"
```

### **requirements.txt**
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

### **Environment Variables**

Create a `.env` file in the project root:

```env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile

# Optional: Streamlit Config
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=false
```

### **Obtaining a Groq API Key**

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Navigate to "API Keys"
4. Create a new API key
5. Copy and paste into `.env`

### **Configuration Best Practices**

- **Never commit `.env` to version control**
- Store API keys in environment variables only
- Use `.gitignore` to exclude sensitive files:
  ```
  .env
  __pycache__/
  *.pyc
  .DS_Store
  venv/
  ```

---

## Usage

### **Starting the Application**

```bash
streamlit run app.py
```

The interface will be available at `http://localhost:8501`

### **Basic Workflow**

#### **1. Upload Data**
- Click "Upload CSV or Excel" button
- Select your file (max 200MB)
- File is validated and loaded into memory

#### **2. Provide Context**
In the text area, describe:
- The purpose of cleaning (ML training, BI, compliance, etc.)
- Any specific business rules or constraints
- Columns to prioritize or preserve

Example:
```
I need this customer data for an ML pipeline to predict churn.
Purpose: Machine learning model training
Rules:
- Keep only customers with age >= 18
- Email and phone_number must be valid
- Remove accounts created less than 6 months ago
- Preserve all other columns
```

#### **3. Review Analysis**
- Analyzer produces a detailed report
- Review null patterns, duplicates, and type issues
- Understand the current state before cleaning

#### **4. Approve Cleaning Plan**
- Planner generates a multi-step strategy
- Review each proposed step
- Modify steps if needed
- Click "Approve & Proceed" to start cleaning

#### **5. Monitor Execution**
- Watch real-time progress logs
- See each tool invocation and result
- Track rows affected and data state changes

#### **6. Download Results**
After validation succeeds:
- Download cleaned CSV file
- Get detailed cleaning logs (TXT)
- Review data quality report (JSON)
- Check transformation summary

### **Advanced Usage**

#### **Custom Business Rules**
Include in context:
```
Custom rules:
- Salary must be >= minimum_wage (adjust for region)
- Email must match company domain
- Age between 18 and 75
- Department must be one of: Sales, Engineering, HR, Marketing
```

#### **Custom Data Transformations**
If standard tools don't suffice, the executor can invoke custom pandas code:
```
# Plan includes:
Tool: execute_custom_pandas()
Code: df['full_name'] = df['first_name'] + ' ' + df['last_name']
```

---

## Tool Reference

### **smart_handle_missing()**

**Purpose**: Fill or remove null/missing values

**Signature**:
```python
def smart_handle_missing(
    df: pd.DataFrame,
    strategy: str,
    columns: Optional[List[str]] = None,
    fill_value: Any = None
) -> pd.DataFrame
```

**Strategies**:
- `'drop'`: Remove rows with nulls in specified columns
- `'constant'`: Fill with a constant value (default: 0 for numeric, "Unknown" for text)
- `'mean'`: Fill numeric columns with mean
- `'median'`: Fill numeric columns with median (recommended for outlier-prone data)
- `'mode'`: Fill with most frequent value
- `'interpolate'`: Linear interpolation for time series or numeric data

**Example**:
```python
# Fill missing age with median
df = smart_handle_missing(df, 'median', columns=['age'])

# Drop rows missing email (unique key)
df = smart_handle_missing(df, 'drop', columns=['email'])

# Fill phone with constant
df = smart_handle_missing(df, 'constant', columns=['phone'], fill_value='000-0000')
```

---

### **rigorous_remove_duplicates()**

**Purpose**: Remove duplicate rows based on all or specific columns

**Signature**:
```python
def rigorous_remove_duplicates(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame
```

**Behavior**:
- If `columns=None`: Remove exact duplicates (all columns match)
- If `columns=['col1', 'col2']`: Remove rows where these columns match
- Keeps first occurrence, removes subsequent duplicates
- Returns clean copy without modifying input

**Example**:
```python
# Remove exact duplicate rows
df = rigorous_remove_duplicates(df)

# Remove rows with duplicate emails (keep first)
df = rigorous_remove_duplicates(df, columns=['email'])
```

---

### **smart_type_conversion()**

**Purpose**: Convert columns to specified data types with error handling

**Signature**:
```python
def smart_type_conversion(
    df: pd.DataFrame,
    column_type_map: Dict[str, str]
) -> pd.DataFrame
```

**Supported Types**:
- `'int'`: Integer (null-safe with Int64)
- `'float'`: Floating point
- `'string'`: Text (nullable string type)
- `'datetime'`: Timestamp
- `'categorical'`: Categorical (for grouping/encoding)

**Behavior**:
- Invalid values coerced to null (not dropped)
- Preserves datetime formats with `format='mixed'`
- Uses pandas nullable types for type safety

**Example**:
```python
type_map = {
    'age': 'int',
    'salary': 'float',
    'join_date': 'datetime',
    'department': 'categorical'
}
df = smart_type_conversion(df, type_map)
```

---

### **detect_outliers_report()**

**Purpose**: Identify statistical outliers without modification (reporting only)

**Signature**:
```python
def detect_outliers_report(
    df: pd.DataFrame,
    columns: List[str]
) -> str
```

**Method**: IQR (Interquartile Range)
- Outliers: values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR
- Returns summary report, does not modify data
- Useful for decision-making before removal

**Example**:
```python
report = detect_outliers_report(df, columns=['salary', 'age'])
print(report)
# Output: "salary: 1200 outliers; age: 50 outliers; ..."
```

---

### **perform_text_cleaning()**

**Purpose**: Normalize text data (lowercase, whitespace, special characters)

**Signature**:
```python
def perform_text_cleaning(
    df: pd.DataFrame,
    columns: List[str],
    actions: List[str]
) -> pd.DataFrame
```

**Available Actions**:
- `'lowercase'`: Convert to lowercase
- `'strip'`: Remove leading/trailing whitespace
- `'remove_special'`: Remove non-alphanumeric characters

**Behavior**:
- Actions applied in sequence
- Non-string values coerced to string first
- Handles null values gracefully

**Example**:
```python
df = perform_text_cleaning(
    df,
    columns=['first_name', 'last_name', 'email'],
    actions=['lowercase', 'strip', 'remove_special']
)
```

---

### **handle_categorical()**

**Purpose**: Encode categorical variables for machine learning

**Signature**:
```python
def handle_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'label'
) -> pd.DataFrame
```

**Methods**:
- `'label'`: Convert categories to integer codes (0, 1, 2, ...)
- `'onehot'`: Create binary columns per category (suitable for tree models)

**Example**:
```python
# Label encode department
df = handle_categorical(df, columns=['department'], method='label')

# One-hot encode region
df = handle_categorical(df, columns=['region'], method='onehot')
```

---

### **execute_custom_pandas()**

**Purpose**: Run arbitrary pandas code for complex transformations

**Signature**:
```python
def execute_custom_pandas(
    df: pd.DataFrame,
    code: str
) -> pd.DataFrame
```

**Execution Context**:
- `df`: Your dataframe (copy)
- `pd`: Pandas module
- `np`: NumPy module
- Code executed in isolated namespace for safety

**Example**:
```python
custom_code = """
df['full_name'] = df['first_name'] + ' ' + df['last_name']
df = df.drop(columns=['first_name', 'last_name'])
df['hire_year'] = df['hire_date'].dt.year
"""
df = execute_custom_pandas(df, custom_code)
```

---

## State Management

### **Understanding AgentState**

The state object maintains the complete workflow context:

```python
state = {
    # Input
    'file_path': '/uploads/customers.csv',
    'user_context': 'ML training data for churn prediction',
    
    # Current data
    'current_df': <DataFrame 145000 rows × 24 columns>,
    'df_history': [<original>, <after dedup>, <after imputation>, ...],
    
    # Analysis
    'analysis_report': '...',
    'validation_profile': {
        'email': {'role': 'email', 'is_key': True, ...},
        'age': {'role': 'numeric', 'semantic_hint': 'age', ...},
        ...
    },
    
    # Execution
    'work_summary': 'Step 1: Removed 3200 duplicate rows | Step 2: ...',
    'logs': [
        '[10:30:45] Analyzer started',
        '[10:31:12] Found 342 null emails',
        ...
    ],
    
    # Control
    'is_clean': False,
    'tool_rounds': 2,
    'max_tool_rounds': 6,
    'validation_rounds': 1,
    'max_validation_rounds': 6,
}
```

### **State Immutability Guarantees**

- **Logs**: Append-only list (uses `update_logs` reducer)
- **DataFrame History**: Append-only (uses `operator.add`)
- **Messages**: Append-only conversation history
- **Current DataFrame**: Updated as new version, history preserved

This ensures:
- Complete audit trail (no lost history)
- Rollback capability (any previous state available)
- Transparency (user sees all state changes)

---

## Troubleshooting

### **Common Issues**

#### **Issue: "API key not found"**
```
Error: GROQ_API_KEY environment variable not set
```

**Solution**:
1. Create `.env` file in project root
2. Add: `GROQ_API_KEY=your_key_here`
3. Restart the application
4. Verify: `echo $GROQ_API_KEY`

---

#### **Issue: "File too large" (> 200MB)**
```
Error: Uploaded file exceeds 200MB limit
```

**Solution**:
- Split file into smaller chunks
- Use data sampling for very large datasets
- Consider cloud storage for preprocessing

---

#### **Issue: Agent enters infinite loop / max rounds exceeded**
```
Validation Report (Round 6):
Issues still detected but max_validation_rounds (6) reached.
Returning best-effort cleaned data.
```

**Solution**:
1. Review validation_feedback for remaining issues
2. Manually address specific issues if needed
3. Increase `max_validation_rounds` in config (if stagnation isn't detected)
4. Check tool parameters in cleaning plan

---

#### **Issue: "Tool execution failed"**
```
Error in executor: smart_handle_missing() raised ValueError
```

**Solution**:
1. Check logs for specific error message
2. Verify column names exist in dataframe
3. Ensure fill_value is compatible with column type
4. Try with different strategy (e.g., 'median' instead of 'mean')

---

#### **Issue: Type conversion fails for datetime column**
```
Error: Cannot convert 'join_date' to datetime
```

**Possible Causes**:
- Mixed date formats (some "2024-01-15", others "01/15/2024")
- Invalid dates (e.g., "2024-13-45")
- Text noise (e.g., " 2024-01-15 ")

**Solution**:
1. Use `perform_text_cleaning()` with 'strip' action first
2. Smart_type_conversion already uses `format='mixed'` for flexibility
3. Check detected outliers in date columns
4. Use custom pandas code to handle edge cases:
   ```python
   custom = """
   df['join_date'] = pd.to_datetime(
       df['join_date'].astype(str).str.strip(),
       errors='coerce',
       format='mixed'
   )
   """
   ```

---

#### **Issue: Email validation too strict**
```
Validation Report:
email column still has 120 invalid values
```

**Solution**:
Email validation uses regex: `^[^@\s]+@[^@\s]+\.[^@\s]+$`

If false positives occur, use custom code:
```python
custom = """
import re
email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
df = df[df['email'].apply(lambda x: bool(email_pattern.match(x)) if pd.notna(x) else True)]
"""
```

---

#### **Issue: Duplicate detection on key column isn't working**
```
Key column 'user_id' still has 50 duplicate values
```

**Possible Causes**:
- Column contains nulls (nulls aren't compared)
- Data type mismatch (e.g., "123" ≠ 123)

**Solution**:
1. Check column data type: `df['user_id'].dtype`
2. Handle nulls first: `smart_handle_missing('drop', columns=['user_id'])`
3. Convert to consistent type: `smart_type_conversion({'user_id': 'int'})`
4. Then deduplicate: `rigorous_remove_duplicates(df, columns=['user_id'])`

---

### **Performance Optimization**

#### **Large Datasets (100MB+)**
- Agent automatically uses token-efficient summarization
- Avoid custom pandas code for complex operations
- Consider preprocessing with external tools first

#### **Slow Execution**
- Check Groq API status and rate limits
- Reduce `max_tool_rounds` if stagnated
- Profile bottleneck: Analyzer > Planner > Executor > Validator

---

### **Debug Mode**

Enable verbose logging:
```python
# In app.py or main execution script
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('langgraph')
logger.setLevel(logging.DEBUG)
```

This shows:
- Every node execution
- State transitions
- Tool invocations with parameters
- LLM prompts and responses

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### **Reporting Issues**
1. Check existing issues first
2. Provide reproducible example with sample data
3. Include error messages and logs
4. Specify Python version and OS

### **Proposing Features**
1. Open a discussion issue
2. Explain use case and benefits
3. Link related issues
4. Await feedback before implementing

### **Code Contributions**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Follow code style: lowercase comments, clear variable names
4. Add tests for new tools or features
5. Update documentation
6. Submit pull request with clear description

### **Code Standards**
- **Python Style**: PEP 8 compliant
- **Comments**: Lowercase, descriptive, explain "why" not "what"
- **Type Hints**: Use TypedDict and proper type annotations
- **Error Handling**: Graceful degradation, informative messages
- **Testing**: Unit tests for tools and nodes

---

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## Project Structure

```
agentic-data-cleaning/
├── app.py                    # Streamlit UI entry point
├── agents/
│   ├── __init__.py
│   ├── state.py              # AgentState definition
│   ├── nodes.py              # Node implementations
│   └── graph.py              # LangGraph workflow
├── tools/
│   ├── __init__.py
│   ├── pandas_tools.py       # Data transformation tools
│   └── validation_tools.py   # Validation & audit tools
├── prompts/
│   ├── analyzer_prompt.py
│   ├── planner_prompt.py
│   └── executor_prompt.py
├── utils/
│   ├── __init__.py
│   ├── llm.py                # Groq API client
│   └── logger.py             # Logging configuration
├── .env.example              # Example environment config
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE                   # MIT License
```

---

## Support & Resources

- **Documentation**: [Read the Docs](https://your-docs-url.com)
- **Issues**: [GitHub Issues](https://github.com/yourusername/agentic-data-cleaning/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/agentic-data-cleaning/discussions)
- **Groq API Docs**: [console.groq.com/docs](https://console.groq.com/docs)
- **LangGraph Docs**: [langchain.com/docs/langgraph](https://langchain.com/docs/langgraph)

---

## Acknowledgments

Built with:
- **LangChain/LangGraph**: Agentic AI framework
- **Groq**: High-performance LLM inference
- **Pandas & NumPy**: Data manipulation
- **Streamlit**: Interactive UI

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready