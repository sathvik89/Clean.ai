Clean.ai

Goal-Aware Agentic Data Cleaning System

Overview

Clean.ai is an intelligent data cleaning system that enables users to prepare datasets based on their intended use case. Instead of relying on static rules or manual scripts, the system understands user intent and generates a structured, step-by-step cleaning plan.

The goal is to make data cleaning accessible to both technical and non-technical users by automating decision-making while maintaining control and transparency.

Problem Statement

Data cleaning is not a one-size-fits-all process.
Different use cases require different approaches:

Machine Learning requires handling missing values, encoding, and scaling
Business Intelligence prioritizes readability and consistency
Auditing requires preserving all original records

Traditional tools do not adapt to these varying needs. Clean.ai addresses this gap by making the process intent-driven.

Key Idea

Clean.ai follows a structured workflow:

Intent → Plan → Execute → Validate

The system acts as a decision-making agent that:

Understands the user’s goal
Analyzes the dataset
Generates a cleaning strategy
Executes transformations through controlled tools
Validates results and iterates if needed
Features
1. Intent-Based Cleaning

Users describe their goal in simple language. The system translates this into actionable cleaning objectives.

2. Metadata-Driven Analysis

The dataset is profiled to extract:

Missing values
Data types
Duplicates
Distribution characteristics

This information guides all decisions.

3. Automated Planning

The system generates a structured execution plan that includes:

Column-level transformations
Handling strategies (imputation, encoding, etc.)
Constraint enforcement
4. Human-in-the-Loop Validation

Users review and approve the plan before execution, ensuring alignment with their requirements.

5. Deterministic Execution

All cleaning operations are performed using predefined, controlled functions to ensure reliability and reproducibility.

6. Transparency and Traceability

Each transformation is recorded, making the system explainable and auditable.

System Architecture
Phase 1: Context and Intent Understanding
Capture user intent
Profile dataset
Generate objectives and constraints
Present plan for user approval
Phase 2: Planning Layer
Convert objectives into technical steps
Ensure compliance with constraints
Produce a structured execution plan
Phase 3 (Planned)
Execute cleaning steps using tools
Validate results
Iterate if necessary
Example

User Input:
"Prepare this dataset for machine learning. Do not drop rows."

System Behavior:

Detect missing values
Plan imputation strategies
Avoid row deletion
Generate step-by-step cleaning actions
Tech Stack
Python
Pandas, NumPy
Groq API (LLM inference)
LangGraph (planned for orchestration)

Vision

Clean.ai aims to bridge the gap between manual data cleaning workflows and intelligent automation. It enables users to work with data more efficiently while maintaining control, accuracy, and transparency.