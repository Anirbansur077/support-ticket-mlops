# Project Brief: Support Ticket Priority & Escalation Prediction

## Problem Statement
Support teams triage tickets manually, causing inconsistent prioritization
and delayed response to high-risk tickets. This project predicts ticket
priority and escalation risk at intake to enable automated routing.

## Success Metrics
- Model: PR-AUC > 0.75 for escalation risk (imbalanced class)
- System: automated retraining pipeline that only promotes models
  beating the current production baseline
- Business framing: reduction in simulated SLA breaches vs. baseline

## Scope (In)
- Priority classification (Low/Med/High/Urgent)
- Escalation risk (binary)
- Full MLOps pipeline: tracking, CI/CD, serving, monitoring

## Non-Goals (Out of scope)
- Real-time streaming ingestion (batch simulation is enough)
- Multi-language ticket support
- Real production data (synthetic/public data only)

## Dataset Strategy
Multilingual Customer Support Tickets (Kaggle), filtered to English and
cleaned (11,922 tickets). Priority label sourced from original data;
escalation risk engineered via rule-based logic requiring severe ticket
type, high priority, and a severity tag together (see
docs/labeling_methodology.md) — ~21% positive class. Time-based 80/20
split (train: Jan 2025–Apr 2026, val: May–Aug 2026).

## Priority Model Findings
Priority classification (accuracy 0.59) performed substantially worse than
escalation classification (PR-AUC 0.763). Class weighting improved "low"
priority recall from 0.19 to 0.51 without improving overall accuracy,
redistributing errors rather than reducing them.

Confusion matrix analysis shows errors concentrated between adjacent
severity levels (low↔medium, medium↔high) rather than extreme
misclassifications (low↔high, only 160/2385 cases combined). This
suggests the model has learned a genuine severity gradient but struggles
with precise boundaries between adjacent classes — consistent with
priority being assigned by multiple human agents with likely-inconsistent
judgment at class boundaries, rather than a pure modeling or feature
limitation.

## SHAP Analysis
Top features driving escalation predictions align with the label
construction logic: severity-related queues (Technical Support, IT
Support) and tags (Outage, Crash, Security, Network, Bug) push predictions
toward "escalated," while routine ticket types (Request, Change) push
away from it. This confirms the model is learning coherent, interpretable
patterns rather than spurious correlations, though the overlap with
label-construction features means this is primarily a consistency check
rather than a novel discovery.
