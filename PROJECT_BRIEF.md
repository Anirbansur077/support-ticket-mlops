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
