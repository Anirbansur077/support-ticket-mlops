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
[To be finalized in Phase 1]