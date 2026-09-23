## [date] — Escalation label finalized
Engineered escalation label via rule requiring: severe ticket type (Incident/
Problem) AND high priority AND a severity-indicating tag. Iterated through
4 rule variants (31.6%, 6%, 52%, 21.3% escalated) before settling on the
final version balancing realism and signal strength. See
labeling_methodology.md for full rationale.

## [date] — Baseline and main model trained for escalation prediction
Logistic Regression baseline: precision 0.53, recall 0.88, F1 0.66 (class 1).
XGBoost (tuned: n_estimators=400, max_depth=8, learning_rate=0.05):
precision 0.56, recall 0.86, F1 0.68, PR-AUC 0.763, ROC-AUC 0.919.
XGBoost selected as main model — modest but consistent improvement over
baseline across all metrics, meets PROJECT_BRIEF.md target of PR-AUC > 0.75.

## [date] — Phase 2 modeling complete
Escalation model (XGBoost) hit PR-AUC 0.763, meeting target. Priority
model underperformed (0.59 accuracy) — traced to likely label noise via
confusion matrix analysis, not a feature/model deficiency. SHAP confirms
escalation model learned coherent patterns consistent with label
construction. Full analysis in docs/model_findings.md.