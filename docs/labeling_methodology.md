# Labeling Methodology

## Priority Label
Sourced directly from the original dataset — assigned by human support
agents at ticket intake. Three classes: low, medium, high.

## Escalation Label (engineered)
No escalation ground truth exists in the source data. Escalation risk was
derived via rule-based logic using three signals, all required to be true:

1. Ticket type is "Incident" or "Problem" (vs. "Request"/"Change" — i.e.
   something is actually broken, not just requested)
2. Priority is marked "high" by the original agent label
3. At least one tag indicates a severe issue (Outage, Crash, Bug, Security,
   Disruption, Network, Server, Data Loss)

A ticket is labeled escalated only if all three conditions hold.

## Threshold Selection
Several rule variants were tested (ranging from 6% to 52% escalated
depending on signal combination and strictness). The final rule was chosen
to produce a realistic minority-class distribution (~21% escalated), 
consistent with real-world escalation rates, while requiring meaningful
evidence across type, priority, and tag signals rather than any single
weak signal.

## Known Limitation
This is a documented proxy label, not real historical escalation outcomes.
In a production setting, this would be replaced with actual ticket
escalation events logged by the support system.