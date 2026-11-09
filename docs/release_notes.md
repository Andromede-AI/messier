## MESSIER v1.0.2

This release corrects the treatment of non-scorable attempts in Harbor evaluations.

- **Harbor attempts.** Attempts without verifier rewards are retained with `result=None` instead of being omitted.
- **Failure handling.** Provider, environment, and evaluator failures are marked as non-scorable when they prevent evaluation, while agent timeouts and complete graded outputs remain scored.
- **Analysis.** The HarveyAI-Lab counterfactual analysis and derived corpus statistics are updated accordingly.
