# Deep audit plan

Working autonomously. Each item: verify from source data, fix, re-verify.

## A. Numerical integrity
- [ ] A1 Every number in README recomputed from results/ artifacts
- [ ] A2 Every number on the site traced to a data file (no hand-entered values)
- [ ] A3 summary.json / localisation.json / calibration.json regenerated and consistent
- [ ] A4 Figures regenerated from current data

## B. Claim integrity
- [ ] B1 No stale claims (the seed-0 ablation/localisation reversals)
- [ ] B2 Claims match the strength of their evidence (noise floors stated)
- [ ] B3 No claim contradicted by another part of the repo

## C. Code correctness
- [ ] C1 Re-examine harness: loss, selection, early stop, seeding
- [ ] C2 Re-examine model: merge, uncertainty, CAM derivation
- [ ] C3 Re-examine metrics: AUPRC/bootstrap/paired test/ECE/pointing game
- [ ] C4 Data pipeline: splits, normalisation, box coordinates

## D. Methodology
- [ ] D1 Shared-LR probe (bug fixed, never executed)
- [ ] D2 Any confound not yet documented
- [ ] D3 Multi-seed everywhere a claim depends on it

## E. Reproducibility
- [ ] E1 Clean-clone runnability
- [ ] E2 requirements complete
- [ ] E3 Tests pass from scratch

## F. Delivery
- [ ] F1 Site renders correctly, no console errors
- [ ] F2 Commit, push, Pages live
