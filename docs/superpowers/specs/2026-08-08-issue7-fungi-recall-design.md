# Issue #7 — Fix fungi detection (organism_type_model recall 0.08)

Date: 2026-08-08

## Problem

`organism_type_model` almost never detects fungi. On the current artifact:

- Fungi recall 0.08 (2 of 25 fungi test images), precision 1.00.
- Overall accuracy 0.817 hides the collapse because bacteria dominate the test split.
- `organism_type` is the root of the inference hierarchy, so fungi images are routed
  into bacteria-only models downstream.

## Scope (Phase 1 of the accuracy-improvement plan)

Only Phase 1 from issue #7. Phases 2 (A2: CLAHE + photometric augmentation) and 3
(A3: species-level features) are separate follow-ups.

## Approach

1. **`_oversample_train(X, y)`** — bootstrap minority classes up toward the majority
   class size, applied inside `_fit_best_ensemble_model()` so RF/ET/KNN all benefit.
   Applied to the training fold only (test fold untouched — no leakage).
2. **Per-task scoring** — add `scoring` parameter to `_fit_best_ensemble_model()`
   (default `accuracy`). `organism_type_model` is selected with `balanced_accuracy`.
   Metrics JSON records the scoring method per task.
3. **Evaluation harness** — `scripts/evaluate_models.py` + `make evaluate` prints
   per-class precision/recall and per-modality accuracy from the metrics JSON.
   Training computes and stores per-modality accuracy.
4. **CI tripwire** — model-quality step reads the metrics JSON, reports fungi
   precision/recall, fails when fungi recall < 0.10. Artifact seeding
   (`make ci-upload-artifact`) also ships the metrics JSON.

## Acceptance criteria

- Fungi recall >= 0.40 on the artifact hold-out split (baseline 0.08). If not
  reachable in Phase 1, report the achieved value and note follow-up.
- `balanced_accuracy` used for `organism_type_model` selection; metrics JSON records
  the scoring method.
- `make evaluate` prints per-class precision/recall + per-modality breakdown.
- CI model-quality step reports fungi precision/recall and fails if recall < 0.10.
- `make test` and `make lint` stay green; new unit tests cover `_oversample_train`
  and the scoring path.
