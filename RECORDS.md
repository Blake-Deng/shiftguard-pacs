# Published Records

All paths are relative to the repository root.

| Record set | Path | Count |
|---|---|---:|
| Screening jobs | reference_results/corrected_screening/runs/ | 24 JSON |
| ERM | reference_results/erm/runs/ | 12 JSON |
| Mixup | reference_results/mixup/runs/ | 12 JSON |
| Source-only CORAL | reference_results/coral/runs/ | 12 JSON |
| Corrected ablation | reference_results/corrected_ablation/runs/ | 48 JSON |
| Corrected formal adaptive runs | reference_results/corrected_formal/runs/ | 12 JSON |
| Exact ViT-S/16 controls | reference_results/corrected_vit/runs/ | 24 JSON |
| Corrected robustness | reference_results/corrected_robustness/results.csv | 150 rows |

Run python summarize_reference_results.py to recompute the JSON-derived
per-domain and per-seed macro statistics.

The 24 screening JSON files record target_evaluations=0 and a null
target_accuracy. The legacy ERM, Mixup, and CORAL JSON schema does not contain
the newer target_evaluations field, so protocol claims for those baselines
must also be checked against their source code.
