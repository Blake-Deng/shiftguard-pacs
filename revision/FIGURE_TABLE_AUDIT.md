# Figure/Table Evidence Audit

Date: 2026-09-03

Scope: revised manuscript Tables I--II and Figure 2. Figure 1 workflow is
excluded from this audit at the author's request.

## Table I: full PACS domain results

- The table reports Photo, Art Painting, Cartoon, Sketch, and macro accuracy
  for every included PACS method; it contains no missing-cell placeholders.
- Submitted PACS ERM, Mixup, source-only CORAL, One-way KL, and Adaptive
  results are marked with a dagger and use three seeds.
- Strong Augmentation, Feature+KL, MixStyle, and SWAD (epoch) use five seeds.
- Macro SD is computed after first averaging the four domains within each
  seed. Domains are not treated as independent replicates.
- Boldface identifies the highest displayed mean in each column.

Primary sources:

- runs/results.csv
- runs/mixup/results.csv
- runs/coral/results.csv
- runs/corrected_ablation/summary.json
- revision/summaries/all_methods_five_seed_summary.json
- revision/summaries/pacs_five_seed_summary.json

## Table II: complete cross-dataset comparison

- Only methods with complete PACS, VLCS, and OfficeHome entries are shown:
  ERM, Strong Augmentation, Feature+KL, MixStyle, and SWAD (epoch).
- PACS ERM retains the submitted three-seed estimate and is dagger-marked.
  All other method cells are five-seed mean and sample SD.
- The final row reports paired Feature+KL minus Strong Augmentation means and
  95% t intervals: PACS +0.38 [-0.50, 1.27], VLCS
  -0.18 [-1.64, 1.29], and OfficeHome +0.22 [-0.64, 1.07].

Primary sources:

- revision/summaries/all_methods_five_seed_summary.json
- revision/summaries/cross_dataset_five_seed_summary.json

## Figure 2

- Panel (a): the same five-seed macro means and seed SDs as Table II for
  Strong Augmentation, Feature+KL, MixStyle, and SWAD (epoch).
- Panel (b): paired seed differences and 95% t intervals from the PACS,
  cross-dataset, and ViT summary JSON files. Every interval contains zero.
- Panel (c): PACS five-seed ResNet-50 and exact ViT-S/16 direct comparisons.
- Panel (d): descriptive seed-42 one-factor sensitivity loaded directly from
  compact_sensitivity_table.csv; open markers denote fixed defaults.

## Narrative closure

- Both tables and Figure 2 are cited and interpreted in the Results section.
- Table I gives complete PACS domain evidence; Table II isolates cross-dataset
  external validity without blank cells.
- The submitted three-seed estimate is explicitly connected to the expanded
  five-seed estimate.
- Paired intervals are used to reject significance and universal-superiority
  claims.
- The sensitivity panel is explicitly single-seed and was not used to retune
  the formal configuration.
- Strict nested selection is explicitly not claimed or shown.

Audit outcome: PASS. No displayed value differs from its recorded summary
after rounding to the shown precision.
