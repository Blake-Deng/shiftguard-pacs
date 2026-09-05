# CGC v2 Protocol Lock

Locked: 2026-09-04, before any 30-epoch CGC formal run reached target
evaluation.

## Scientific configuration

- Gate threshold: exactly 0; it will not be changed after target results.
- Feature weight: 0.10.
- KL weight: 0.05.
- Temperature: 2.
- Consistency ramp: 5 epochs.
- RandAugment: N=2, M=9.
- Optimizer: AdamW, learning rate 3e-4, weight decay 1e-4.
- Schedule: cosine, 30 epochs, batch size 64, input size 224.
- Formal seeds: 42, 123, 3407, 2026, 2027.
- The training seed is the replicate; target domains are not independent
  replicates.

## Target isolation

- The held-out target is absent from optimization, early stopping, and
  checkpoint selection.
- Formal target data are constructed only after restoring the checkpoint
  selected by source validation.
- Every formal run must record target_evaluations=1.
- Strict nested screening must record target_evaluations=0, target_accuracy
  null, and n_test=0.

## Locked task matrix

- 80 formal CGC runs: PACS, VLCS, and OfficeHome with ResNet-50, plus PACS
  with exact ViT-S/16; four targets by five seeds in each setting.
- 20 PACS ResNet-50 EMA Mean Teacher runs with alpha=0.999.
- 16 PACS seed-42 component-gate ablations.
- 8 matched efficiency reruns for Strong Augmentation and Feature+KL.
- 108 strict nested PACS screening runs: four outer targets, three inner
  validation domains, three candidates, and three screening seeds.
- Existing five-seed Strong Augmentation and Feature+KL results are reused as
  formal controls and as the outer evaluations after nested selection.

Total new queued jobs: 232.

## Preflight evidence

- PACS Art Painting and Sketch two-epoch CGC smoke tests completed without
  NaN, restored a source-selected checkpoint, and evaluated the target once.
- Feature keep rates were approximately 0.81--0.87 and KL keep rates were
  approximately 0.65--0.69; neither mask collapsed.
- Four concurrent ResNet-50 CGC jobs and four concurrent ViT-S/16 CGC jobs
  completed on one RTX 5090 without OOM.
- A nested preflight trained on Photo and Sketch, validated on Cartoon, and
  recorded zero Art Painting target evaluations.

## Code hashes at formal launch

- cgc_experiment.py:
  4860910abd2f2b2a5bfcbdf2864dad567b2075264a28eb697a3a2024416d0751
- run_cgc_complete_queue.py:
  c4535ac681afca8e63b0b069518a75726418887e247207037fce2d45be3541bb
- summarize_cgc_results.py:
  31ea8e832e70e61810300919815569603706e13cf3ea84638c03b2841c7a939c

## Operational note

The first tmux launch exposed empty epochs and batch-size fields in the
generated manifest and exited in argument parsing before training. That
session was terminated, the generator was fixed, every non-complete row was
restored to pending, and the full command was verified with epochs=30 and
batch_size=64 before relaunch. Those parser failures are not experimental
runs and produce no result JSON.
