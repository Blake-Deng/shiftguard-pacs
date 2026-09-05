# VLCS And OfficeHome Data Handoff

The server currently cannot reach Hugging Face or GitHub. Download the datasets locally from a trustworthy source and validate the extracted folders before uploading. Prefer the official dataset distribution or the exact version used by DomainBed; avoid repackaged train/test splits that remove images.

## Required versions

### VLCS

- Standard shared five-class VLCS used by DomainBed.
- 10,729 images total.
- Classes: `bird`, `car`, `chair`, `dog`, `person`.
- Original domain folder aliases commonly appear as `CALTECH`, `LABELME`, `PASCAL`, and `SUN`.

Accepted canonical layout:

```text
VLCS/
  Caltech101/{bird,car,chair,dog,person}/...
  LabelMe/{bird,car,chair,dog,person}/...
  SUN09/{bird,car,chair,dog,person}/...
  VOC2007/{bird,car,chair,dog,person}/...
```

The validator also recognizes the original `CALTECH/LABELME/PASCAL/SUN` names.

### OfficeHome

- Standard Office-Home release used by DomainBed.
- 15,588 images total and 65 shared classes.
- Domains: `Art`, `Clipart`, `Product`, `Real World`.

Accepted canonical layout:

```text
OfficeHome/
  Art/<65 class folders>/...
  Clipart/<65 class folders>/...
  Product/<65 class folders>/...
  Real_World/<65 class folders>/...
```

The validator also recognizes `Real World` and `RealWorld`.

## Validate locally

Place `validate_revision_dataset.py` next to the extracted dataset, then run:

```bash
python validate_revision_dataset.py \
  --dataset vlcs \
  --root /path/to/extracted/VLCS \
  --inventory vlcs_inventory.json \
  --verify-images

python validate_revision_dataset.py \
  --dataset officehome \
  --root /path/to/extracted/OfficeHome \
  --inventory officehome_inventory.json \
  --verify-images
```

Both commands must finish without an exception. Keep the two inventory JSON files for comparison after upload.

## Upload destinations

Upload the extracted folders, not only the inventory files:

```text
data/VLCS/
data/OfficeHome/
```

After placement, run the same validator again from the repository root. Formal manifests will only be created if the image count, shared classes, readable-image check, and deterministic fingerprint pass.
