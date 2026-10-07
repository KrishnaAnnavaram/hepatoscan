# Data

This repository commits no CT data, no masks, no model weights and no user data.

## Real data: LiTS

| Item | Value |
|---|---|
| Source name | LiTS, the Liver Tumor Segmentation Benchmark (131 training CT volumes with masks) |
| URL | Challenge: https://competitions.codalab.org/competitions/17094 . Mirrors on Kaggle: "Liver Tumor Segmentation" parts 1 and 2 |
| License / terms | CC BY-NC-SA 4.0 (non-commercial). Read the terms before you use or share anything made from it. Cite the LiTS paper (Bilic et al., Medical Image Analysis, 2023) |
| Size | Tens of GB |

LiTS gives `volume-<n>.nii` (CT, Hounsfield units) and `segmentation-<n>.nii` (labels 0 = background,
1 = liver, 2 = tumor). hepatoscan reads `.npz` volumes listed in a `manifest.csv`. Convert each case
once:

```python
import nibabel as nib, numpy as np
img, seg = nib.load("volume-0.nii"), nib.load("segmentation-0.nii")
image = np.transpose(img.get_fdata(dtype=np.float32), (2, 1, 0))     # (slice, row, column)
mask = np.transpose(np.asarray(seg.dataobj), (2, 1, 0)).astype(np.uint8)
zooms = img.header.get_zooms()[:3]
np.savez_compressed("data/lits/case_000.npz", image=image, spacing=np.array(zooms[::-1]), mask=mask)
```

Then write `data/lits/manifest.csv` with the columns `case_id,file`.

## Expected files

| File | Contents |
|---|---|
| `manifest.csv` | Columns `case_id`, `file` (required), other columns optional |
| `<case_id>.npz` | `image` (D, H, W) float32 HU, `spacing` (3,) mm along (D, H, W), `mask` (D, H, W) uint8 labels 0/1/2 |

## Synthetic data

`hepatoscan synth --cases 30 --out data/synthetic` writes phantoms in the same format. A phantom has
air, fat, soft tissue, a spine, a liver and zero to three hypodense lesions, with noise. It is not a
real CT scan. The demo and the tests use phantoms only.
