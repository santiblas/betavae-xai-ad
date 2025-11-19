# betavae-xai-ad

Code and experiments for the paper:

> **"Explainable Latent Representation Learning for Alzheimer’s Disease:  
> A β-VAE and Saliency Map Framework"**

## Structure

- `src/betavae_xai/`: core Python package (VAE, classifiers, CV, XAI).
- `scripts/`: entry-point scripts to reproduce experiments (paper main, ablations, site decodability, XAI).
- `configs/`: YAML configs describing each experiment.
- `data/`: **not versioned**; see `data/README.md` for expected structure.
- `results/`: local outputs (logs, metrics, figures); ignored by Git.
- `notebooks/`: QC and figure-generation notebooks.

## Environment

```bash
conda env create -f environment.yml
conda activate serentipia_gpu   # o el nombre que elijas
```

## Experiment drivers

### Scripts

- `src/betavae_xai/feature_extraction.py`: pipeline `v6.5.19` that loads the QC’ed subject list (`SUBJECT_METADATA_CSV_PATH`), reads ROI signals from `data/ROISignals_AAL3_NiftiPreprocessedAllBatchesNorm/`, applies the Yeo-17 reordering, and computes each connectivity channel (Pearson OMST/fallback, MI, dFC, DistanceCorr, optional Granger). Running `python -m src.betavae_xai.feature_extraction` generates per-subject tensors under `data/AAL3_dynamicROIs_fmri_tensor_*/individual_subject_tensors/` and the aggregated `GLOBAL_TENSOR.npz`. Those `.npz` files seed the β-VAE training/evaluation experiments described in `scripts/` (paper main run, ablation sweeps, site decodability) and power the saliency/XAI pipeline.

### Notebooks

- `notebooks/conn_matrix.ipynb`: interactive QC notebook to sanity-check the connectivity extraction before launching full experiments. It visualises ROI metadata from `ROI_MNI_V7_vol.txt`, inspects the atlas alignment (`AAL3v1.nii.gz` vs. Yeo-17), and can call helpers from `feature_extraction.py` to inspect single-subject matrices and confirm ROI ordering. Use it to document figure-ready connectivity examples and to ensure the tensors that feed the β-VAE training match the configuration used in the scripts above.
