# betavae-xai-ad

Code and experiments for the paper:

> **"Explainable Latent Representation Learning for Alzheimer’s Disease: A β-VAE and Saliency Map Framework"**

This repository contains the connectivity-extraction pipeline, QC notebooks, and experiment drivers used to build β-VAE latent spaces from multi-channel resting-state fMRI connectomes and to generate explainable Alzheimer’s disease classifiers.



---

## Repository layout

- `src/betavae_xai/`  
  Core Python package (connectivity feature extraction, VAE models, downstream classifiers, cross-validation, XAI utilities).

- `scripts/`  
  Command-line entry points to reproduce the main paper experiments (β-VAE training, ablation studies, site decodability, saliency / XAI analyses).  
  Run each script with `--help` for details.

- `configs/`  
  YAML configuration files describing each experiment (data paths, model hyper-parameters, CV setup, etc.).

- `data/`  
  Working directory for all neuroimaging inputs and derived connectivity tensors.  
  Heavy data (fMRI NIfTI, full tensors, model weights) are **ignored by Git**; only small metadata and mapping files are versioned.  
  See `data/README.md` for the expected structure.

- `notebooks/`  
  Jupyter notebooks for QC and exploratory analysis:
  - `feature_extraction.ipynb`: sanity checks on the connectivity tensors, inspection of `GLOBAL_TENSOR_from_*.npz`, and visualization of the average connectome per channel (7 connectivity channels × 131 AAL3 ROIs).
  - `qc_fmri_bold.ipynb`: basic fMRI BOLD quality control (motion, temporal SNR, time-series inspection) used to define the QC-ed subject list that feeds the connectivity pipeline.  
  Derived HTML exports (`*.html`) and large QC outputs live here but are not meant to be tracked in Git.

- `results/`  
  Local outputs (logs, metrics, figures, trained models).  
  This directory is **ignored by Git**; use it as a scratch space to reproduce the paper’s numbers and figures.

- `environment.yml`  
  Conda environment specification for reproducing the software stack (Python, PyTorch, scientific Python ecosystem).

---

## Key data and mapping files

The repository ships only lightweight, non-sensitive files; large neuroimaging data and tensors must be provided locally.

Inside `data/` you will typically have:

- `aal3_131_to_yeo17_mapping.csv`  
  Atlas-mapping CSV that links each of the 131 AAL3 ROIs (after exclusions) to:
  - its AAL3 label,
  - MNI coordinates / volumetric information,
  - and the overlapping **Yeo-17 network** label.  

  This file is used by the connectivity pipeline and notebooks to:
  - define the final ROI order,
  - reorder connectivity matrices by functional network,
  - and generate network-level summaries and figures.

- `AAL3v1.nii.gz` *(local, not tracked)* 3D AAL3 parcellation in MNI space.

- `ROI_MNI_V7_vol.txt`  
  Text file with ROI volumetric / coordinate information (used for sanity checks and visualizations).

- `ROISignals_AAL3_NiftiPreprocessedAllBatchesNorm/` *(local, not tracked)* Directory with subject-level preprocessed BOLD time series, one NIfTI/NPY per subject and ROI.

- `SubjectsData.csv`, `SubjectsData_AAL3_procesado2.csv` *(local, usually not tracked)* Subject-level metadata and inclusion/QC flags. Depending on your data-sharing policy, you may keep these local and untracked.

- `AAL3_dynamicROIs_fmri_tensor_*` *(local, not tracked)* Output folders created by the connectivity pipeline, e.g.:

  ```text
  AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned/