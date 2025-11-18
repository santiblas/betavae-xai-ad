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