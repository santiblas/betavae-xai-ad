# Explainable Latent Representation Learning for Alzheimer’s Disease

> **A β-VAE and Saliency Map Framework**

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Este repositorio contiene el pipeline oficial del paper **"Explainable Latent Representation Learning for Alzheimer’s Disease"**.

El framework construye conectomas multi-canal a partir de rs-fMRI (atlas AAL3), aprende representaciones latentes desenredadas utilizando un **β-VAE convolucional**, y entrena clasificadores supervisados para distinguir Alzheimer (AD) de Controles (CN). Incluye un módulo completo de **Explainable AI (XAI)** para generar mapas de saliency y controlar fugas de información por sitio de adquisición.

## 🧠 Características Principales

* **Conectividad Multi-canal:** Procesamiento de series temporales ROI con múltiples métricas de conectividad estática y dinámica:
    * Pearson (Full & OMST)
    * Mutual Information (kNN)
    * Distance Correlation
    * Métricas dinámicas (Mean/STD de ventanas deslizantes)
* **Deep Learning Generativo:** Arquitectura **Convolutional β-VAE** con *cyclical annealing* para aprender variedades latentes robustas.
* **Clasificación Robusta:** Tuning automático de clasificadores (SVM, RF, XGBoost, LogReg, MLP) usando **Optuna** y validación cruzada anidada.
* **Interpretabilidad (XAI):** Mapeo de importancia desde el espacio latente hacia las conexiones cerebrales (ROI-to-ROI).
* **Control de Calidad (QC):** Detección automática de *scanner leakage* y análisis de reconstrucción de conectomas.

## 📂 Estructura del Repositorio

```text
├── src/betavae_xai/          # Código fuente del paquete
│   ├── feature_extraction.py # Pipeline de extracción de conectomas y tensores
│   ├── analysis_qc.py        # Módulos de QC y detección de bias
│   └── models/               # Arquitecturas (β-VAE CNN) y Clasificadores
├── scripts/                  # Scripts ejecutables
│   └── run_vae_clf_ad.py     # Driver principal (Entrenamiento, CV, QC)
├── notebooks/                # Exploración y Generación de Figuras
│   └── Figures_Nature/       # Figuras finales del paper
├── data/                     # Insumos (Atlas, Metadatos) - Datos crudos ignorados
└── results/                  # Salidas de modelos y logs (Ignorado por git)