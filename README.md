# betavae-xai-ad

Repositorio del pipeline usado en el paper **"Explainable Latent Representation Learning for Alzheimer’s Disease: A β-VAE and Saliency Map Framework"**. Construye conectomas multi-canal de rs-fMRI (AAL3), aprende representaciones latentes con un β-VAE convolucional y entrena clasificadores clásicos para distinguir AD vs CN, con análisis de interpretabilidad y control de fuga de sitio.

## Visión general
- Extracción de conectividad: series temporales ROI → filtros band-pass → conectividad estática/dinámica (Pearson OMST, Pearson full, Mutual Information, Distance-Corr, dFC mean/STD, opcional Granger y precision). Reordenamiento AAL3 → Yeo-17 y armado de tensores `[N, C, 131, 131]`.
- Modelado: β-VAE CNN con annealing cíclico de β, dropout/GroupNorm, activación final configurable. Latentes (`mu` o `z`) alimentan clasificadores (RF, SVM-RBF, LogReg, MLP, LightGBM, XGBoost, CatBoost) con OptunaSearchCV y SMOTE opcional.
- QC / XAI: histogramas raw vs normalizado vs reconstruido, métricas de fuga de sitio (scanner leakage) en espacio crudo y latente, silhouette de separación AD/CN.

## Estructura del repositorio
- `src/betavae_xai/`
  - `feature_extraction.py`: pipeline de conectividad y armado del tensor global (señales AAL3, filtrado, ventana dFC, canales múltiples, guardado `GLOBAL_TENSOR_from_<run>.npz` con nombres de ROIs y redes).
  - `models/`: `convolutional_vae.py` (β-VAE CNN) y `classifiers.py` (pipelines + espacios de búsqueda Optuna para RF/SVM/LogReg/MLP/GB/XGB/Cat).
  - `analysis_qc.py`: stats de distribución y fuga de sitio con LogisticRegression CV.
- `scripts/run_vae_clf_ad.py`: driver principal de experimentos (CV anidada, entrenamiento VAE, tuning de clasificadores, QC, guardado de artefactos).
- `notebooks/`: QC y exploración de conectomas/latentes (`feature_extraction.ipynb`, `qc_fmri_bold.ipynb`).
- `data/`: insumos locales (atlas AAL3, mapeo a Yeo17, metadatos) y salidas pesadas (conectomas, tensores) **no versionadas**.
- `results/`: carpeta de trabajo ignorada por Git para logs, métricas, figuras y modelos entrenados.
- `environment.yml`: especificación Conda (Python, PyTorch, Optuna, scikit-learn, LightGBM/XGBoost/CatBoost, Nilearn, etc.).

## Datos esperados
- Tensor global (`GLOBAL_TENSOR_from_*.npz`) con llaves: `global_tensor_data` (`N,C,131,131`), `subject_ids`, `roi_names_in_order`, `network_labels_in_order` (opcional), `channel_names`, hiperparámetros de preprocesado.
- Metadatos CSV con columna `SubjectID`; el script espera `ResearchGroup_Mapped` (CN/AD) y opcionalmente sitio/escáner (`Manufacturer*`, `Vendor*` o `Site*`), sexo (`Sex`), `Age_Group` u otras columnas para estratificar o añadir como features.
- Archivos de apoyo en `data/`: `aal3_131_to_yeo17_mapping.csv`, `AAL3v1.nii.gz`, `ROI_MNI_V7_vol.txt`, `SubjectsData*.csv`, más las carpetas con señales ROI preprocesadas y conectomas per-sujeto (no incluidas en Git).

## Flujo de trabajo
1) **Extraer conectividad** (`src/betavae_xai/feature_extraction.py`): procesa las series ROI (band-pass, HRF opcional, ventanas deslizantes), calcula los canales de conectividad definidos en los flags `USE_*_CHANNEL`, filtra/excluye ROIs pequeños y reordena por Yeo-17. El resultado es una carpeta `AAL3_dynamicROIs_fmri_tensor_*` con conectomas por sujeto y un `GLOBAL_TENSOR_from_<run>.npz`. La normalización inter-canal para modelado debe hacerse dentro de cada fold (ver script principal) para evitar data leakage.
2) **Entrenar VAE + clasificador** (`scripts/run_vae_clf_ad.py`): carga el tensor global y metadatos, separa folds estratificados, normaliza por canal usando solo el train del fold, entrena β-VAE con annealing y scheduler opcional, obtiene latentes y entrena clasificadores con OptunaSearchCV; ejecuta QC (silhouette, fugas de sitio, histogramas si se activa). GPU se usa automáticamente si está disponible (PyTorch, CuPy, LightGBM/XGBoost).

## Uso rápido del script principal
```bash
python scripts/run_vae_clf_ad.py \
  --global_tensor_path data/AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned/GLOBAL_TENSOR_from_AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned.npz \
  --metadata_path data/SubjectsData_AAL3_procesado2.csv \
  --output_dir results/vae_clf_run1 \
  --classifier_types rf svm \
  --qc_check_scanner_leakage \
  --qc_analyze_distributions
```
Argumentos clave:
- Datos: `--global_tensor_path`, `--metadata_path`, `--channels_to_use` (índices según `DEFAULT_CHANNEL_NAMES` en el script), `--output_dir`.
- VAE: `--latent_dim`, `--beta_vae`, `--num_conv_layers_encoder`, `--vae_final_activation` (`tanh/sigmoid/linear`), `--lr_scheduler_type` (`plateau/cosine_warm`), `--cyclical_beta_*`.
- Clasificador: `--classifier_types`, `--use_smote`, `--classifier_use_class_weight`, `--latent_features_type` (`mu/z`), `--metadata_features` para concatenar columnas de metadatos.
- CV: `--outer_folds`, `--repeated_outer_folds_n_repeats`, `--inner_folds`, `--classifier_stratify_cols`.
- QC: `--qc_check_scanner_leakage`, `--qc_analyze_distributions`, `--save_vae_training_history`, `--save_fold_artefacts`.

## Modelos y componentes
- **Convolutional β-VAE** (`src/betavae_xai/models/convolutional_vae.py`): encoder conv con GroupNorm y dropout, FC intermedia opcional, decoder conv-transpose; outputs `recon, mu, logvar, z`; annealing de β y AMP en el entrenamiento.
- **Clasificadores clásicos** (`src/betavae_xai/models/classifiers.py`): pipelines imblearn con escalado, SMOTE opcional, espacios de búsqueda Optuna para RF/GB/SVM/LogReg/MLP/XGB/Cat, soporte GPU cuando lo permite la librería.
- **QC** (`src/betavae_xai/analysis_qc.py`): stats por canal/off-diagonal raw vs normalizado vs reconstruido, histogramas, evaluación de fuga de sitio en conectoma normalizado vs latente.

## Salidas principales
- Por fold (en `--output_dir/fold_*`): `test_predictions_<clf>.csv`, `vae_norm_params.joblib`, métricas QC (`latent_qc_metrics.csv`), artefactos opcionales del clasificador (`classifier_*_pipeline_*.joblib`).
- Resumen global: `all_folds_metrics_MULTI_*.csv` y `summary_metrics_MULTI_*.txt` con argumentos, hash de Git y métricas promedio/desviación por clasificador.

## Entorno
1) `conda env create -f environment.yml`
2) `conda activate betavae-xai-ad` (o el nombre definido en el YAML)
3) Verificar GPU opcional: `python - <<'PY'\nimport torch, cupy; print(torch.cuda.is_available(), cupy.is_available())\nPY`
