#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature extraction module for BetaVAE XAI pipeline.
This module includes functions for processing fMRI ROI time series data,
computing various connectivity measures, and preparing data for VAE input.
The location of this file is: src/betavae_xai/feature_extraction.py
"""
import itertools
from dcor import distance_correlation 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression
# Para Graphical Lasso (Sugerencia Tesis 5)
# from sklearn.covariance import GraphicalLassoCV 
from nilearn.glm.first_level import spm_hrf, glover_hrf 
from nilearn.datasets import fetch_atlas_yeo_2011 # Para reordenamiento de ROIs
from nilearn import image as nli_image # Para resampling
import nibabel as nib # Para cargar atlas NIfTI
from scipy.signal import butter, filtfilt, deconvolve, windows
from scipy.interpolate import interp1d
from tqdm import tqdm
import os
import scipy
import scipy.io as sio
from pathlib import Path
import psutil
import gc
import logging
import time
from typing import List, Tuple, Dict, Optional, Any, Union 
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.tsa.stattools import grangercausalitytests 
import networkx as nx 
import warnings
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import argparse # SUGEERENCIA (Mantenibilidad): Para configurar parámetros desde CLI

# --- Configuración del Logger ---
# SUGEERENCIA (Mantenibilidad): Considerar hacer el nivel de logging configurable (ej. vía argparse o var de entorno)
# para alternar entre INFO para producción y DEBUG para depuración.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Importación de OMST usando dyconnmap ---
OMST_PYTHON_LOADED = False
orthogonal_minimum_spanning_tree = None
PEARSON_OMST_CHANNEL_NAME_PRIMARY = "Pearson_OMST_GCE_Signed_Weighted" 
PEARSON_OMST_FALLBACK_NAME = "Pearson_Full_FisherZ_Signed" 
PEARSON_OMST_CHANNEL_NAME = PEARSON_OMST_FALLBACK_NAME # Default
try:
    from dyconnmap.graphs.threshold import threshold_omst_global_cost_efficiency
    orthogonal_minimum_spanning_tree = threshold_omst_global_cost_efficiency 
    logger.info("Successfully imported 'threshold_omst_global_cost_efficiency' from 'dyconnmap.graphs.threshold' and aliased as 'orthogonal_minimum_spanning_tree'.")
    OMST_PYTHON_LOADED = True
    PEARSON_OMST_CHANNEL_NAME = PEARSON_OMST_CHANNEL_NAME_PRIMARY 
except ImportError:
    logger.error("ERROR: Dyconnmap module or 'threshold_omst_global_cost_efficiency' not found. "
                 f"Channel '{PEARSON_OMST_FALLBACK_NAME}' will be used as fallback. "
                 "Please ensure dyconnmap is installed: pip install dyconnmap")
except Exception as e_import: 
    logger.error(f"ERROR during dyconnmap import: {e_import}. "
                 f"Channel '{PEARSON_OMST_FALLBACK_NAME}' will be used as fallback.")

# --- 0. Global Configuration and Constants ---
# SUGEERENCIA (Mantenibilidad): Mover estas constantes a un archivo de configuración (e.g., config.yaml o config.py)
# y cargarlas usando una librería como OmegaConf o simplemente importándolas.
# Esto facilita la modificación de parámetros sin tocar el código del pipeline.

# --- Configurable Parameters ---
# Detectar la raíz del proyecto asumiendo estructura:
#   <repo_root>/src/betavae_xai/feature_extraction.py
# y usar <repo_root>/data como base para todos los archivos AAL3.
THIS_FILE = Path(__file__).resolve()
# parents[0] = .../src/betavae_xai
# parents[1] = .../src
# parents[2] = .../betavae-xai-ad  (raíz del repo)
PROJECT_ROOT = THIS_FILE.parents[2]

BASE_PATH_AAL3 = PROJECT_ROOT / 'data'
QC_OUTPUT_DIR = BASE_PATH_AAL3 / 'qc_outputs_doctoral_v3.2_aal3_shrinkage_flexible_thresh_fix'
SUBJECT_METADATA_CSV_PATH = BASE_PATH_AAL3 / 'SubjectsData_AAL3_procesado2.csv'

QC_REPORT_CSV_PATH = PROJECT_ROOT / 'notebooks' / 'qc_outputs_doctoral' / 'report_qc_final_with_discard_flags_v3.2.csv'
#QC_REPORT_CSV_PATH = QC_OUTPUT_DIR / 'report_qc_final_with_discard_flags_v3.2.csv'
ROI_SIGNALS_DIR_PATH_AAL3 = BASE_PATH_AAL3 / 'ROISignals_AAL3_NiftiPreprocessedAllBatchesNorm'
ROI_FILENAME_TEMPLATE = 'ROISignals_{subject_id}.mat'
AAL3_META_PATH = BASE_PATH_AAL3 / 'ROI_MNI_V7_vol.txt' 
# !!! IMPORTANTE: Especificar la ruta a tu archivo NIfTI del atlas AAL3 !!!
# Debe estar en el mismo espacio que el atlas Yeo (ej. MNI152 2mm).
# Si AAL3_NIFTI_PATH es 1mm y Yeo es 2mm, se intentará remuestrear AAL3.
AAL3_NIFTI_PATH = BASE_PATH_AAL3 / "AAL3v1.nii.gz" #  <--- ACTUALIZA ESTA RUTA A TU ATLAS AAL3 EN ESPACIO MNI 2mm

TR_SECONDS = 3.0 
LOW_CUT_HZ = 0.01
HIGH_CUT_HZ = 0.08
FILTER_ORDER = 2 
TAPER_ALPHA = 0.1 

RAW_DATA_EXPECTED_COLUMNS = 170 
AAL3_MISSING_INDICES_1BASED = [35, 36, 81, 82] 
EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL = RAW_DATA_EXPECTED_COLUMNS - len(AAL3_MISSING_INDICES_1BASED)
SMALL_ROI_VOXEL_THRESHOLD = 100 
# NOTA NEUROCIENCIA (Sugerencia Tesis 1.3): Considerar revisar si alguno de los ROIs excluidos por ser < SMALL_ROI_VOXEL_THRESHOLD 
# (35 ROIs en la configuración actual) son consistentemente implicados en la patología del Alzheimer 
# según la literatura que utiliza el atlas AAL. Si una región pequeña pero crucial (ej. hipocampo, amígdala) 
# se excluye debido a atrofia, podría ser problemático. Evaluar ajuste de umbral o estrategias alternativas
# (ej. fusión con ROIs adyacentes si la señal es muy pobre, o usar parcelación híbrida anatómico-funcional).
# SUGEERENCIA NEUROCIENCIA (Sugerencia Tesis 1 - Parcelación): Considerar enfoques multiescala (ej. Schaefer 200/400/1000 + Yeo17)
# o parcelaciones híbridas (AAL3 límbico + Schaefer neocórtex) para capturar efectos a diferentes granularidades.

N_ROIS_EXPECTED = 131 # Se actualizará en _initialize_aal3_roi_processing_info
TARGET_LEN_TS = 140 

N_NEIGHBORS_MI = 5 
# NOTA NEUROCIENCIA (MI_KNN): La elección de k (N_NEIGHBORS_MI) en KNN para estimar MI puede ser sensible. 
# El valor de 5 es un punto de partida común. Explorar la robustez a diferentes valores de k podría ser 
# un análisis de sensibilidad útil si los resultados de MI son particularmente influyentes.

DFC_WIN_POINTS = 30 
DFC_STEP = 5      
# NOTA NEUROCIENCIA (dFC - Sugerencia Tesis 2.1 & 3):
# Con TR = 3.0s:
# - Longitud de ventana (DFC_WIN_POINTS): 30 puntos * 3s/punto = 90 segundos.
# - Paso de ventana (DFC_STEP): 5 puntos * 3s/punto = 15 segundos.
# - Número de ventanas para TARGET_LEN_TS = 140: 23 ventanas.
# SUGEERENCIA NEUROCIENCIA (Tesis):
#   1. Experimentar con diferentes longitudes de ventana (ej. 20-40 TRs, i.e., 60-120s)
#      o métodos de enventanado (ej. con tapering tipo Hamming) podría ser beneficioso.
#   2. Considerar ventanas adaptativas guiadas por varianza instantánea o "change-point detection"
#      (ej. usando librerías como `ruptures`, ver Zhang et al., 2024) para reducir el "blur" temporal y capturar eventos transitorios.
#      Esto es más avanzado y requeriría modificar significativamente el cálculo de dFC.

APPLY_HRF_DECONVOLUTION = False 
HRF_MODEL = 'glover' 
# NOTA NEUROCIENCIA (Variabilidad HRF - Sugerencia Tesis 2):
# El uso de un kernel HRF canónico fijo (Glover/SPM) ignora la variabilidad inter-sujeto, inter-región,
# y los cambios potenciales debidos a edad o patología (atrofia).
# Si se decidiera activar la deconvolución:
#   1. Considerar la estimación de HRF por sujeto/región (ej. usando modelos FIR o bases derivadas).
#   2. Alternativamente, incluir derivadas temporales y de dispersión del HRF canónico como regresores
#      en el GLM previo a la extracción de series temporales para capturar variaciones de latencia/ancho.
#   3. Si se usa deconvolución directa, aplicar regularización (ej. Wiener filter) para evitar amplificación de ruido.
# Por ahora, mantener APPLY_HRF_DECONVOLUTION = False es una opción prudente dada la complejidad.

# Parámetros para Causalidad de Granger
USE_GRANGER_CHANNEL = True
GRANGER_MAX_LAG = 1 
# NOTA NEUROCIENCIA (Causalidad de Granger con TR largo):
# Con TR_SECONDS = 3.0s y GRANGER_MAX_LAG = 1, el modelo de Granger intentará predecir
# la señal de un ROI en el tiempo `t` usando la información de otro ROI en el tiempo `t-3s`.
# - Limitaciones e Interpretación: Ver comentarios extensos en versiones previas y en la literatura.
#   La interpretación debe ser como "influencia predictiva" y no causalidad neuronal directa.
# - Simetrización: Pierde direccionalidad.
# SUGEERENCIA NEUROCIENCIA (Tesis): Asegurar estacionariedad de las series para Granger. Evaluar su contribución real al modelo.
# Considerar alternativas como DCM o Granger multivariado para subconjuntos de ROIs si la direccionalidad es clave.

deconv_str = "_deconv" if APPLY_HRF_DECONVOLUTION else ""
granger_suffix_global = f"GrangerLag{GRANGER_MAX_LAG}" if USE_GRANGER_CHANNEL else "NoEffConn"
OUTPUT_CONNECTIVITY_DIR_NAME_BASE = f"AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17" # Versión actualizada

POSSIBLE_ROI_KEYS = ["signals", "ROISignals", "roi_signals", "ROIsignals_AAL3", "AAL3_signals", "roi_ts"] 

USE_PEARSON_OMST_CHANNEL = True 
USE_PEARSON_FULL_SIGNED_CHANNEL = True 
USE_MI_CHANNEL_FOR_THESIS = True 
USE_DFC_ABS_DIFF_MEAN_CHANNEL = True 
USE_DFC_STDDEV_CHANNEL = True 
# --- NUEVOS CANALES (paper Liu et al., Nat. Methods 2025) ---
USE_PRECISION_CHANNEL = False          # Inverse-covariance (hub friendly)
USE_DCOR_CHANNEL      = True          # Distance-Correlation (no linealidad)


CONNECTIVITY_CHANNEL_NAMES: List[str] = [] 
N_CHANNELS = 0 

try:
    TOTAL_CPU_CORES = multiprocessing.cpu_count()
    MAX_WORKERS = max(1, TOTAL_CPU_CORES // 2 if TOTAL_CPU_CORES > 2 else 1)
except NotImplementedError:
    logger.warning("multiprocessing.cpu_count() no está implementado en esta plataforma. Usando MAX_WORKERS = 1.")
    TOTAL_CPU_CORES = 1
    MAX_WORKERS = 1
logger.info(f"Global MAX_WORKERS for ProcessPoolExecutor set to: {MAX_WORKERS} (based on {TOTAL_CPU_CORES} total cores)")

VALID_AAL3_ROI_INFO_DF_166: Optional[pd.DataFrame] = None
AAL3_MISSING_INDICES_0BASED: Optional[List[int]] = None
INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166: Optional[List[int]] = None
FINAL_N_ROIS_EXPECTED: Optional[int] = None 
OUTPUT_CONNECTIVITY_DIR_NAME: Optional[str] = None 
AAL3_ROI_ORDER_MAPPING: Optional[Dict[str, Any]] = None 

YEO17_LABELS_TO_NAMES = {
    0: "Background/NonCortical", 
    1: "Visual_Peripheral", 2: "Visual_Central",
    3: "Somatomotor_A", 4: "Somatomotor_B",
    5: "DorsalAttention_A", 6: "DorsalAttention_B",
    7: "Salience_VentralAttention_A", 8: "Salience_VentralAttention_B",
    9: "Limbic_A_TempPole", 10: "Limbic_B_OFC",
    11: "Control_C", 12: "Control_A", 13: "Control_B",
    14: "DefaultMode_Temp", 15: "DefaultMode_Core",
    16: "DefaultMode_DorsalMedial", 17: "DefaultMode_VentralMedial"
}

# --- Funciones para Reordenamiento de ROIs ---
# Esta función AHORA se define ANTES de _initialize_aal3_roi_processing_info
def _get_aal3_network_mapping_and_order() -> Optional[Dict[str, Any]]:
    """
    Carga/define el mapeo de ROIs AAL3 a redes funcionales Yeo-17 y el nuevo orden.
    Requiere:
        - AAL3_NIFTI_PATH: Ruta al archivo NIfTI del atlas AAL3.
        - AAL3_META_PATH: Ruta al archivo .txt con metadatos de AAL3 (para nombres y colores).
        - Variables globales: VALID_AAL3_ROI_INFO_DF_166, INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166, FINAL_N_ROIS_EXPECTED.
    SUGERENCIA (Tesis 1): Considerar también ordenar por gradientes de conectividad (e.g., Margulies et al.)
                         o usar parcelaciones multiescala (e.g., Schaefer + Yeo17).
    """
    logger.info("Attempting to map AAL3 ROIs to Yeo-17 networks and reorder.")

    if not AAL3_NIFTI_PATH.exists():
        logger.error(f"AAL3 NIfTI file NOT found at: {AAL3_NIFTI_PATH}. Cannot perform ROI reordering.")
        return None
    if VALID_AAL3_ROI_INFO_DF_166 is None or INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166 is None or FINAL_N_ROIS_EXPECTED is None:
        logger.error("Global AAL3 processing variables (VALID_AAL3_ROI_INFO_DF_166, etc.) not initialized. Cannot perform ROI reordering.")
        return None

    try:
        # 1. Cargar atlas Yeo-17
        logger.info("Fetching Yeo 17-network atlas...")
        yeo_atlas_obj = fetch_atlas_yeo_2011() # Corregido: sin argumento 'version'
        yeo_img = nib.load(yeo_atlas_obj.thick_17) # Acceder al atributo .thick_17
        yeo_data = yeo_img.get_fdata().astype(int)
        logger.info(f"Yeo-17 atlas loaded. Shape: {yeo_data.shape}, Affine: \n{yeo_img.affine}")

        # 2. Cargar atlas AAL3 NIfTI
        logger.info(f"Loading AAL3 NIfTI from: {AAL3_NIFTI_PATH}")
        aal_img_orig = nib.load(AAL3_NIFTI_PATH)
        logger.info(f"Original AAL3 NIfTI atlas loaded. Shape: {aal_img_orig.shape}, Affine: \n{aal_img_orig.affine}")

        # Resample AAL3 to Yeo space if affines don't match or shapes differ significantly
        if not np.allclose(aal_img_orig.affine, yeo_img.affine, atol=1e-3) or aal_img_orig.shape != yeo_img.shape:
            logger.warning("Affines or shapes of AAL3 and Yeo atlases do not match. "
                           "Attempting to resample AAL3 to Yeo space using nearest neighbor interpolation.")
            try:
                aal_img_resampled = nli_image.resample_to_img(aal_img_orig, yeo_img, interpolation='nearest')
                aal_data = aal_img_resampled.get_fdata().astype(int)
                logger.info(f"AAL3 atlas resampled. New Shape: {aal_data.shape}, New Affine: \n{aal_img_resampled.affine}")
            except Exception as e_resample:
                logger.error(f"Failed to resample AAL3 atlas: {e_resample}. ROI reordering will be skipped.")
                return None
        else:
            aal_data = aal_img_orig.get_fdata().astype(int)
            logger.info("AAL3 and Yeo atlases appear to be in the same space. No resampling performed.")


        # 3. Identificar las ROIs finales de AAL3 y sus etiquetas originales (colores)
        final_aal3_rois_info_df = VALID_AAL3_ROI_INFO_DF_166.drop(INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166).reset_index(drop=True)
        
        if len(final_aal3_rois_info_df) != FINAL_N_ROIS_EXPECTED:
            logger.error(f"Mismatch in expected final ROI count. Expected {FINAL_N_ROIS_EXPECTED}, "
                         f"derived {len(final_aal3_rois_info_df)} from VALID_AAL3_ROI_INFO_DF_166. Cannot proceed with reordering.")
            return None

        original_131_aal3_colors = final_aal3_rois_info_df['color'].tolist()
        original_131_aal3_names = final_aal3_rois_info_df['nom_c'].tolist()
        
        logger.info(f"Mapping {len(original_131_aal3_colors)} AAL3 ROIs to Yeo-17 networks...")
        roi_network_mapping = [] 

        for aal3_idx, aal3_color in enumerate(original_131_aal3_colors):
            aal3_name = original_131_aal3_names[aal3_idx]
            aal3_roi_mask = (aal_data == aal3_color)
            
            if not np.any(aal3_roi_mask):
                logger.warning(f"AAL3 ROI color {aal3_color} ('{aal3_name}') not found in (potentially resampled) AAL3 NIfTI data. Assigning to NonCortical.")
                roi_network_mapping.append((aal3_color, aal3_name, 0, YEO17_LABELS_TO_NAMES[0], 0.0, aal3_idx))
                continue

            overlapping_yeo_voxels = yeo_data[aal3_roi_mask]
            
            if overlapping_yeo_voxels.size > 0:
                unique_yeo_labels, counts = np.unique(overlapping_yeo_voxels, return_counts=True)
                valid_overlap_mask = unique_yeo_labels != 0
                unique_yeo_labels = unique_yeo_labels[valid_overlap_mask]
                counts = counts[valid_overlap_mask]

                if len(counts) > 0:
                    winner_yeo_label_idx = np.argmax(counts)
                    winner_yeo_label = unique_yeo_labels[winner_yeo_label_idx]
                    total_roi_voxels = np.sum(aal3_roi_mask)
                    overlap_percentage = (counts[winner_yeo_label_idx] / total_roi_voxels) * 100 if total_roi_voxels > 0 else 0.0
                    yeo17_name = YEO17_LABELS_TO_NAMES.get(winner_yeo_label, f"UnknownYeo{winner_yeo_label}")
                    if overlap_percentage < 5.0 : 
                         logger.debug(f"AAL3 ROI {aal3_color} ('{aal3_name}') has low overlap ({overlap_percentage:.2f}%) with Yeo-17 Label {winner_yeo_label} ('{yeo17_name}'). May be subcortical or cerebellar.")
                else: 
                    winner_yeo_label = 0 
                    yeo17_name = YEO17_LABELS_TO_NAMES[0] 
                    overlap_percentage = 0.0
                    logger.debug(f"AAL3 ROI {aal3_color} ('{aal3_name}') has no overlap with cortical Yeo-17 networks. Assigning to NonCortical.")
            else: 
                winner_yeo_label = 0
                yeo17_name = YEO17_LABELS_TO_NAMES[0]
                overlap_percentage = 0.0
                logger.warning(f"AAL3 ROI {aal3_color} ('{aal3_name}') mask is empty in AAL3 NIfTI data. Assigning to NonCortical.")
            
            roi_network_mapping.append((aal3_color, aal3_name, winner_yeo_label, yeo17_name, overlap_percentage, aal3_idx ))
        
        roi_network_mapping_sorted = sorted(roi_network_mapping, key=lambda x: (x[2] == 0, x[2], x[0]))

        new_order_indices = [item[5] for item in roi_network_mapping_sorted] 
        roi_names_new_order = [item[1] for item in roi_network_mapping_sorted]
        network_labels_new_order = [item[3] for item in roi_network_mapping_sorted]
        
        if len(new_order_indices) != FINAL_N_ROIS_EXPECTED or len(set(new_order_indices)) != FINAL_N_ROIS_EXPECTED:
            logger.error("Error en la generación de new_order_indices para reordenamiento. Longitud o unicidad incorrecta.")
            return None

        logger.info("Successfully mapped AAL3 ROIs to Yeo-17 networks and determined new ROI order.")
        
        mapping_df = pd.DataFrame(roi_network_mapping_sorted, columns=['AAL3_Color', 'AAL3_Name', 'Yeo17_Label', 'Yeo17_Network', 'Overlap_Percent', 'Original_Index_0_N'])
        mapping_filename = BASE_PATH_AAL3 / f"aal3_{FINAL_N_ROIS_EXPECTED}_to_yeo17_mapping.csv"
        try:
            mapping_filename.parent.mkdir(parents=True, exist_ok=True) 
            mapping_df.to_csv(mapping_filename, index=False)
            logger.info(f"AAL3 to Yeo-17 mapping saved to: {mapping_filename}")
        except Exception as e_save_map:
            logger.warning(f"Could not save AAL3 to Yeo-17 mapping CSV: {e_save_map}")

        return {
            'order_name': 'aal3_to_yeo17_overlap_sorted',
            'roi_indices_original_order': list(range(FINAL_N_ROIS_EXPECTED)), 
            'roi_names_original_order': original_131_aal3_names, 
            'roi_names_new_order': roi_names_new_order,
            'network_labels_new_order': network_labels_new_order,
            'new_order_indices': new_order_indices 
        }

    except FileNotFoundError as e_fnf:
        logger.error(f"Atlas file not found during ROI reordering: {e_fnf}. ROI reordering will be skipped.")
        return None
    except Exception as e:
        logger.error(f"Error during ROI reordering: {e}", exc_info=True)
        return None

def _initialize_aal3_roi_processing_info(): # Definida ANTES de su llamada a nivel de módulo
    global VALID_AAL3_ROI_INFO_DF_166, AAL3_MISSING_INDICES_0BASED, \
           INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166, FINAL_N_ROIS_EXPECTED, \
           N_ROIS_EXPECTED, OUTPUT_CONNECTIVITY_DIR_NAME, CONNECTIVITY_CHANNEL_NAMES, N_CHANNELS, \
           PEARSON_OMST_CHANNEL_NAME, granger_suffix_global, AAL3_ROI_ORDER_MAPPING

    logger.info("--- Initializing AAL3 ROI Processing Information ---")
    
    omst_suffix_for_dir = "OMST_GCE_Signed" if OMST_PYTHON_LOADED and orthogonal_minimum_spanning_tree is not None and USE_PEARSON_OMST_CHANNEL else "PearsonFullSigned"
    current_pearson_channel_to_use_as_base = PEARSON_OMST_CHANNEL_NAME_PRIMARY if OMST_PYTHON_LOADED and USE_PEARSON_OMST_CHANNEL else PEARSON_OMST_FALLBACK_NAME
    
    channel_norm_suffix = "_ChNorm" 
    roi_reorder_suffix = "_ROIreorderedYeo17" 
    current_roi_order_suffix = "" 

    if not AAL3_META_PATH.exists():
        logger.error(f"AAL3 metadata file NOT found: {AAL3_META_PATH}. Cannot perform ROI reduction or reordering. "
                     f"Using placeholder N_ROIS_EXPECTED = {N_ROIS_EXPECTED}.")
        FINAL_N_ROIS_EXPECTED = N_ROIS_EXPECTED 
        AAL3_ROI_ORDER_MAPPING = None 
    else:
        try:
            meta_aal3_df = pd.read_csv(AAL3_META_PATH, sep='\t')
            meta_aal3_df['color'] = pd.to_numeric(meta_aal3_df['color'], errors='coerce')
            meta_aal3_df.dropna(subset=['color'], inplace=True)
            meta_aal3_df['color'] = meta_aal3_df['color'].astype(int)
            
            if not all(col in meta_aal3_df.columns for col in ['nom_c', 'color', 'vol_vox']):
                raise ValueError("AAL3 metadata must contain 'nom_c', 'color', 'vol_vox'.")

            AAL3_MISSING_INDICES_0BASED = [idx - 1 for idx in AAL3_MISSING_INDICES_1BASED]
            VALID_AAL3_ROI_INFO_DF_166 = meta_aal3_df[~meta_aal3_df['color'].isin(AAL3_MISSING_INDICES_1BASED)].copy()
            VALID_AAL3_ROI_INFO_DF_166.sort_values(by='color', inplace=True) 
            VALID_AAL3_ROI_INFO_DF_166.reset_index(drop=True, inplace=True)

            if len(VALID_AAL3_ROI_INFO_DF_166) != EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL:
                logger.warning(f"Expected {EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL} ROIs in AAL3 meta after filtering known missing, "
                               f"but found {len(VALID_AAL3_ROI_INFO_DF_166)}. Check AAL3_META_PATH content and AAL3_MISSING_INDICES_1BASED.")
            
            small_rois_mask_on_166 = VALID_AAL3_ROI_INFO_DF_166['vol_vox'] < SMALL_ROI_VOXEL_THRESHOLD
            INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166 = VALID_AAL3_ROI_INFO_DF_166[small_rois_mask_on_166].index.tolist()
            
            FINAL_N_ROIS_EXPECTED = EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL - len(INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166)
            N_ROIS_EXPECTED = FINAL_N_ROIS_EXPECTED 
            
            logger.info(f"AAL3 ROI processing info initialized (prior to reordering attempt):") # Log before reorder attempt
            logger.info(f"  Indices of 4 AAL3 systemically missing ROIs (0-based, from 170): {AAL3_MISSING_INDICES_0BASED}")
            logger.info(f"  Number of ROIs in AAL3 meta after excluding systemically missing: {len(VALID_AAL3_ROI_INFO_DF_166)} (Expected: {EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL})")
            logger.info(f"  Indices of small ROIs to drop (from the {len(VALID_AAL3_ROI_INFO_DF_166)} set, 0-based): {INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166}")
            logger.info(f"  Number of small ROIs to drop: {len(INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166)}")
            logger.info(f"  FINAL_N_ROIS_EXPECTED for connectivity analysis: {FINAL_N_ROIS_EXPECTED} (This should be 131 if matching QC script)")

            # Llamar a _get_aal3_network_mapping_and_order DESPUÉS de que las variables globales estén listas.
            AAL3_ROI_ORDER_MAPPING = _get_aal3_network_mapping_and_order() 

        except Exception as e:
            logger.error(f"Error initializing AAL3 ROI processing info or during reordering attempt: {e}", exc_info=True)
            FINAL_N_ROIS_EXPECTED = N_ROIS_EXPECTED # Fallback al valor global si existe
            AAL3_ROI_ORDER_MAPPING = None 
    
    current_roi_order_suffix = roi_reorder_suffix if AAL3_ROI_ORDER_MAPPING and AAL3_ROI_ORDER_MAPPING.get("new_order_indices") is not None else ""
    if current_roi_order_suffix:
        logger.info(f"ROI reordering WILL BE APPLIED based on '{AAL3_ROI_ORDER_MAPPING.get('order_name', 'custom') if AAL3_ROI_ORDER_MAPPING else 'unknown'}'.")
    else:
        logger.warning("ROI reordering is INACTIVE. Matrices will use default AAL3-derived order. Consider implementing for improved CNN performance and interpretability.")

    # Construir el nombre del directorio de salida
    base_name_for_dir = f"{OUTPUT_CONNECTIVITY_DIR_NAME_BASE}_AAL3_{N_ROIS_EXPECTED if N_ROIS_EXPECTED is not None else 'Unknown'}ROIs_{omst_suffix_for_dir}_{granger_suffix_global}{deconv_str}{channel_norm_suffix}{current_roi_order_suffix}"
    
    # Simplificar la lógica del nombre del directorio
    if FINAL_N_ROIS_EXPECTED is None or not AAL3_META_PATH.exists(): 
        OUTPUT_CONNECTIVITY_DIR_NAME = f"{base_name_for_dir}_ERR_INIT"
    elif roi_reorder_suffix and not current_roi_order_suffix: # Si se esperaba reordenar pero falló
        OUTPUT_CONNECTIVITY_DIR_NAME = f"{base_name_for_dir}_ERR_REORDER_FAIL"
    else:
        OUTPUT_CONNECTIVITY_DIR_NAME = f"{base_name_for_dir}_ParallelTuned"

            
    temp_channels = []
    if USE_PEARSON_OMST_CHANNEL:
        temp_channels.append(current_pearson_channel_to_use_as_base)
        if not (OMST_PYTHON_LOADED and orthogonal_minimum_spanning_tree is not None) and current_pearson_channel_to_use_as_base == PEARSON_OMST_CHANNEL_NAME_PRIMARY:
            logger.warning(f"OMST function from dyconnmap not loaded or is None, but primary OMST channel name was set. "
                           f"The channel '{PEARSON_OMST_CHANNEL_NAME_PRIMARY}' will effectively be '{PEARSON_OMST_FALLBACK_NAME}'.")
        elif not (OMST_PYTHON_LOADED and orthogonal_minimum_spanning_tree is not None):
            logger.info(f"OMST function from dyconnmap not loaded or is None. Using '{PEARSON_OMST_FALLBACK_NAME}' for the Pearson-based channel.")
    
    if USE_PEARSON_FULL_SIGNED_CHANNEL and current_pearson_channel_to_use_as_base != PEARSON_OMST_FALLBACK_NAME : 
        if PEARSON_OMST_FALLBACK_NAME not in temp_channels:
             temp_channels.append(PEARSON_OMST_FALLBACK_NAME) 

    if USE_MI_CHANNEL_FOR_THESIS: temp_channels.append("MI_KNN_Symmetric")
    if USE_DFC_ABS_DIFF_MEAN_CHANNEL: temp_channels.append("dFC_AbsDiffMean")
    if USE_DFC_STDDEV_CHANNEL: temp_channels.append("dFC_StdDev") 
    if USE_PRECISION_CHANNEL:
        temp_channels.append("Precision_FisherZ")

    if USE_DCOR_CHANNEL:
        temp_channels.append("DistanceCorr")

    
    if USE_GRANGER_CHANNEL: 
        granger_channel_name = f"Granger_F_lag{GRANGER_MAX_LAG}" 
        temp_channels.append(granger_channel_name)
    
    CONNECTIVITY_CHANNEL_NAMES = list(dict.fromkeys(temp_channels)) 
    N_CHANNELS = len(CONNECTIVITY_CHANNEL_NAMES)
    return True

# --- Llamada a la inicialización a nivel de módulo ---
# Esto se ejecuta cuando el script es importado o ejecutado.
# Asegurarse que todas las funciones que llama _initialize_aal3_roi_processing_info
# (como _get_aal3_network_mapping_and_order) estén definidas ANTES de esta llamada.
if not _initialize_aal3_roi_processing_info():
    logger.critical("CRITICAL: ROI processing info could not be initialized properly. Aborting pipeline.")
    exit() 
else:
    logger.info(f"Final N_ROIS_EXPECTED after initialization: {N_ROIS_EXPECTED}")
    logger.info(f"Final OUTPUT_CONNECTIVITY_DIR_NAME: {OUTPUT_CONNECTIVITY_DIR_NAME}")
    logger.info(f"Connectivity channels to be computed: {CONNECTIVITY_CHANNEL_NAMES}") 
    logger.info(f"Total number of channels (for VAE): {N_CHANNELS}")

# --- ADVERTENCIA IMPORTANTE SOBRE PREPROCESAMIENTO PREVIO ---
# (Sugerencia Tesis 1: Preprocesamiento & control de confundidos)
logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
logger.critical("!! CRÍTICO: Este script ASUME que las señales BOLD de los ROIs ya han sido PREPROCESADAS RIGUROSAMENTE !!")
logger.critical("!! La calidad de las matrices de conectividad depende DIRECTAMENTE de este preprocesamiento previo.     !!")
logger.critical("!! Este preprocesamiento DEBERÍA HABER INCLUIDO (como mínimo):                                       !!")
logger.critical("!!   - Corrección de movimiento (realineación).                                                       !!")
logger.critical("!!   - Coregistro a imagen anatómica T1w.                                                             !!")
logger.critical("!!   - Normalización a un espacio estándar (ej. MNI).                                                 !!")
logger.critical("!!   - SCRUBBING/CENSURA de volúmenes con movimiento excesivo (ej. FD > 0.5mm, DVARS anómalo).        !!")
logger.critical("!!     (Si no se hizo, considerar implementarlo ANTES o adaptando este script si se dispone de FD/DVARS por TP). !!")
logger.critical("!!     (Revisar umbrales de scrubbing; FD > 0.2mm con DVARS adaptativo podría ser más sensible).       !!")
logger.critical("!!   - REGRESIÓN DE CONFOUNDS:                                                                        !!")
logger.critical("!!     - Parámetros de movimiento (ej. 6 básicos + derivadas + cuadrados = 24 parámetros).            !!")
logger.critical("!!     - Señales medias de sustancia blanca (WM) y líquido cefalorraquídeo (CSF).                     !!")
logger.critical("!!     - Componentes de CompCor (anatómico o temporal, aCompCor/tCompCor) son altamente recomendables.  !!")
logger.critical("!!     - Regresión fisiológica avanzada (RETROICOR/PhysIO si se dispone de datos de pulso/respiración). !!")
logger.critical("!!     - Considerar ICA-AROMA o FIX para eliminar componentes de movimiento residuales.               !!")
logger.critical("!!     - Considerar Global Signal Regression (GSR) con conocimiento de sus efectos (puede inducir     !!")
logger.critical("!!       anti-correlaciones pero también mejorar la especificidad de la red). Evaluar con/sin GSR.    !!")
logger.critical("!!   - Filtrado temporal (ej. pasabanda) si no se realiza en este script (aquí se aplica 0.01-0.08Hz). !!")
logger.critical("!! SI ESTOS PASOS NO SE HAN REALIZADO, LA INTERPRETACIÓN NEUROCIENTÍFICA DE LOS RESULTADOS PUEDE VERSE !!")
logger.critical("!! SEVERAMENTE COMPROMETIDA POR ARTEFACTOS Y RUIDO.                                                 !!")
logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


# --- 1. Subject Metadata Loading and Merging ---
def load_metadata(
    subject_meta_csv_path: Path,
    qc_report_csv_path: Path) -> Optional[pd.DataFrame]:
    logger.info("--- Starting Subject Metadata Loading and QC Integration ---")
    try:
        if not subject_meta_csv_path.exists():
            logger.critical(f"Subject metadata CSV file NOT found: {subject_meta_csv_path}")
            return None
        if not qc_report_csv_path.exists():
            logger.critical(f"QC report CSV file NOT found: {qc_report_csv_path}")
            return None

        subjects_db_df = pd.read_csv(subject_meta_csv_path)
        subjects_db_df['SubjectID'] = subjects_db_df['SubjectID'].astype(str).str.strip() 
        logger.info(f"Loaded main metadata from {subject_meta_csv_path}. Shape: {subjects_db_df.shape}")
        if 'SubjectID' not in subjects_db_df.columns:
            logger.critical("Column 'SubjectID' missing in main metadata CSV.")
            return None
        if 'ResearchGroup' not in subjects_db_df.columns:
            logger.warning("Column 'ResearchGroup' missing in main metadata CSV. May be needed for downstream VAE tasks.")

        qc_df = pd.read_csv(qc_report_csv_path)
        logger.info(f"Loaded QC report from {qc_report_csv_path}. Shape: {qc_df.shape}")

        if 'Subject' in qc_df.columns and 'SubjectID' not in qc_df.columns:
            logger.info("Found 'Subject' column in QC report, renaming to 'SubjectID'.")
            qc_df.rename(columns={'Subject': 'SubjectID'}, inplace=True)
        
        if 'SubjectID' in qc_df.columns:
            qc_df['SubjectID'] = qc_df['SubjectID'].astype(str).str.strip()
        else:
            logger.critical("Neither 'Subject' nor 'SubjectID' column found in QC report CSV.")
            return None
        
        essential_qc_cols = ['SubjectID', 'ToDiscard_Overall', 'TimePoints']
        if not all(col in qc_df.columns for col in essential_qc_cols):
            logger.critical(f"Essential columns ({essential_qc_cols}) missing in QC report CSV.")
            return None

        merged_df = pd.merge(subjects_db_df, qc_df, on='SubjectID', how='inner', suffixes=('_meta', '_qc'))
        
        if 'TimePoints_qc' in merged_df.columns: 
            merged_df['Timepoints_final_for_script'] = merged_df['TimePoints_qc']
        elif 'TimePoints' in merged_df.columns: 
             merged_df['Timepoints_final_for_script'] = merged_df['TimePoints']
        else: 
            logger.critical("Definitive 'TimePoints' column from QC report could not be identified after merge.")
            return None
        
        merged_df['Timepoints_final_for_script'] = pd.to_numeric(merged_df['Timepoints_final_for_script'], errors='coerce').fillna(0).astype(int)

        initial_subject_count = len(merged_df)
        # SUGEERENCIA NEUROCIENCIA/ANÁLISIS DE DATOS (Sugerencia Tesis 1.2): Revisar los criterios que definen 'ToDiscard_Overall'.
        # La revisión doctoral sugiere que los umbrales podrían ser conservadores (ej. permitir hasta ~17.5% outliers multivariantes).
        # Considerar si umbrales más estrictos (ej. <10% outliers MV y <5% univariados, o umbrales FD/DVARS estrictos)
        # mejorarían la calidad del dataset final, aunque reduzca ligeramente el N. Un dataset más limpio es a menudo preferible.
        # Comprobar que la tasa de descarte sea similar entre grupos (AD vs CN) para evitar sesgos.
        logger.info("SUGGESTION (Data Quality): Review 'ToDiscard_Overall' criteria from QC script. "
                    "Consider if stricter thresholds for subject exclusion (e.g., based on percentage of multivariate outliers, mean FD, DVARS) "
                    "would yield a cleaner dataset for modeling, even if N is slightly reduced. Ensure discard rate is similar across groups.")
        subjects_passing_qc_df = merged_df[merged_df['ToDiscard_Overall'] == False].copy()
        num_discarded = initial_subject_count - len(subjects_passing_qc_df)
        
        logger.info(f"Total subjects after merge: {initial_subject_count}")
        logger.info(f"Subjects discarded based on QC ('ToDiscard_Overall' == True): {num_discarded}")
        logger.info(f"Subjects passing QC and to be processed: {len(subjects_passing_qc_df)}")

        if subjects_passing_qc_df.empty:
            logger.warning("No subjects passed QC. Check your QC criteria and report.")
            return None 
            
        min_tp_after_qc = subjects_passing_qc_df['Timepoints_final_for_script'].min()
        max_tp_after_qc = subjects_passing_qc_df['Timepoints_final_for_script'].max()
        logger.info(f"Timepoints for subjects passing QC (from QC report): Min={min_tp_after_qc}, Max={max_tp_after_qc}.")
        logger.info(f"These will be homogenized to TARGET_LEN_TS = {TARGET_LEN_TS} for connectivity calculation.")

        final_cols_to_keep = ['SubjectID']
        subjects_passing_qc_df.rename(columns={'Timepoints_final_for_script': 'Timepoints'}, inplace=True)
        final_cols_to_keep.append('Timepoints')

        if 'ResearchGroup_meta' in subjects_passing_qc_df.columns: 
            subjects_passing_qc_df.rename(columns={'ResearchGroup_meta': 'ResearchGroup'}, inplace=True)
        elif 'ResearchGroup_qc' in subjects_passing_qc_df.columns and 'ResearchGroup' not in subjects_passing_qc_df.columns:
            subjects_passing_qc_df.rename(columns={'ResearchGroup_qc': 'ResearchGroup'}, inplace=True)
        
        if 'ResearchGroup' in subjects_passing_qc_df.columns:
             final_cols_to_keep.append('ResearchGroup')
        else:
            logger.warning("Creating placeholder 'ResearchGroup' column as it was not found. This is important for classification.")
            subjects_passing_qc_df['ResearchGroup'] = 'Unknown' 
            final_cols_to_keep.append('ResearchGroup')
        
        final_cols_to_keep = list(dict.fromkeys(final_cols_to_keep)) 
        return subjects_passing_qc_df[final_cols_to_keep]

    except FileNotFoundError as e:
        logger.critical(f"CRITICAL Error loading CSV files: {e}")
        return None
    except ValueError as e:
        logger.critical(f"Value error in metadata processing: {e}")
        return None
    except Exception as e:
        logger.critical(f"Unexpected error during metadata loading/QC integration: {e}", exc_info=True)
        return None






# --- Funciones para Reordenamiento de ROIs (Definidas ANTES de _initialize_aal3_roi_processing_info) ---
# (La definición de _get_aal3_network_mapping_and_order ya está arriba)

def _reorder_rois_by_network_for_timeseries(
    timeseries_data: np.ndarray, 
    new_order_indices: List[int],
    subject_id: str) -> np.ndarray:
    if new_order_indices is None or len(new_order_indices) != timeseries_data.shape[1]:
        # logger.warning(f"S {subject_id}: No se proporcionaron índices de reordenamiento válidos o no coinciden con el número de ROIs ({timeseries_data.shape[1]}). No se reordenarán las series temporales.") # Verbose
        return timeseries_data
    
    logger.info(f"S {subject_id}: Reordenando series temporales de ROIs ({timeseries_data.shape}) según el nuevo orden de redes (longitud de índices: {len(new_order_indices)}).")
    return timeseries_data[:, new_order_indices]

def _reorder_connectivity_matrix_by_network(
    matrix: np.ndarray, 
    new_order_indices: List[int],
    subject_id: str,
    channel_name: str) -> np.ndarray:
    if new_order_indices is None or len(new_order_indices) != matrix.shape[0]:
        # logger.warning(f"S {subject_id}, Canal {channel_name}: No se proporcionaron índices de reordenamiento válidos o no coinciden con la dimensión de la matriz ({matrix.shape[0]}). No se reordenará la matriz de conectividad.") # Verbose
        return matrix

    logger.info(f"S {subject_id}, Canal {channel_name}: Reordenando matriz de conectividad ({matrix.shape}) según el nuevo orden de redes (longitud de índices: {len(new_order_indices)}).")
    return matrix[np.ix_(new_order_indices, new_order_indices)]


# --- 2. Time Series Loading and Preprocessing Functions ---
def _load_signals_from_mat(mat_path: Path, possible_keys: List[str]) -> Optional[np.ndarray]:
    try:
        data = sio.loadmat(mat_path)
    except Exception as e_load:
        logger.error(f"Could not load .mat file: {mat_path}. Error: {e_load}")
        return None
    
    for key in possible_keys:
        if key in data and isinstance(data[key], np.ndarray) and data[key].ndim >= 2:
            # logger.debug(f"Found signals under key '{key}' in {mat_path.name}. Shape: {data[key].shape}") # Verbose
            return data[key].astype(np.float64) 
            
    logger.warning(f"No valid signal keys {possible_keys} found in {mat_path.name}. Keys present: {list(data.keys())}")
    return None

def _orient_and_reduce_rois(
    raw_sigs: np.ndarray, 
    subject_id: str,
    initial_expected_cols: int, 
    aal3_missing_0based: Optional[List[int]], 
    small_rois_indices_from_166: Optional[List[int]], 
    final_expected_rois: Optional[int] 
) -> Optional[np.ndarray]:
    if raw_sigs.ndim != 2:
        logger.warning(f"S {subject_id}: Raw signal matrix has incorrect dimensions {raw_sigs.ndim} (expected 2). Skipping.")
        return None
    
    oriented_sigs = raw_sigs.copy()
    if oriented_sigs.shape[0] == initial_expected_cols and oriented_sigs.shape[1] != initial_expected_cols:
        logger.info(f"S {subject_id}: Transposing raw matrix from {oriented_sigs.shape} to ({oriented_sigs.shape[1]}, {oriented_sigs.shape[0]}) to match (TPs, ROIs_initial).")
        oriented_sigs = oriented_sigs.T
    elif oriented_sigs.shape[1] == initial_expected_cols and oriented_sigs.shape[0] != initial_expected_cols:
        pass 
    elif oriented_sigs.shape[0] == initial_expected_cols and oriented_sigs.shape[1] == initial_expected_cols:
         logger.warning(f"S {subject_id}: Raw signal matrix is square ({oriented_sigs.shape}) and matches initial_expected_cols. Assuming [Timepoints, ROIs_initial]. Careful if TPs also equals initial_expected_cols.")
    else: 
        logger.warning(f"S {subject_id}: Neither dimension of raw signal matrix ({oriented_sigs.shape}) matches initial_expected_cols ({initial_expected_cols}). Skipping.")
        return None

    if oriented_sigs.shape[1] != initial_expected_cols: 
        logger.warning(f"S {subject_id}: After orientation, raw ROI count ({oriented_sigs.shape[1]}) != initial_expected_cols ({initial_expected_cols}). Skipping.")
        return None
    
    if aal3_missing_0based is None:
        logger.warning(f"S {subject_id}: AAL3 missing ROI indices (0-based) not available. Skipping AAL3 known missing ROI removal. Using {oriented_sigs.shape[1]} ROIs for next step.")
        sigs_after_known_missing_removed = oriented_sigs 
    else:
        try:
            sigs_after_known_missing_removed = np.delete(oriented_sigs, aal3_missing_0based, axis=1)
            if sigs_after_known_missing_removed.shape[1] != EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL:
                 logger.warning(f"S {subject_id}: After removing known missing ROIs, shape is {sigs_after_known_missing_removed.shape}, but expected (..., {EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL}).")
        except IndexError as e:
            logger.error(f"S {subject_id}: IndexError removing known missing AAL3 ROIs (indices: {aal3_missing_0based}) from matrix of shape {oriented_sigs.shape}. Error: {e}. Using original {oriented_sigs.shape[1]} ROIs for next step.")
            sigs_after_known_missing_removed = oriented_sigs 
            
    if small_rois_indices_from_166 is None:
        logger.warning(f"S {subject_id}: Small ROI indices (from 166-set) not available. Skipping small ROI removal. Using {sigs_after_known_missing_removed.shape[1]} ROIs.")
        sigs_final_rois = sigs_after_known_missing_removed
    elif sigs_after_known_missing_removed.shape[1] != EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL:
        logger.warning(f"S {subject_id}: Cannot remove small ROIs because the matrix (shape {sigs_after_known_missing_removed.shape}) does not have the expected {EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL} columns after first reduction step. Using current ROIs ({sigs_after_known_missing_removed.shape[1]}).")
        sigs_final_rois = sigs_after_known_missing_removed
    else:
        try:
            sigs_final_rois = np.delete(sigs_after_known_missing_removed, small_rois_indices_from_166, axis=1)
        except IndexError as e:
            logger.error(f"S {subject_id}: IndexError removing small ROIs (indices: {small_rois_indices_from_166}) from matrix of shape {sigs_after_known_missing_removed.shape}. Error: {e}. Using {sigs_after_known_missing_removed.shape[1]} ROIs.")
            sigs_final_rois = sigs_after_known_missing_removed 

    if final_expected_rois is not None and sigs_final_rois.shape[1] != final_expected_rois:
        logger.warning(f"S {subject_id}: Final ROI count ({sigs_final_rois.shape[1]}) != FINAL_N_ROIS_EXPECTED ({final_expected_rois}). "
                       "This may indicate issues in AAL3 metadata or reduction logic. Proceeding with current matrix.")
    elif final_expected_rois is None:
        logger.warning(f"S {subject_id}: FINAL_N_ROIS_EXPECTED is None. Cannot validate final ROI count. Proceeding with {sigs_final_rois.shape[1]} ROIs.")
        
    return sigs_final_rois

def _bandpass_filter_signals(sigs: np.ndarray, lowcut: float, highcut: float, fs: float, order: int, subject_id: str, taper_alpha: float = 0.1) -> np.ndarray:
    nyquist_freq = 0.5 * fs
    low_norm = lowcut / nyquist_freq
    high_norm = highcut / nyquist_freq

    if not (0 < low_norm < 1 and 0 < high_norm < 1 and low_norm < high_norm):
        logger.error(f"S {subject_id}: Invalid critical frequencies for filter (low_norm={low_norm}, high_norm={high_norm}). Nyquist={nyquist_freq}. Skipping filtering.")
        return sigs
    try:
        b, a = butter(order, [low_norm, high_norm], btype='band', analog=False)
        filtered_sigs = np.zeros_like(sigs)
        padlen_required = 3 * (max(len(a), len(b))) 
        
        for i in range(sigs.shape[1]): 
            roi_signal = sigs[:, i].copy() 
            
            if len(roi_signal) > padlen_required: 
                try:
                    tukey_window = windows.tukey(len(roi_signal), alpha=taper_alpha)
                    roi_signal_tapered = roi_signal * tukey_window
                except Exception as e_taper:
                    logger.warning(f"S {subject_id}, ROI {i}: Error applying Tukey window: {e_taper}. Proceeding without taper.")
                    roi_signal_tapered = roi_signal 
            else:
                roi_signal_tapered = roi_signal 

            if np.all(np.isclose(roi_signal_tapered, roi_signal_tapered[0] if len(roi_signal_tapered)>0 else 0.0)): 
                filtered_sigs[:, i] = roi_signal_tapered 
            elif len(roi_signal_tapered) <= padlen_required :
                logger.warning(f"S {subject_id}, ROI {i}: Signal too short ({len(roi_signal_tapered)} pts, need > {padlen_required}) for filtfilt. Skipping filter for this ROI.")
                filtered_sigs[:, i] = roi_signal_tapered
            else:
                filtered_sigs[:, i] = filtfilt(b, a, roi_signal_tapered)
        return filtered_sigs
    except Exception as e:
        logger.error(f"S {subject_id}: Error during bandpass filtering: {e}. Returning original signals.", exc_info=False)
        return sigs

def _hrf_deconvolution(sigs: np.ndarray, tr: float, hrf_model_type: str, subject_id: str) -> np.ndarray:
    logger.info(f"S {subject_id}: Attempting HRF deconvolution (Model: {hrf_model_type}, TR: {tr}s).")
    if hrf_model_type == 'glover': 
        hrf_kernel = glover_hrf(tr, oversampling=1) 
    elif hrf_model_type == 'spm': 
        hrf_kernel = spm_hrf(tr, oversampling=1)
    else: 
        logger.error(f"S {subject_id}: Unknown HRF model type '{hrf_model_type}'. Skipping deconvolution.")
        return sigs

    if len(hrf_kernel) == 0 or np.all(np.isclose(hrf_kernel, 0)):
        logger.error(f"S {subject_id}: HRF kernel is empty or all zeros for model '{hrf_model_type}'. Skipping deconvolution.")
        return sigs

    deconvolved_sigs = np.zeros_like(sigs)
    for i in range(sigs.shape[1]): 
        signal_roi = sigs[:, i]
        if len(signal_roi) < len(hrf_kernel): 
            logger.warning(f"S {subject_id}, ROI {i}: Signal length ({len(signal_roi)}) is shorter than HRF kernel length ({len(hrf_kernel)}). Skipping deconvolution for this ROI.")
            deconvolved_sigs[:, i] = signal_roi
            continue
        try:
            quotient, _ = deconvolve(signal_roi, hrf_kernel)
            if len(quotient) < sigs.shape[0]:
                deconvolved_sigs[:, i] = np.concatenate([quotient, np.zeros(sigs.shape[0] - len(quotient))])
            else: 
                deconvolved_sigs[:, i] = quotient[:sigs.shape[0]]
        except Exception as e_deconv:
            logger.error(f"S {subject_id}, ROI {i}: Deconvolution failed: {e_deconv}. Using original signal for this ROI.", exc_info=False)
            deconvolved_sigs[:, i] = signal_roi
            
    logger.info(f"S {subject_id}: HRF deconvolution attempt finished.")
    return deconvolved_sigs

def _preprocess_time_series(
    sigs: np.ndarray, 
    target_len_ts_val: int, 
    subject_id: str, 
    eff_conn_max_lag_val: int, 
    tr_seconds_val: float, low_cut_val: float, high_cut_val: float, filter_order_val: int,
    apply_hrf_deconv_val: bool, hrf_model_type_val: str,
    taper_alpha_val: float
) -> Optional[np.ndarray]:
    original_length, current_n_rois = sigs.shape
    fs = 1.0 / tr_seconds_val 
    
    logger.info(f"S {subject_id}: Preprocessing. Input TPs: {original_length}, ROIs: {current_n_rois} (should be {FINAL_N_ROIS_EXPECTED}), TR: {tr_seconds_val}s. Target TPs for output: {target_len_ts_val}.")
    
    sigs_processed = _bandpass_filter_signals(sigs, low_cut_val, high_cut_val, fs, filter_order_val, subject_id, taper_alpha=taper_alpha_val)
    
    if apply_hrf_deconv_val:
        sigs_processed = _hrf_deconvolution(sigs_processed, tr_seconds_val, hrf_model_type_val, subject_id)
        if np.isnan(sigs_processed).any() or np.isinf(sigs_processed).any():
            logger.warning(f"S {subject_id}: NaNs/Infs detected after HRF deconvolution. Cleaning by replacing with 0.0.")
            sigs_processed = np.nan_to_num(sigs_processed, nan=0.0, posinf=0.0, neginf=0.0)
            
    min_len_for_granger_var = eff_conn_max_lag_val + 10 
    min_len_for_dfc = DFC_WIN_POINTS if DFC_WIN_POINTS > 0 else 5 
    min_overall_len = max(5, min_len_for_granger_var, min_len_for_dfc) 
    if sigs_processed.shape[0] < min_overall_len:
        logger.warning(f"S {subject_id}: Timepoints after processing ({sigs_processed.shape[0]}) are less than minimum required ({min_overall_len}) for all connectivity measures. Skipping subject.")
        return None
        
    if np.isnan(sigs_processed).any():
        logger.warning(f"S {subject_id}: NaNs detected in signals before scaling. Filling with 0.0. This might affect results.")
        sigs_processed = np.nan_to_num(sigs_processed, nan=0.0) 
        
    try:
        scaler = StandardScaler() 
        sigs_normalized = scaler.fit_transform(sigs_processed)
        if np.isnan(sigs_normalized).any(): 
            logger.warning(f"S {subject_id}: NaNs detected after StandardScaler. Filling with 0.0. This is unusual.")
            sigs_normalized = np.nan_to_num(sigs_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    except ValueError as e_scale: 
        logger.warning(f"S {subject_id}: StandardScaler failed (e.g. all-zero data after processing): {e_scale}. Attempting column-wise scaling or zeroing.")
        sigs_normalized = np.zeros_like(sigs_processed, dtype=np.float32)
        for i in range(sigs_processed.shape[1]):
            col_data = sigs_processed[:, i].reshape(-1,1)
            if np.std(col_data) > 1e-9: 
                try: 
                    sigs_normalized[:, i] = StandardScaler().fit_transform(col_data).flatten()
                except Exception as e_col_scale:
                    logger.error(f"S {subject_id}, ROI {i}: Column-wise scaling failed: {e_col_scale}. Setting to zero.")
                    sigs_normalized[:, i] = 0.0 
            else: 
                sigs_normalized[:, i] = 0.0 
        if np.isnan(sigs_normalized).any(): 
            sigs_normalized = np.nan_to_num(sigs_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    current_length_norm, num_rois_norm = sigs_normalized.shape
    if current_length_norm != target_len_ts_val:
        logger.info(f"S {subject_id}: Homogenizing time series length from {current_length_norm} to {target_len_ts_val}.")
        if current_length_norm < target_len_ts_val:
            # logger.debug(f"S {subject_id}: Interpolating from {current_length_norm} to {target_len_ts_val} points.") # Verbose
            sigs_homogenized = np.zeros((target_len_ts_val, num_rois_norm), dtype=np.float32)
            if current_length_norm > 1: 
                x_old = np.linspace(0, 1, current_length_norm)
                x_new = np.linspace(0, 1, target_len_ts_val)
                for i in range(num_rois_norm):
                    f_interp = interp1d(x_old, sigs_normalized[:, i], kind='linear', fill_value="extrapolate")
                    sigs_homogenized[:, i] = f_interp(x_new)
            elif current_length_norm == 1: 
                 for i in range(num_rois_norm):
                    sigs_homogenized[:,i] = sigs_normalized[0,i] 

            if np.isnan(sigs_homogenized).any(): 
                logger.warning(f"S {subject_id}: NaNs found after interpolation/length adjustment. Filling with 0.0.")
                sigs_homogenized = np.nan_to_num(sigs_homogenized, nan=0.0)
        else: 
            sigs_homogenized = sigs_normalized[:target_len_ts_val, :]
    else:
        sigs_homogenized = sigs_normalized 
        
    return sigs_homogenized.astype(np.float32)

def load_and_preprocess_single_subject_series(
    subject_id: str, 
    target_len_ts_val: int,
    current_roi_signals_dir_path: Path, current_roi_filename_template: str,
    possible_roi_keys_list: List[str], 
    eff_conn_max_lag_val: int, 
    tr_seconds_val: float, low_cut_val: float, high_cut_val: float, filter_order_val: int,
    apply_hrf_deconv_val: bool, hrf_model_type_val: str,
    taper_alpha_val: float,
    roi_order_info: Optional[Dict[str, Any]] 
) -> Tuple[Optional[np.ndarray], str, bool]:
    mat_path = current_roi_signals_dir_path / current_roi_filename_template.format(subject_id=subject_id)
    if not mat_path.exists(): 
        return None, f"MAT file not found: {mat_path.name}", False
    
    try:
        loaded_sigs_raw_170 = _load_signals_from_mat(mat_path, possible_roi_keys_list)
        if loaded_sigs_raw_170 is None: 
            return None, f"No valid signal keys or load error in {mat_path.name}", False
        
        sigs_reduced_rois = _orient_and_reduce_rois(
            loaded_sigs_raw_170, subject_id, 
            RAW_DATA_EXPECTED_COLUMNS, 
            AAL3_MISSING_INDICES_0BASED, 
            INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166, 
            FINAL_N_ROIS_EXPECTED 
        )
        del loaded_sigs_raw_170; gc.collect() 
        if sigs_reduced_rois is None: 
            return None, f"ROI orientation, reduction, or validation failed for S {subject_id}.", False
        
        if FINAL_N_ROIS_EXPECTED is not None and sigs_reduced_rois.shape[1] != FINAL_N_ROIS_EXPECTED:
            error_msg = (f"S {subject_id}: Post-reduction ROI count ({sigs_reduced_rois.shape[1]}) "
                         f"does not match FINAL_N_ROIS_EXPECTED ({FINAL_N_ROIS_EXPECTED}).")
            logger.error(error_msg)
            return None, error_msg, False
        elif FINAL_N_ROIS_EXPECTED is None:
             logger.warning(f"S {subject_id}: FINAL_N_ROIS_EXPECTED is None, cannot strictly validate ROI count. Proceeding with {sigs_reduced_rois.shape[1]} ROIs.")

        # --- INICIO: Reordenamiento de ROIs para Series Temporales ---
        if roi_order_info and roi_order_info.get("new_order_indices") is not None:
            new_indices = roi_order_info["new_order_indices"]
            if len(new_indices) == sigs_reduced_rois.shape[1]:
                sigs_reduced_rois = _reorder_rois_by_network_for_timeseries(sigs_reduced_rois, new_indices, subject_id)
            # El log de advertencia ya está en _reorder_rois_by_network_for_timeseries si hay mismatch
        # --- FIN: Reordenamiento de ROIs para Series Temporales ---

        original_tp_count = sigs_reduced_rois.shape[0]
        
        sigs_processed = _preprocess_time_series(
            sigs_reduced_rois, target_len_ts_val,
            subject_id, eff_conn_max_lag_val, 
            tr_seconds_val, low_cut_val, high_cut_val, filter_order_val,
            apply_hrf_deconv_val, hrf_model_type_val,
            taper_alpha_val=taper_alpha_val 
        )
        del sigs_reduced_rois; gc.collect() 
        if sigs_processed is None: 
            return None, f"Preprocessing (filtering, scaling, or length adjustment) failed for S {subject_id}. Original TPs: {original_tp_count}", False
        
        final_shape_str = f"({sigs_processed.shape[0]}, {sigs_processed.shape[1]})"
        if FINAL_N_ROIS_EXPECTED is not None and sigs_processed.shape[1] != FINAL_N_ROIS_EXPECTED:
            error_msg = (f"S {subject_id}: Processed signal ROI count ({sigs_processed.shape[1]}) "
                         f"mismatches FINAL_N_ROIS_EXPECTED ({FINAL_N_ROIS_EXPECTED}) after all preprocessing. "
                         "This could indicate an issue with ROI reordering logic if active, or prior reduction.")
            logger.error(error_msg)
            return None, error_msg, False

        logger.info(f"S {subject_id}: Successfully loaded and preprocessed. Original TPs: {original_tp_count}, Final Shape for conn: {final_shape_str}")
        return sigs_processed, f"OK. Original TPs: {original_tp_count}, final shape for conn: {final_shape_str}", True
        
    except Exception as e:
        logger.error(f"Unhandled exception during load_and_preprocess for S {subject_id} ({mat_path.name}): {e}", exc_info=True)
        return None, f"Exception processing {mat_path.name}: {str(e)}", False

# --- 3. Connectivity Calculation Functions ---
def fisher_r_to_z(r_matrix: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    r_clean = np.nan_to_num(r_matrix.astype(np.float32), nan=0.0) 
    r_clipped = np.clip(r_clean, -1.0 + eps, 1.0 - eps)
    z_matrix = np.arctanh(r_clipped)
    np.fill_diagonal(z_matrix, 0.0) 
    return z_matrix.astype(np.float32)

def calculate_pearson_full_fisher_z_signed(ts_subject: np.ndarray, sid: str) -> Optional[np.ndarray]: 
    if ts_subject.shape[0] < 2:
        logger.warning(f"Pearson_Full_FisherZ_Signed (S {sid}): Insufficient timepoints ({ts_subject.shape[0]} < 2).")
        return None
    try:
        corr_matrix = np.corrcoef(ts_subject, rowvar=False).astype(np.float32)
        if corr_matrix.ndim == 0: 
            logger.warning(f"Pearson_Full_FisherZ_Signed (S {sid}): Correlation resulted in a scalar. Input shape: {ts_subject.shape}.")
            num_rois = ts_subject.shape[1]
            return np.zeros((num_rois, num_rois), dtype=np.float32) if num_rois > 0 else None
        
        z_transformed_matrix = fisher_r_to_z(corr_matrix) 
        # logger.info(f"Pearson_Full_FisherZ_Signed (S {sid}): Successfully calculated.") # Can be verbose
        return z_transformed_matrix
    except Exception as e:
        logger.error(f"Error calculating Pearson_Full_FisherZ_Signed for S {sid}: {e}", exc_info=True)
        return None

def calculate_pearson_omst_signed_weighted(ts_subject: np.ndarray, sid: str) -> Optional[np.ndarray]: 
    if not OMST_PYTHON_LOADED or orthogonal_minimum_spanning_tree is None:
        logger.error(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): Dyconnmap OMST function not available. Cannot calculate.")
        return None 
    
    if ts_subject.shape[0] < 2: 
        logger.warning(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): Insufficient timepoints ({ts_subject.shape[0]} < 2).")
        return None
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning) 
            
            corr_matrix = np.corrcoef(ts_subject, rowvar=False).astype(np.float32)
            
            if corr_matrix.ndim == 0: 
                logger.warning(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): Correlation resulted in a scalar. Input shape: {ts_subject.shape}. Returning zero matrix.")
                num_rois = ts_subject.shape[1]
                return np.zeros((num_rois, num_rois), dtype=np.float32) if num_rois > 0 else None
                
            z_transformed_matrix = fisher_r_to_z(corr_matrix) 
            weights_for_omst_gce = np.abs(z_transformed_matrix) 
            np.fill_diagonal(weights_for_omst_gce, 0.0) 

            if np.all(np.isclose(weights_for_omst_gce, 0)):
                 logger.warning(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): All input weights for OMST GCE are zero. Returning zero matrix (original Z-transformed).")
                 return z_transformed_matrix.astype(np.float32) 
                 
            # logger.info(f"S {sid}: Calling dyconnmap.threshold_omst_global_cost_efficiency with ABSOLUTE weights shape {weights_for_omst_gce.shape}") # Verbose
            
            omst_outputs = orthogonal_minimum_spanning_tree(weights_for_omst_gce, n_msts=None) 
            
            if isinstance(omst_outputs, tuple) and len(omst_outputs) >= 2:
                omst_adjacency_matrix_gce_weighted = np.asarray(omst_outputs[1]).astype(np.float32) 
                # logger.debug(f"S {sid}: dyconnmap.threshold_omst_global_cost_efficiency returned multiple outputs. Using the second one (CIJtree) as omst_adjacency_matrix.") # Verbose
            else:
                logger.error(f"S {sid}: dyconnmap.threshold_omst_global_cost_efficiency returned an unexpected type or insufficient outputs: {type(omst_outputs)}. Cannot extract OMST matrix.")
                return None

            if not isinstance(omst_adjacency_matrix_gce_weighted, np.ndarray): 
                logger.error(f"S {sid}: Extracted omst_adjacency_matrix_gce_weighted is not a numpy array (type: {type(omst_adjacency_matrix_gce_weighted)}). Cannot proceed.")
                return None
            
            binary_omst_mask = (omst_adjacency_matrix_gce_weighted > 0).astype(int)
            signed_weighted_omst_matrix = z_transformed_matrix * binary_omst_mask
            np.fill_diagonal(signed_weighted_omst_matrix, 0.0) 
            
            # logger.info(f"Pearson_OMST_GCE_Signed_Weighted (S {sid}): Successfully calculated. Matrix density: {np.count_nonzero(signed_weighted_omst_matrix) / signed_weighted_omst_matrix.size:.4f}") # Verbose
            return signed_weighted_omst_matrix.astype(np.float32)

    except AttributeError as ae:
        if 'from_numpy_matrix' in str(ae).lower() or 'from_numpy_array' in str(ae).lower(): 
            logger.error(f"Error calculating Pearson_OMST_GCE_Signed_Weighted (dyconnmap) for S {sid}: NetworkX version incompatibility. "
                         f"Dyconnmap (v1.0.4) may be using a deprecated NetworkX function. "
                         f"Your NetworkX version: {nx.__version__}. Consider using NetworkX 2.x. Original error: {ae}", exc_info=False) 
        else:
            logger.error(f"AttributeError calculating Pearson_OMST_GCE_Signed_Weighted (dyconnmap) for S {sid}: {ae}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error calculating Pearson_OMST_GCE_Signed_Weighted (dyconnmap) connectivity for S {sid}: {e}", exc_info=True)
        return None
    
import numpy as np
from sklearn.covariance import LedoitWolf, GraphicalLasso
from joblib import Parallel, delayed
from sklearn.utils.validation import check_random_state

from sklearn.covariance import LedoitWolf, GraphicalLasso
from sklearn.utils.validation import check_random_state
import numpy as np
import warnings


from sklearn.covariance import GraphicalLassoCV, LedoitWolf
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.preprocessing import RobustScaler
import numpy as np
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning
from typing import Optional

from sklearn.covariance import GraphicalLassoCV
import numpy as np
import sklearn

def calculate_precision_partial_corr_fisher_z(ts, sid,
                                              cv_folds=3,
                                              n_alphas=40,
                                              max_iter=1500,
                                              use_scaler=True):

    if use_scaler:
        ts = RobustScaler().fit_transform(ts)

    rng = check_random_state(42)
    p = ts.shape[1]
    best_prec = None            # <- make sure it exists

    try:
        skl_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
        if skl_version < (1, 3):
            glasso = GraphicalLassoCV(n_alphas=n_alphas, cv=cv_folds,
                                      max_iter=max_iter, n_jobs=-1,
                                      assume_centered=True)
        else:
            alpha_grid = np.logspace(-4, 0, n_alphas)
            glasso = GraphicalLassoCV(alphas=alpha_grid, cv=cv_folds,
                                      max_iter=max_iter, n_jobs=-1,
                                      assume_centered=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            glasso.fit(ts)
            best_prec = glasso.precision_
            logger.info(f"S {sid}: GLassoCV converged. α*={glasso.alpha_:.4g}")

    except Exception as e:
        logger.error(f"S {sid}: GLassoCV failed – {e}. Falling back to Ledoit-Wolf.")
        try:
            lw = LedoitWolf(assume_centered=True).fit(ts)
            best_prec = lw.precision_
            logger.info(f"S {sid}: Ledoit-Wolf fallback succeeded.")
        except Exception as e2:
            logger.error(f"S {sid}: Ledoit-Wolf fallback failed – {e2}.")
            return None

    # From here on best_prec is guaranteed to exist
    d = np.sqrt(np.diag(best_prec))
    d[d < 1e-8] = 1.0
    partial_corr = -best_prec / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)
    z = fisher_r_to_z(partial_corr)
    return z.astype(np.float32)






# 2) Usar dcor para distancia si tu SciPy es <1.10
def calculate_distance_correlation(ts_subject: np.ndarray, sid: str) -> Optional[np.ndarray]:
    """
    Calcula la Distance Correlation usando la librería 'dcor'.
    """
    try:
        import itertools
        n_rois = ts_subject.shape[1]
        distcorr_mat = np.zeros((n_rois, n_rois), dtype=np.float32)

        # Pre-calcular las series temporales para evitar rebanado repetido
        series = [ts_subject[:, i] for i in range(n_rois)]

        for i, j in itertools.combinations(range(n_rois), 2):
            # Usar la función importada directamente
            dc = distance_correlation(series[i], series[j])
            distcorr_mat[i, j] = distcorr_mat[j, i] = dc
        return distcorr_mat
    except Exception as e:
        logger.error(f"DistanceCorr (S {sid}): Error durante el cálculo: {e}")
        return None


def _calculate_mi_for_pair(X_i_reshaped, y_j, n_neighbors_val):    
    try:
        mi_val = mutual_info_regression(X_i_reshaped, y_j, n_neighbors=n_neighbors_val, random_state=42, discrete_features=False)[0]
        return mi_val
    except Exception:
        return 0.0 

def calculate_mi_knn_connectivity(ts_subject: np.ndarray, n_neighbors_val: int, sid: str) -> Optional[np.ndarray]:
    n_tp, n_rois = ts_subject.shape
    if n_tp == 0: 
        logger.warning(f"MI_KNN (S {sid}): 0 Timepoints provided. Cannot calculate MI.")
        return None
    if n_tp <= n_neighbors_val: 
        logger.warning(f"MI_KNN (S {sid}): Timepoints ({n_tp}) <= n_neighbors ({n_neighbors_val}). Skipping MI calculation.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None
        
    mi_matrix = np.zeros((n_rois, n_rois), dtype=np.float32)
    
    tasks = []
    for i in range(n_rois):
        for j in range(i + 1, n_rois): 
            tasks.append({'i': i, 'j': j, 
                          'data_i': ts_subject[:, i].reshape(-1, 1), 
                          'data_j': ts_subject[:, j], 
                          'n_neighbors': n_neighbors_val})

    global MAX_WORKERS, TOTAL_CPU_CORES 
    if MAX_WORKERS == 1: 
        n_jobs_mi = max(1, TOTAL_CPU_CORES - 1 if TOTAL_CPU_CORES > 1 else 1) 
    else: 
        n_jobs_mi = max(1, TOTAL_CPU_CORES // MAX_WORKERS if MAX_WORKERS > 0 else 1) 
    # logger.debug(f"MI_KNN (S {sid}): Using n_jobs={n_jobs_mi} for joblib.Parallel. Global MAX_WORKERS for subjects: {MAX_WORKERS}") # Verbose

    try:
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore", ConvergenceWarning)
            results_list = Parallel(n_jobs=n_jobs_mi)(
                delayed(_calculate_mi_for_pair)(task['data_i'], task['data_j'], task['n_neighbors']) 
                for task in tasks
            )
            results_list_ji = Parallel(n_jobs=n_jobs_mi)( 
                delayed(_calculate_mi_for_pair)(task['data_j'].reshape(-1,1), task['data_i'].flatten(), task['n_neighbors'])
                for task in tasks
            )
    except Exception as e_parallel:
        logger.error(f"MI_KNN (S {sid}): Error during parallel MI calculation: {e_parallel}. Falling back to serial.")
        results_list = [_calculate_mi_for_pair(task['data_i'], task['data_j'], task['n_neighbors']) for task in tasks]
        results_list_ji = [_calculate_mi_for_pair(task['data_j'].reshape(-1,1), task['data_i'].flatten(), task['n_neighbors']) for task in tasks]

    for k, task in enumerate(tasks):
        i, j = task['i'], task['j']
        mi_val_ij = results_list[k]
        mi_val_ji = results_list_ji[k]
        mi_matrix[i, j] = mi_matrix[j, i] = (mi_val_ij + mi_val_ji) / 2.0
        # if mi_val_ij == 0.0 and mi_val_ji == 0.0 and (np.std(ts_subject[:,i]) > 1e-6 and np.std(ts_subject[:,j]) > 1e-6) : # Verbose
             # logger.debug(f"MI_KNN (S {sid}): MI for pair ({i},{j}) resulted in 0.0 (possibly due to error in _calculate_mi_for_pair or true zero MI).")

    # logger.info(f"MI_KNN_Symmetric (S {sid}): Successfully calculated.") # Can be verbose
    return mi_matrix

def calculate_custom_dfc_abs_diff_mean(ts_subject: np.ndarray, win_points_val: int, step_val: int, sid: str) -> Optional[np.ndarray]:
    n_tp, n_rois = ts_subject.shape
    if n_tp < win_points_val: 
        logger.warning(f"dFC_AbsDiffMean (S {sid}): Timepoints ({n_tp}) < window length ({win_points_val}). Skipping.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None 
        
    num_windows = (n_tp - win_points_val) // step_val + 1
    if num_windows < 2: 
        logger.warning(f"dFC_AbsDiffMean (S {sid}): Fewer than 2 windows ({num_windows}) can be formed. Skipping.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None 
        
    sum_abs_diff_matrix = np.zeros((n_rois, n_rois), dtype=np.float64) 
    n_diffs_calculated = 0
    prev_corr_matrix_abs: Optional[np.ndarray] = None
    
    for idx in range(num_windows):
        start_idx = idx * step_val
        end_idx = start_idx + win_points_val
        window_ts = ts_subject[start_idx:end_idx, :]
        
        if window_ts.shape[0] < 2: continue 
            
        try:
            corr_matrix_window = np.corrcoef(window_ts, rowvar=False)
            if corr_matrix_window.ndim < 2 or corr_matrix_window.shape != (n_rois, n_rois):
                logger.warning(f"dFC_AbsDiffMean (S {sid}), Window {idx}: corrcoef returned unexpected shape {corr_matrix_window.shape}. Using zeros for this window's contribution.")
                corr_matrix_window = np.full((n_rois, n_rois), 0.0, dtype=np.float32) 
            else:
                corr_matrix_window = np.nan_to_num(corr_matrix_window.astype(np.float32), nan=0.0) 
            
            current_corr_matrix_abs = np.abs(corr_matrix_window)
            np.fill_diagonal(current_corr_matrix_abs, 0) 
            
            if prev_corr_matrix_abs is not None:
                sum_abs_diff_matrix += np.abs(current_corr_matrix_abs - prev_corr_matrix_abs)
                n_diffs_calculated += 1
            prev_corr_matrix_abs = current_corr_matrix_abs
        except Exception as e: 
            logger.error(f"dFC_AbsDiffMean (S {sid}), Window {idx}: Error calculating/processing correlation: {e}")
            
    if n_diffs_calculated == 0: 
        logger.warning(f"dFC_AbsDiffMean (S {sid}): No valid differences between windowed correlations were calculated. Returning zero matrix.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None
        
    mean_abs_diff_matrix = (sum_abs_diff_matrix / n_diffs_calculated).astype(np.float32)
    np.fill_diagonal(mean_abs_diff_matrix, 0) 
    return mean_abs_diff_matrix

def calculate_dfc_std_dev(ts_subject: np.ndarray, win_points_val: int, step_val: int, sid: str) -> Optional[np.ndarray]: 
    n_tp, n_rois = ts_subject.shape
    if n_tp < win_points_val:
        logger.warning(f"dFC_StdDev (S {sid}): Timepoints ({n_tp}) < window length ({win_points_val}). Skipping.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None
        
    num_windows = (n_tp - win_points_val) // step_val + 1
    if num_windows < 2: 
        logger.warning(f"dFC_StdDev (S {sid}): Fewer than 2 windows ({num_windows}) can be formed. StdDev would be trivial (0). Skipping and returning zero matrix.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None
        
    window_corr_matrices_list = []
    
    for idx in range(num_windows):
        start_idx = idx * step_val
        end_idx = start_idx + win_points_val
        window_ts = ts_subject[start_idx:end_idx, :]
        
        if window_ts.shape[0] < 2: continue 
            
        try:
            corr_matrix_window = np.corrcoef(window_ts, rowvar=False)
            if corr_matrix_window.ndim < 2 or corr_matrix_window.shape != (n_rois, n_rois):
                logger.warning(f"dFC_StdDev (S {sid}), Window {idx}: corrcoef returned unexpected shape {corr_matrix_window.shape}. Skipping this window for StdDev.")
                continue 
            else:
                corr_matrix_window = np.nan_to_num(corr_matrix_window.astype(np.float32), nan=0.0) 
            
            np.fill_diagonal(corr_matrix_window, 0) 
            window_corr_matrices_list.append(corr_matrix_window)
        except Exception as e: 
            logger.error(f"dFC_StdDev (S {sid}), Window {idx}: Error calculating/processing correlation: {e}")
            
    if len(window_corr_matrices_list) < 2: 
        logger.warning(f"dFC_StdDev (S {sid}): Fewer than 2 valid windowed correlation matrices were calculated ({len(window_corr_matrices_list)}). Cannot compute StdDev. Returning zero matrix.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None
        
    stacked_corr_matrices = np.stack(window_corr_matrices_list, axis=0) 
    std_dev_matrix = np.std(stacked_corr_matrices, axis=0).astype(np.float32)
    np.fill_diagonal(std_dev_matrix, 0) 
    
    # logger.info(f"dFC_StdDev (S {sid}): Successfully calculated from {len(window_corr_matrices_list)} windows.") # Can be verbose
    return std_dev_matrix

def _granger_pair(ts1, ts2, maxlag, sid, i, j):     
    f_ij, f_ji = 0.0, 0.0
    try:
        data_for_ij = np.column_stack([ts2, ts1]) 
        if np.any(np.std(data_for_ij, axis=0) < 1e-6): 
             # logger.debug(f"S {sid}: GC pair ({i}->{j}): Datos con varianza casi nula. Saltando F_ij.") # Verbose
             pass
        else:
            granger_result_ij = grangercausalitytests(data_for_ij, maxlag=[maxlag], verbose=False)
            f_ij = granger_result_ij[maxlag][0]['ssr_ftest'][0] 
        
        data_for_ji = np.column_stack([ts1, ts2]) 
        if np.any(np.std(data_for_ji, axis=0) < 1e-6):
            # logger.debug(f"S {sid}: GC pair ({j}->{i}): Datos con varianza casi nula. Saltando F_ji.") # Verbose
            pass
        else:
            granger_result_ji = grangercausalitytests(data_for_ji, maxlag=[maxlag], verbose=False)
            f_ji = granger_result_ji[maxlag][0]['ssr_ftest'][0]
            
        return f_ij, f_ji
    except Exception as e:
        # logger.debug(f"S {sid}: GC pair ({i},{j}) failed: {e}. Returning (0.0, 0.0)") # Verbose
        return 0.0, 0.0 
        
def calculate_granger_f_matrix(ts_subject: np.ndarray, maxlag: int, sid: str) -> Optional[np.ndarray]: 
    n_tp, n_rois = ts_subject.shape
    if n_tp <= maxlag * 4 + 5: 
        logger.warning(f"Granger (S {sid}): Too few TPs ({n_tp}) for lag {maxlag} and {n_rois} ROIs. Need > ~{maxlag * 4 + 5}.")
        return np.zeros((n_rois, n_rois), dtype=np.float32) if n_rois > 0 else None
    
    gc_mat_symmetric = np.zeros((n_rois, n_rois), dtype=np.float32)
    
    tasks = []
    for i in range(n_rois):
        for j in range(i + 1, n_rois): 
            tasks.append({'i': i, 'j': j, 
                          'ts1': ts_subject[:, i], 
                          'ts2': ts_subject[:, j], 
                          'maxlag': maxlag, 'sid': sid})
    
    global MAX_WORKERS, TOTAL_CPU_CORES
    if MAX_WORKERS == 1:
        n_jobs_granger = max(1, TOTAL_CPU_CORES - 1 if TOTAL_CPU_CORES > 1 else 1) 
    else:
        n_jobs_granger = max(1, TOTAL_CPU_CORES // MAX_WORKERS if MAX_WORKERS > 0 else 1)
    # logger.debug(f"Granger (S {sid}): Using n_jobs={n_jobs_granger} for joblib.Parallel. Global MAX_WORKERS for subjects: {MAX_WORKERS}") # Verbose

    try:
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore", FutureWarning)
            results_pairs = Parallel(n_jobs=n_jobs_granger)(
                delayed(_granger_pair)(task['ts1'], task['ts2'], task['maxlag'], task['sid'], task['i'], task['j']) 
                for task in tasks
            )
    except Exception as e_parallel_granger:
        logger.error(f"Granger (S {sid}): Error during parallel Granger calculation: {e_parallel_granger}. Falling back to serial.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            results_pairs = [_granger_pair(task['ts1'], task['ts2'], task['maxlag'], task['sid'], task['i'], task['j']) for task in tasks]

    for k, task in enumerate(tasks):
        i, j = task['i'], task['j']
        f_ij, f_ji = results_pairs[k]
        f_sym = (f_ij + f_ji) / 2.0 
        gc_mat_symmetric[i, j] = gc_mat_symmetric[j, i] = f_sym
            
    np.fill_diagonal(gc_mat_symmetric, 0) 
    # logger.info(f"Granger_F_lag{maxlag} (S {sid}): done.") # Can be verbose
    return gc_mat_symmetric

# --- 4. Function to Calculate All Connectivity Modalities ---
# SUGEERENCIA NEUROCIENCIA/DEEP LEARNING (Sugerencia Tesis 5):
# Considerar añadir métricas de conectividad alternativas/complementarias como canales adicionales.
# Esto se haría añadiendo nuevas funciones de cálculo y luego añadiendo sus nombres a CONNECTIVITY_CHANNEL_NAMES.
# 1. Precisión (Inverso de Covarianza) con Graphical Lasso: 
#    - from sklearn.covariance import GraphicalLassoCV
#    - cov_estimator = GraphicalLassoCV(cv=3).fit(subject_ts_data) # Ajustar CV folds
#    - precision_matrix = cov_estimator.precision_
#    - Podría reemplazar o complementar Pearson_Full si se busca conectividad parcial.
# 2. Edge-Time-Series (ETS) / Co-fluctuation Patterns:
#    - Calcular series temporales de co-fluctuación para cada par de ROIs.
#    - Extraer características de estas ETS (ej. media de amplitud, varianza, CAPs de ETS).
#    - Shine J. (2023) "Untangling edge-time series" para más detalles.
# 3. Modelos de Estados (HMM / CAPs):
#    - Estimar estados cerebrales dinámicos (ej. 4-6 estados) usando HMM (hmmlearn) o k-means sobre ventanas (CAPs).
#    - Derivar métricas como tiempo de ocupación por estado, probabilidades de transición, dwell-time.
#    - Estas podrían ser features adicionales o incluso canales si se mapean a una matriz.
# 4. Métricas de Topología de Grafos (Persistencia Homológica, MST):
#    - Calcular métricas globales/nodales de la red (ej. eficiencia, modularidad, grado) a partir de una de las matrices (ej. OMST o Pearson umbralizada).
#    - Podrían añadirse como features vectoriales o como un canal diagonal si son nodales.
def calculate_all_connectivity_modalities_for_subject(
    subject_id: str, subject_ts_data: np.ndarray,
    n_neighbors_mi_param: int,
    dfc_win_points_param: int, dfc_step_param: int,
    granger_lag_param: int,
    roi_order_info: Optional[Dict[str, Any]] 
) -> Dict[str, Any]:
    matrices: Dict[str, Optional[np.ndarray]] = {name: None for name in CONNECTIVITY_CHANNEL_NAMES}
    errors_in_calculation: Dict[str, str] = {}
    timings: Dict[str, float] = {}

    for channel_name in CONNECTIVITY_CHANNEL_NAMES:
        logger.info(f"Calculating {channel_name} for S {subject_id} (TS shape: {subject_ts_data.shape})...")
        start_time_channel = time.time()
        matrix_result: Optional[np.ndarray] = None
        error_msg: Optional[str] = None
        is_omst_primary_with_fallback_pending = False 

        try:
            if channel_name == PEARSON_OMST_CHANNEL_NAME_PRIMARY: 
                is_omst_primary_with_fallback_pending = (PEARSON_OMST_FALLBACK_NAME in CONNECTIVITY_CHANNEL_NAMES and
                                                         matrices.get(PEARSON_OMST_FALLBACK_NAME) is None)
                if OMST_PYTHON_LOADED and orthogonal_minimum_spanning_tree is not None:
                    matrix_result = calculate_pearson_omst_signed_weighted(subject_ts_data, subject_id) 
                    if matrix_result is None: 
                        error_msg = f"Primary OMST GCE (signed) calculation failed for S {subject_id}."
                        logger.error(error_msg) 
                else: 
                    error_msg = f"OMST function not loaded, cannot calculate '{PEARSON_OMST_CHANNEL_NAME_PRIMARY}' for S {subject_id}."
                    logger.error(error_msg)
                    if PEARSON_OMST_FALLBACK_NAME not in CONNECTIVITY_CHANNEL_NAMES:
                         matrices[channel_name] = None 
            
            elif channel_name == PEARSON_OMST_FALLBACK_NAME: 
                 matrix_result = calculate_pearson_full_fisher_z_signed(subject_ts_data, subject_id) 
                 if matrix_result is None:
                     error_msg = f"Fallback/Full Pearson calculation failed for S {subject_id}."
                     logger.error(error_msg)
                 matrices[channel_name] = matrix_result 

            elif channel_name == "MI_KNN_Symmetric" and USE_MI_CHANNEL_FOR_THESIS: 
                matrix_result = calculate_mi_knn_connectivity(subject_ts_data, n_neighbors_mi_param, subject_id)
                matrices[channel_name] = matrix_result

            elif channel_name == "dFC_AbsDiffMean" and USE_DFC_ABS_DIFF_MEAN_CHANNEL:
                matrix_result = calculate_custom_dfc_abs_diff_mean(subject_ts_data, dfc_win_points_param, dfc_step_param, subject_id)
                matrices[channel_name] = matrix_result
            
            elif channel_name == "dFC_StdDev" and USE_DFC_STDDEV_CHANNEL: 
                matrix_result = calculate_dfc_std_dev(subject_ts_data, dfc_win_points_param, dfc_step_param, subject_id)
                matrices[channel_name] = matrix_result
            
            elif channel_name.startswith("Granger_F_lag") and USE_GRANGER_CHANNEL: 
                try:
                    current_lag = int(channel_name.split("lag")[-1].split("_")[0]) 
                except (IndexError, ValueError):
                    logger.error(f"Could not parse lag from Granger channel name: {channel_name}. Using granger_lag_param ({granger_lag_param}).")
                    current_lag = granger_lag_param
                matrix_result = calculate_granger_f_matrix(subject_ts_data, current_lag, subject_id)
                matrices[channel_name] = matrix_result

            elif channel_name == "Precision_FisherZ" and USE_PRECISION_CHANNEL:
                matrix_result = calculate_precision_partial_corr_fisher_z(subject_ts_data, subject_id)
                matrices[channel_name] = matrix_result

            elif channel_name == "DistanceCorr" and USE_DCOR_CHANNEL:
                matrix_result = calculate_distance_correlation(subject_ts_data, subject_id)
                matrices[channel_name] = matrix_result

            
            if matrix_result is not None and channel_name == PEARSON_OMST_CHANNEL_NAME_PRIMARY:
                 matrices[channel_name] = matrix_result 
            elif matrix_result is not None and channel_name != PEARSON_OMST_CHANNEL_NAME_PRIMARY : 
                 matrices[channel_name] = matrix_result


            if matrix_result is None and error_msg is None and channel_name in CONNECTIVITY_CHANNEL_NAMES and matrices.get(channel_name) is None:
                error_msg = f"'{channel_name}' was in CONNECTIVITY_CHANNEL_NAMES but not calculated or its function returned None without specific error."
                logger.warning(error_msg)
                matrices[channel_name] = None 

        except Exception as e: 
            error_msg = str(e)
            logger.error(f"Unexpected error while attempting to calculate {channel_name} for S {subject_id}: {e}", exc_info=True)
            if channel_name in matrices: matrices[channel_name] = None 
        
        if error_msg and channel_name not in errors_in_calculation: 
            if not (channel_name == PEARSON_OMST_CHANNEL_NAME_PRIMARY and 
                    PEARSON_OMST_FALLBACK_NAME in CONNECTIVITY_CHANNEL_NAMES and
                    matrices.get(PEARSON_OMST_FALLBACK_NAME) is None): 
                 errors_in_calculation[channel_name] = error_msg
        
        timings[f"{channel_name}_time_sec"] = time.time() - start_time_channel
        current_matrix_for_log = matrices.get(channel_name) 
        if current_matrix_for_log is not None:
            logger.info(f"{channel_name} for S {subject_id} calculated. Shape: {current_matrix_for_log.shape}. Took {timings[f'{channel_name}_time_sec']:.2f}s.")
        elif channel_name in CONNECTIVITY_CHANNEL_NAMES:
            is_omst_primary_with_fallback_pending_for_log = (
                channel_name == PEARSON_OMST_CHANNEL_NAME_PRIMARY and
                PEARSON_OMST_FALLBACK_NAME in CONNECTIVITY_CHANNEL_NAMES and
                matrices.get(PEARSON_OMST_FALLBACK_NAME) is None
            )
            if not is_omst_primary_with_fallback_pending_for_log:
                logger.warning(f"{channel_name} for S {subject_id} failed or returned None. Took {timings[f'{channel_name}_time_sec']:.2f}s. Error: {errors_in_calculation.get(channel_name, 'N/A')}")
            
    if USE_PEARSON_OMST_CHANNEL and PEARSON_OMST_CHANNEL_NAME_PRIMARY in CONNECTIVITY_CHANNEL_NAMES:
        if matrices.get(PEARSON_OMST_CHANNEL_NAME_PRIMARY) is None and matrices.get(PEARSON_OMST_FALLBACK_NAME) is not None:
            logger.info(f"S {subject_id}: Using fallback '{PEARSON_OMST_FALLBACK_NAME}' data for primary OMST channel '{PEARSON_OMST_CHANNEL_NAME_PRIMARY}'.")
            matrices[PEARSON_OMST_CHANNEL_NAME_PRIMARY] = matrices[PEARSON_OMST_FALLBACK_NAME]

    successful_count = 0
    final_errors_summary = {}

    for ch_name in CONNECTIVITY_CHANNEL_NAMES:
        if matrices.get(ch_name) is not None:
            successful_count += 1
        elif ch_name in errors_in_calculation:
            final_errors_summary[ch_name] = errors_in_calculation[ch_name]
        else:
            final_errors_summary[ch_name] = "Matrix is None, specific error not logged or calculation skipped."


    if successful_count < N_CHANNELS: 
        logger.warning(f"Connectivity for S {subject_id}: {successful_count}/{N_CHANNELS} selected modalities computed. Errors: {final_errors_summary}")
    else:
        logger.info(f"Connectivity for S {subject_id}: All {successful_count}/{N_CHANNELS} selected modalities computed successfully.")

    return {"matrices": matrices, "errors_conn_calc": final_errors_summary, "timings_conn_calc": timings}

# --- 5. Per-Subject Processing Pipeline (for Multiprocessing) ---
def process_single_subject_pipeline(subject_row_tuple: Tuple[int, pd.Series]) -> Dict[str, Any]:
    idx, subject_row = subject_row_tuple
    subject_id = str(subject_row['SubjectID']).strip()
    process = psutil.Process(os.getpid()) 
    ram_initial_mb = process.memory_info().rss / (1024**2)
    
    result: Dict[str, Any] = {
        "id": subject_id, 
        "status_preprocessing": "PENDING", "detail_preprocessing": "",
        "status_connectivity_calc": "NOT_ATTEMPTED", "errors_connectivity_calc": {},
        "timings_connectivity_calc_sec": {}, "path_saved_tensor": None,
        "status_overall": "PENDING",
        "ram_usage_mb_initial": ram_initial_mb, "ram_usage_mb_final": -1.0
    }
    series_data: Optional[np.ndarray] = None 

    try:
        eff_conn_lag_for_preprocess = GRANGER_MAX_LAG if USE_GRANGER_CHANNEL else 1 
        
        current_roi_order_info = AAL3_ROI_ORDER_MAPPING 

        series_data, detail_msg_preproc, success_preproc = load_and_preprocess_single_subject_series(
            subject_id, 
            TARGET_LEN_TS,
            ROI_SIGNALS_DIR_PATH_AAL3, ROI_FILENAME_TEMPLATE, POSSIBLE_ROI_KEYS,
            eff_conn_lag_for_preprocess, 
            TR_SECONDS, LOW_CUT_HZ, HIGH_CUT_HZ, FILTER_ORDER, 
            APPLY_HRF_DECONVOLUTION, HRF_MODEL,
            taper_alpha_val=TAPER_ALPHA,
            roi_order_info=current_roi_order_info 
        )
        result["status_preprocessing"] = "SUCCESS" if success_preproc else "FAILED"
        result["detail_preprocessing"] = detail_msg_preproc
        
        if not success_preproc or series_data is None:
            result["status_overall"] = "PREPROCESSING_FAILED"
            return result 

        connectivity_results = calculate_all_connectivity_modalities_for_subject(
            subject_id, series_data, N_NEIGHBORS_MI,
            DFC_WIN_POINTS, DFC_STEP, 
            GRANGER_MAX_LAG,
            roi_order_info=current_roi_order_info 
        )
        del series_data; series_data = None; gc.collect() 
        
        calculated_matrices_dict = connectivity_results["matrices"]
        result["errors_connectivity_calc"] = connectivity_results["errors_conn_calc"]
        result["timings_connectivity_calc_sec"] = connectivity_results.get("timings_conn_calc", {})

        all_modalities_valid_and_present = True
        final_matrices_to_stack_list = []
        
        current_expected_rois_for_matrices = FINAL_N_ROIS_EXPECTED if FINAL_N_ROIS_EXPECTED is not None else N_ROIS_EXPECTED 
        if current_expected_rois_for_matrices is None: 
            logger.critical(f"S {subject_id}: CRITICAL - current_expected_rois_for_matrices is None. Cannot validate matrix shapes.")
            result["status_overall"] = "FAILURE_CRITICAL_ROI_COUNT_UNSET"
            result["status_connectivity_calc"] = "FAILURE_CRITICAL_ROI_COUNT_UNSET"
            return result

        expected_matrix_shape = (current_expected_rois_for_matrices, current_expected_rois_for_matrices)

        for channel_name in CONNECTIVITY_CHANNEL_NAMES: 
            matrix = calculated_matrices_dict.get(channel_name)
            if matrix is None:
                all_modalities_valid_and_present = False
                err_msg = f"Modality '{channel_name}' result is None (check calculation logs for S {subject_id})." 
                logger.error(f"S {subject_id}: {err_msg}")
                if channel_name not in result["errors_connectivity_calc"]: 
                    result["errors_connectivity_calc"][channel_name] = err_msg
                break 
            elif matrix.shape != expected_matrix_shape:
                all_modalities_valid_and_present = False
                err_msg = f"Modality '{channel_name}' shape {matrix.shape} != expected {expected_matrix_shape}."
                logger.error(f"S {subject_id}: {err_msg}")
                result["errors_connectivity_calc"][channel_name] = result["errors_connectivity_calc"].get(channel_name, "") + (" | " if result["errors_connectivity_calc"].get(channel_name) else "") + err_msg
                break 
            else:
                # --- Normalización por Canal con RobustScaler (mejorada para excluir diagonal del fit) ---
                n_rois_ch = matrix.shape[0]
                scaled_matrix = np.zeros_like(matrix, dtype=np.float32)
                
                if n_rois_ch > 1: 
                    off_diag_mask = ~np.eye(n_rois_ch, dtype=bool)
                    off_diag_values = matrix[off_diag_mask]

                    if off_diag_values.size > 0: 
                        scaler = RobustScaler()
                        try:
                            scaled_off_diag_values = scaler.fit_transform(off_diag_values.reshape(-1,1)).flatten()
                            scaled_matrix[off_diag_mask] = scaled_off_diag_values
                            # logger.info(f"S {subject_id}: Channel '{channel_name}' off-diagonal elements normalized with RobustScaler.") # Can be verbose
                        except ValueError as e_scale_channel: 
                            logger.warning(f"S {subject_id}: RobustScaler failed for off-diagonal elements of channel '{channel_name}': {e_scale_channel}. Using original matrix for this channel.")
                            scaled_matrix = matrix.astype(np.float32) 
                    else: 
                        logger.warning(f"S {subject_id}: Channel '{channel_name}' has no off-diagonal elements or they are constant. Using original matrix.")
                        scaled_matrix = matrix.astype(np.float32)
                else: 
                     scaled_matrix = matrix.astype(np.float32)

                final_matrices_to_stack_list.append(scaled_matrix)
                # --- Fin Normalización por Canal ---
        
        if all_modalities_valid_and_present and len(final_matrices_to_stack_list) == N_CHANNELS: 
            result["status_connectivity_calc"] = "SUCCESS_ALL_MODALITIES_VALID_AND_NORMALIZED"
            try:
                subject_tensor = np.stack(final_matrices_to_stack_list, axis=0).astype(np.float32)
                del final_matrices_to_stack_list, calculated_matrices_dict; gc.collect() 
                
                output_dir_individual_tensors = BASE_PATH_AAL3 / OUTPUT_CONNECTIVITY_DIR_NAME / "individual_subject_tensors"
                output_dir_individual_tensors.mkdir(parents=True, exist_ok=True) 
                
                output_path = output_dir_individual_tensors / f"tensor_{N_CHANNELS}ch_{current_expected_rois_for_matrices}rois_{subject_id}.npz"
                
                save_metadata_dict = {
                    'tensor_data': subject_tensor, 
                    'subject_id': subject_id, 
                    'channel_names': np.array(CONNECTIVITY_CHANNEL_NAMES, dtype=str),
                    'rois_count': current_expected_rois_for_matrices,
                    'target_len_ts': TARGET_LEN_TS,
                    'channel_normalization_method': "RobustScaler_per_channel_per_subject_off_diagonal"
                }
                if current_roi_order_info and current_roi_order_info.get("new_order_indices") is not None:
                    save_metadata_dict['roi_order_name'] = current_roi_order_info.get('order_name', 'custom_network_order')
                    if 'roi_names_new_order' in current_roi_order_info:
                        save_metadata_dict['roi_names_in_order'] = np.array(current_roi_order_info.get('roi_names_new_order', []), dtype=str)
                    if 'network_labels_new_order' in current_roi_order_info: 
                        save_metadata_dict['network_labels_in_order'] = np.array(current_roi_order_info.get('network_labels_new_order', []), dtype=str)
                else:
                    save_metadata_dict['roi_order_name'] = 'aal3_original_reduced_order' 
                
                np.savez_compressed(output_path, **save_metadata_dict)
                result["path_saved_tensor"] = str(output_path)
                result["status_overall"] = "SUCCESS_ALL_PROCESSED_AND_SAVED"
                logger.info(f"S {subject_id}: Successfully processed. Tensor (with channel normalization) saved to {output_path.name}")
                del subject_tensor; gc.collect() 
            except Exception as e_save:
                logger.error(f"Error saving tensor for S {subject_id}: {e_save}", exc_info=True)
                result["errors_connectivity_calc"]["save_error"] = str(e_save)
                result["status_overall"] = "FAILURE_DURING_TENSOR_SAVING"
                result["status_connectivity_calc"] = "FAILURE_DURING_SAVING"
        else: 
            result["status_overall"] = "FAILURE_IN_CONNECTIVITY_CALC_VALIDATION_OR_NORMALIZATION"
            result["status_connectivity_calc"] = "FAILURE_MISSING_INVALID_OR_UNNORMALIZED_MODALITIES"
            if not all_modalities_valid_and_present: 
                logger.error(f"S {subject_id}: Not all connectivity modalities were valid or present. Tensor not saved. Errors: {result['errors_connectivity_calc']}")
            elif len(final_matrices_to_stack_list) != N_CHANNELS:
                logger.error(f"S {subject_id}: Number of matrices to stack ({len(final_matrices_to_stack_list)}) does not match N_CHANNELS ({N_CHANNELS}). Tensor not saved.")
    
    except Exception as e_pipeline: 
        logger.critical(f"CRITICAL UNHANDLED EXCEPTION for S {subject_id} in pipeline: {e_pipeline}", exc_info=True)
        result["status_overall"] = "CRITICAL_PIPELINE_EXCEPTION"
        result["detail_preprocessing"] = result.get("detail_preprocessing","") + " | Pipeline Exc: " + str(e_pipeline)
        result["errors_connectivity_calc"]["pipeline_exception"] = str(e_pipeline)
    
    finally: 
        if series_data is not None: del series_data; gc.collect() 
        result["ram_usage_mb_final"] = process.memory_info().rss / (1024**2)
    return result

# --- 6. Main Script Execution Flow ---
# SUGEERENCIA (Mantenibilidad): Refactorizar main() en funciones más pequeñas:
#  - def parse_arguments(): (usando argparse)
#  - def setup_pipeline(args): (configurar paths, logging, etc.)
#  - def load_data_and_qc(args): (llama a load_metadata)
#  - def run_subject_processing(subject_df, args): (bucle con ProcessPoolExecutor)
#  - def assemble_and_finalize_tensor(results_list, args):
#  - def run_pipeline(): (llama a las anteriores en orden)
def _normalize_global_tensor_inter_channel(global_tensor: np.ndarray, train_indices: np.ndarray, method: str = 'zscore_channels_train_params') -> Tuple[np.ndarray, Optional[Dict]]:    
    """
    Normaliza el tensor global para que cada tipo de canal tenga una escala comparable,
    calculando parámetros SOLO en el conjunto de entrenamiento.
    global_tensor: (N_subjects, N_channels, N_ROIs, N_ROIs)
    train_indices: Índices de los sujetos que pertenecen al conjunto de entrenamiento.
    method: 'zscore_channels_train_params' u otro.
    
    Retorna el tensor normalizado y los parámetros de normalización (para aplicar a test/val).
    """
    logger.info(f"Applying inter-channel normalization (method: {method}) using training set parameters.")
    normalized_global_tensor = global_tensor.copy() 
    norm_params = {'method': method, 'params_per_channel': []}

    if method == 'zscore_channels_train_params':
        for c_idx in range(global_tensor.shape[1]): 
            channel_data_train = global_tensor[train_indices, c_idx, :, :]
            
            off_diag_mask_ch = ~np.eye(channel_data_train.shape[1], dtype=bool) 
            
            all_off_diag_train_values = []
            for subj_idx_in_train in range(channel_data_train.shape[0]):
                all_off_diag_train_values.extend(channel_data_train[subj_idx_in_train][off_diag_mask_ch])
            
            if not all_off_diag_train_values: 
                mean_val = 0.0
                std_val = 0.0
            else:
                mean_val = np.mean(all_off_diag_train_values) 
                std_val = np.std(all_off_diag_train_values)   
            
            norm_params['params_per_channel'].append({'mean': mean_val, 'std': std_val})
            
            if std_val > 1e-9: 
                for subj_glob_idx in range(global_tensor.shape[0]):
                    current_matrix = global_tensor[subj_glob_idx, c_idx, :, :]
                    scaled_matrix_ch = current_matrix.copy()
                    scaled_matrix_ch[off_diag_mask_ch] = (current_matrix[off_diag_mask_ch] - mean_val) / std_val
                    normalized_global_tensor[subj_glob_idx, c_idx, :, :] = scaled_matrix_ch
                logger.info(f"Global tensor: Channel {c_idx} off-diagonal z-scored using train_mean={mean_val:.3f}, train_std={std_val:.3f}.")
            else:
                logger.warning(f"Global tensor: Channel {c_idx} has zero/low std in training set off-diagonal elements ({std_val:.3e}). Not scaling this channel.")
        return normalized_global_tensor, norm_params
    else:
        logger.warning(f"Inter-channel normalization method '{method}' not implemented. Returning original tensor.")
        return global_tensor, None

def main():
    # SUGEERENCIA (Mantenibilidad): Usar argparse para configurar parámetros desde la línea de comandos.
    # parser = argparse.ArgumentParser(description="fMRI Connectivity Feature Extraction Pipeline")
    # parser.add_argument("--base_path", type=str, default="/home/diego/Escritorio/AAL3", help="Base path for data")
    # ... otros argumentos ...
    # args = parser.parse_args()
    # Luego usar args.base_path en lugar de BASE_PATH_AAL3, etc.

    try:
        logger.info(f"RUNTIME NetworkX version being used: {nx.__version__}")
        if nx.__version__ != '2.6.3' and OMST_PYTHON_LOADED:
            logger.warning(f"Dyconnmap's OMST typically requires networkx 2.6.3 for full compatibility. "
                           f"Current version is {nx.__version__}. This might lead to errors with OMST.")
    except ImportError:
        logger.error("RUNTIME: NetworkX is not installed or importable.")

    np.random.seed(42) 

    script_start_time = time.time()
    main_process_info = psutil.Process(os.getpid())
    logger.info(f"Main process RAM at start: {main_process_info.memory_info().rss / (1024**2):.2f} MB")
    logger.info(f"--- Starting fMRI Connectivity Pipeline (Version: {OUTPUT_CONNECTIVITY_DIR_NAME_BASE}) ---") 
    
    if FINAL_N_ROIS_EXPECTED is None or OUTPUT_CONNECTIVITY_DIR_NAME is None:
        logger.critical("CRITICAL: FINAL_N_ROIS_EXPECTED or OUTPUT_CONNECTIVITY_DIR_NAME was not set during initialization. Aborting.")
        return

    logger.info(f"--- Final Expected ROIs for Connectivity Matrices: {FINAL_N_ROIS_EXPECTED} ---")
    logger.info(f"--- Target Homogenized Time Series Length: {TARGET_LEN_TS} ---")
    logger.info(f"--- Output Directory Name: {OUTPUT_CONNECTIVITY_DIR_NAME} ---")
    logger.info(f"--- Selected Connectivity Channels for VAE: {CONNECTIVITY_CHANNEL_NAMES} ({N_CHANNELS} channels) ---")
    logger.info(f"--- Per-subject, per-channel normalization with RobustScaler (off-diagonal) will be applied before stacking. ---")
    
    roi_reorder_status = "ACTIVE (Yeo-17 mapping implemented)" if AAL3_ROI_ORDER_MAPPING and AAL3_ROI_ORDER_MAPPING.get("new_order_indices") is not None else "INACTIVE (default AAL3 order)"
    logger.info(f"--- ROI reordering by network is currently: {roi_reorder_status}. ---")
    if not (AAL3_ROI_ORDER_MAPPING and AAL3_ROI_ORDER_MAPPING.get("new_order_indices") is not None) :
         logger.warning("Neuroscientific/Deep Learning Recommendation: ROI reordering (e.g., by Yeo-17 networks or connectome gradients) "
                        "is highly recommended for improving CNN-based model performance and interpretability. "
                        "Current implementation of _get_aal3_network_mapping_and_order is a placeholder or failed.")


    if USE_PEARSON_OMST_CHANNEL and not (OMST_PYTHON_LOADED and orthogonal_minimum_spanning_tree is not None):
        logger.warning(f"Note: OMST from dyconnmap could not be loaded. '{PEARSON_OMST_FALLBACK_NAME}' will be used instead of '{PEARSON_OMST_CHANNEL_NAME_PRIMARY}' if enabled and fallback is selected.")

    if not BASE_PATH_AAL3.exists() or not ROI_SIGNALS_DIR_PATH_AAL3.exists():
        logger.critical(f"CRITICAL: Base AAL3 path ({BASE_PATH_AAL3}) or ROI signals directory ({ROI_SIGNALS_DIR_PATH_AAL3}) not found. Aborting.")
        return

    subject_metadata_df = load_metadata(SUBJECT_METADATA_CSV_PATH, QC_REPORT_CSV_PATH)
    if subject_metadata_df is None or subject_metadata_df.empty:
        logger.critical("Metadata loading failed or no subjects passed QC to process. Aborting.")
        return

    output_main_directory = BASE_PATH_AAL3 / OUTPUT_CONNECTIVITY_DIR_NAME 
    output_individual_tensors_dir = output_main_directory / "individual_subject_tensors" 
    try:
        output_main_directory.mkdir(parents=True, exist_ok=True)
        output_individual_tensors_dir.mkdir(parents=True, exist_ok=True) 
        logger.info(f"Main output directory created/exists: {output_main_directory}")
    except OSError as e:
        logger.critical(f"Could not create output directories: {e}. Aborting."); return

    logger.info(f"Total CPU cores available: {TOTAL_CPU_CORES}. Using MAX_WORKERS = {MAX_WORKERS} for ProcessPoolExecutor.")
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    logger.warning(f"Available system RAM at start of parallel processing: {available_ram_gb:.2f} GB. Monitor usage closely.")

    subject_rows_to_process = list(subject_metadata_df.iterrows()) 
    num_subjects_to_process = len(subject_rows_to_process)
    if num_subjects_to_process == 0: 
        logger.critical("No subjects to process after metadata loading and QC filtering. Aborting.")
        return
    logger.info(f"Starting parallel processing for {num_subjects_to_process} subjects.")
    
    all_subject_results_list = [] 
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_subject_id_map = {
            executor.submit(process_single_subject_pipeline, subject_tuple): str(subject_tuple[1]['SubjectID']).strip()
            for subject_tuple in subject_rows_to_process
        }
        for future in tqdm(as_completed(future_to_subject_id_map), total=num_subjects_to_process, desc="Processing Subjects"):
            subject_id_for_log = future_to_subject_id_map[future]
            try:
                subject_result = future.result() 
                all_subject_results_list.append(subject_result)
            except Exception as exc: 
                logger.critical(f"CRITICAL WORKER EXCEPTION for S {subject_id_for_log}: {exc}", exc_info=True)
                all_subject_results_list.append({ 
                    "id": subject_id_for_log, "status_overall": "CRITICAL_WORKER_EXCEPTION",
                    "detail_preprocessing": f"Worker process crashed: {str(exc)}", 
                    "errors_connectivity_calc": {"worker_exception": str(exc)}
                })

    processing_log_df = pd.DataFrame(all_subject_results_list)
    log_file_path = output_main_directory / f"pipeline_log_{output_main_directory.name}.csv"
    try: 
        processing_log_df.to_csv(log_file_path, index=False)
        logger.info(f"Detailed processing log saved to: {log_file_path}")
    except Exception as e_log_save: 
        logger.error(f"Failed to save detailed processing log: {e_log_save}")

    successful_subject_entries_list = [
        res for res in all_subject_results_list
        if res.get("status_overall") == "SUCCESS_ALL_PROCESSED_AND_SAVED" and \
           res.get("path_saved_tensor") and Path(res["path_saved_tensor"]).exists()
    ]
    num_successful_subjects_for_tensor = len(successful_subject_entries_list)
    
    logger.info(f"--- Overall Processing Summary ---")
    logger.info(f"Total subjects attempted: {num_subjects_to_process}")
    logger.info(f"Successfully processed and individual tensors saved: {num_successful_subjects_for_tensor}")
    if num_successful_subjects_for_tensor < num_subjects_to_process:
        num_failed = num_subjects_to_process - num_successful_subjects_for_tensor
        logger.warning(f"{num_failed} subjects failed at some stage. Check the detailed log: {log_file_path}")

    if num_successful_subjects_for_tensor > 0:
        logger.info(f"Attempting to assemble global tensor for {num_successful_subjects_for_tensor} successfully processed subjects.")
        global_conn_tensor_list = []
        final_subject_ids_in_tensor = []
        
        current_expected_rois_for_assembly = FINAL_N_ROIS_EXPECTED 
        if current_expected_rois_for_assembly is None:
            logger.critical("Cannot assemble global tensor: FINAL_N_ROIS_EXPECTED is None after all processing.")
        else:
            logger.warning("Assembling global tensor using np.stack. This may be memory-intensive for large datasets.")
            try:
                for s_entry in tqdm(successful_subject_entries_list, desc="Assembling Global Tensor"):
                    s_id = s_entry["id"]
                    tensor_path_str = s_entry["path_saved_tensor"]
                    try:
                        with np.load(tensor_path_str) as loaded_npz:
                            s_tensor_data = loaded_npz['tensor_data']
                            if s_tensor_data.shape == (N_CHANNELS, current_expected_rois_for_assembly, current_expected_rois_for_assembly):
                                global_conn_tensor_list.append(s_tensor_data)
                                final_subject_ids_in_tensor.append(s_id)
                            else: 
                                logger.error(f"Tensor for S {s_id} from {tensor_path_str} has shape mismatch: {s_tensor_data.shape}. "
                                             f"Expected: ({N_CHANNELS}, {current_expected_rois_for_assembly}, {current_expected_rois_for_assembly}). Skipping.")
                        del s_tensor_data; gc.collect()
                    except Exception as e_load_ind_tensor: 
                        logger.error(f"Error loading individual tensor for S {s_id} from {tensor_path_str}: {e_load_ind_tensor}. Skipping.")
                
                if global_conn_tensor_list:
                    global_conn_tensor = np.stack(global_conn_tensor_list, axis=0).astype(np.float32)
                    del global_conn_tensor_list; gc.collect() 
                    
                    # --- NOTA IMPORTANTE SOBRE NORMALIZACIÓN INTER-CANAL GLOBAL Y DATA LEAKAGE (Sugerencia Tesis 8) ---
                    logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    logger.critical("!! DATA LEAKAGE ALERT & THESIS BEST PRACTICE: Inter-Channel Normalization of Global Tensor        !!")
                    logger.critical("!! La función '_normalize_global_tensor_inter_channel' es un placeholder.                         !!")
                    logger.critical("!! Si se implementa para escalar los canales del 'global_conn_tensor' entre sí (ej. Z-score):    !!")
                    logger.critical("!!   1. DEBE realizarse DENTRO de cada fold de validación cruzada en el script de MODELADO.      !!")
                    logger.critical("!!   2. Los parámetros de normalización (media, std por canal) se calculan ÚNICAMENTE sobre el    !!")
                    logger.critical("!!      CONJUNTO DE ENTRENAMIENTO (train_indices) de ESE FOLD.                                        !!")
                    logger.critical("!!   3. Estos parámetros (ej. guardados en 'norm_params') se APLICAN de forma fija a los          !!")
                    logger.critical("!!      conjuntos de validación y prueba de ESE FOLD.                                             !!")
                    logger.critical("!!   4. NO calcular estos parámetros sobre el tensor global completo ANTES de dividir en folds.    !!")
                    logger.critical("!!   5. Para la tesis: documentar este procedimiento y considerar mostrar resultados con/sin      !!")
                    logger.critical("!!      esta normalización global para demostrar su impacto (y la ausencia de 'peeking').        !!")
                    logger.critical("!!      Considerar guardar el hash del split de CV para trazabilidad.                             !!")
                    logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # Ejemplo de llamada DENTRO DE UN BUCLE DE CV en el script de modelado:
                    # (Asumiendo que 'global_conn_tensor_loaded' es el tensor cargado desde el .npz)
                    # (y 'train_indices_for_current_fold' son los índices de los sujetos de entrenamiento para el fold actual)
                    # normalized_train_tensor_fold, norm_params_fold = _normalize_global_tensor_inter_channel(
                    #     global_conn_tensor_loaded[train_indices_for_current_fold], 
                    #     np.arange(len(train_indices_for_current_fold)) # Los índices dentro del subset de entrenamiento
                    # )
                    # # Luego, para aplicar a validación/test de ESE FOLD:
                    # # normalized_val_tensor_fold = aplicar_transformacion(global_conn_tensor_loaded[val_indices_for_current_fold], norm_params_fold)
                    # # normalized_test_tensor_fold = aplicar_transformacion(global_conn_tensor_loaded[test_indices_for_current_fold], norm_params_fold)

                    global_tensor_fname = f"GLOBAL_TENSOR_from_{output_main_directory.name}.npz" 
                    global_tensor_path = output_main_directory / global_tensor_fname
                    
                    global_save_metadata = {
                        'global_tensor_data': global_conn_tensor, 
                        'subject_ids': np.array(final_subject_ids_in_tensor, dtype=str), 
                        'channel_names': np.array(CONNECTIVITY_CHANNEL_NAMES, dtype=str),
                        'rois_count': current_expected_rois_for_assembly,
                        'target_len_ts': TARGET_LEN_TS,
                        'tr_seconds': TR_SECONDS,
                        'filter_low_hz': LOW_CUT_HZ,
                        'filter_high_hz': HIGH_CUT_HZ,
                        'hrf_deconvolution_applied': APPLY_HRF_DECONVOLUTION,
                        'hrf_model': HRF_MODEL if APPLY_HRF_DECONVOLUTION else "N/A",
                        'channel_normalization_method_subject': "RobustScaler_per_channel_per_subject_off_diagonal", 
                        'notes_on_further_normalization': "Inter-channel global normalization (e.g., z-scoring each of the 6 channels based on training set statistics) should be performed within the cross-validation loop of the modeling phase to prevent data leakage."
                    }
                    if AAL3_ROI_ORDER_MAPPING and AAL3_ROI_ORDER_MAPPING.get("new_order_indices") is not None:
                       global_save_metadata['roi_order_name'] = AAL3_ROI_ORDER_MAPPING.get('order_name', 'custom_network_order_placeholder')
                       if 'roi_names_new_order' in AAL3_ROI_ORDER_MAPPING: 
                           global_save_metadata['roi_names_in_order'] = np.array(AAL3_ROI_ORDER_MAPPING.get('roi_names_new_order',[]), dtype=str)
                       if 'network_labels_new_order' in AAL3_ROI_ORDER_MAPPING:
                           global_save_metadata['network_labels_in_order'] = np.array(AAL3_ROI_ORDER_MAPPING.get('network_labels_new_order',[]), dtype=str)
                    else:
                       global_save_metadata['roi_order_name'] = 'aal3_original_reduced_order'


                    np.savez_compressed(global_tensor_path, **global_save_metadata)
                    logger.info(f"Global tensor successfully assembled and saved: {global_tensor_path.name}")
                    logger.info(f"Global tensor shape: {global_conn_tensor.shape} (Subjects, Channels, ROIs, ROIs)")
                    del global_conn_tensor; gc.collect() 
                else: 
                    logger.warning("No valid individual tensors were loaded for global assembly. Global tensor not created.")
            except MemoryError: 
                logger.critical("MEMORY ERROR during global tensor assembly (np.stack). Dataset might be too large.")
            except Exception as e_global: 
                logger.critical(f"An unexpected error occurred during global tensor assembly: {e_global}", exc_info=True)

    total_time_min = (time.time() - script_start_time) / 60
    logger.info(f"--- fMRI Connectivity Pipeline Finished ---")
    logger.info(f"Total execution time: {total_time_min:.2f} minutes.")
    logger.info(f"Final main process RAM: {main_process_info.memory_info().rss / (1024**2):.2f} MB")
    logger.info(f"All outputs, logs, and tensors should be in: {output_main_directory}")
    logger.info("TESIS DOCTORAL - RECORDATORIOS CLAVE (Checklist Anti-Leakage & Buenas Prácticas):")
    logger.info("  1. PREPROCESAMIENTO fMRI PREVIO (CRÍTICO): Documentar exhaustivamente todos los pasos (movimiento, confounds, scrubbing con umbrales FD/DVARS, GSR si aplica, RETROICOR/ICA-AROMA si se usaron). La calidad de estas matrices depende de ello.")
    logger.info("  2. SELECCIÓN/EXCLUSIÓN DE ROIs Y SUJETOS: Justificar umbral de volumen para ROIs (verificar ROIs clave AD). Justificar criterios de exclusión de sujetos (ej. % outliers, FD medio); asegurar que no introducen sesgo de grupo.")
    logger.info("  3. VARIABILIDAD HRF: Documentar uso de HRF canónica y considerar discusión de sus limitaciones/alternativas (HRF por sujeto, TDM).")
    logger.info("  4. PARCELACIÓN: Documentar AAL3. Considerar discutir alternativas (multi-escala, híbrida, ej. Schaefer+Yeo, o parcelaciones específicas para AD) para trabajos futuros.")
    logger.info("  5. REORDENAMIENTO DE ROIs (MUY RECOMENDADO PARA CNNs): Si se implementa, detallar el mapeo a redes funcionales (Yeo-17, gradientes de conectividad) y el nuevo orden. Guardar esta información.")
    logger.info("  6. NORMALIZACIÓN DE MATRICES:")
    logger.info("     a. Intra-Canal/Sujeto: RobustScaler (off-diagonal) (implementado) - documentar.")
    logger.info("     b. Inter-Canal Global: DEBE realizarse DENTRO de los folds de CV en el script de modelado (parámetros del set de entrenamiento aplicados a validación/test) para evitar data leakage. Documentar método (ej. Z-score) y este procedimiento. Considerar guardar hash del split de CV para trazabilidad.")
    logger.info("  7. SELECCIÓN DE CANALES DE CONECTIVIDAD: Justificar elección inicial. Planificar y documentar análisis de ablación, gating o importancia de características (SHAP, mapas de sensibilidad) para evaluar la contribución de cada canal (especialmente MI y Granger) en el modelo VAE final y decidir sobre su retención.")
    logger.info("     (Sugerencia práctica: empezar con Pearson-Full, añadir OMST, dFC-StdDev; evaluar MI y Granger con cautela).")
    logger.info("     (Considerar añadir métricas como Graphical Lasso/Correlación Parcial, Edge-Time-Series, o HMM/CAPs si el tiempo lo permite y se justifica).")
    logger.info("  8. INTERPRETACIÓN DE MEDIDAS: Ser cauto con la interpretación de Granger. Correlacionar hallazgos de dFC con clínica si es posible. Usar técnicas de decodificación (connectivity gradient decoding, BrainMap) para interpretar factores latentes del VAE.")
    logger.info("  9. VALIDACIÓN DEL MODELO VAE Y CLASIFICADOR: Describir la arquitectura del VAE (considerar GNN/VAE híbrido), función de pérdida (ej. β-VAE, aprendizaje contrastivo), y la estrategia de validación cruzada ANIDADA para el clasificador final, asegurando que no haya data leakage en ningún paso (incluida la optimización de hiperparámetros con Optuna/W&B en el bucle interno).")
    logger.info(" 10. CONFIABILIDAD Y ESTADÍSTICA: Si hay datos test-retest, calcular ICC de las matrices/canales (descartar canales con ICC < ~0.4). Para comparaciones de grupo o mapas de importancia, usar métodos estadísticos robustos (NBS, TFCE, spin-tests).")
    logger.info(" 11. HARMONIZACIÓN MULTI-SITIO (Sugerencia Tesis 7): Si los datos provienen de múltiples sitios/escáneres, aplicar harmonización (ej. ComBat, neuroHarmonize) sobre las características derivadas DENTRO de los folds de CV (parámetros del set de entrenamiento).")
    logger.info(" 12. DIMENSIONALIDAD: Reconocer que se aumenta la dimensionalidad de las features, no el N. Mitigar con regularización en el VAE (β alto, dropout de canal, gating, etc.) y early stopping basado en métricas de validación (AUROC, no solo reconstrucción). Considerar data augmentation (random masking de ROIs/canales).")
    logger.info(" 13. REPRODUCIBILIDAD (Sugerencia Tesis 10): Exportar resultados en formato BIDS-Derivatives si es posible. Publicar código (ej. Zenodo con DOI) y describir detalladamente el pipeline (versión v6.5.17).") # Actualizar versión aquí también
    logger.info(" 14. CITAS: Citar dyconnmap si se usó OMST, y todas las herramientas/paquetes relevantes.")


if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()