#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitario para Jupyter Notebook: Extracción de Características de Conectividad fMRI
Versión: v6.5.19_Standalone (Sin dependencia de Reporte QC externo)

Cambios respecto a v6.5.18:
- Eliminada dependencia de 'report_qc_final_with_discard_flags.csv'.
- La función load_metadata ahora lee directamente una lista de sujetos aprobados.
- Limpieza de imports duplicados.
"""

import itertools
import numpy as np
import pandas as pd
from dcor import distance_correlation 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression
from nilearn.glm.first_level import spm_hrf, glover_hrf 
from nilearn.datasets import fetch_atlas_yeo_2011
from nilearn import image as nli_image 
import nibabel as nib 
from scipy.signal import butter, filtfilt, deconvolve, windows
from scipy.interpolate import interp1d
from tqdm import tqdm
import os
import scipy.io as sio
from pathlib import Path
import psutil
import gc
import logging
import time
from typing import List, Tuple, Dict, Optional, Any 
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.tsa.stattools import grangercausalitytests 
import networkx as nx 
import warnings
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.utils.validation import check_random_state
import sklearn

# --- Configuración del Logger ---
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
    logger.info("Successfully imported 'threshold_omst_global_cost_efficiency' from 'dyconnmap.graphs.threshold'.")
    OMST_PYTHON_LOADED = True
    PEARSON_OMST_CHANNEL_NAME = PEARSON_OMST_CHANNEL_NAME_PRIMARY 
except ImportError:
    logger.warning("Dyconnmap module not found. Using fallback Pearson channel.")
except Exception as e_import: 
    logger.error(f"ERROR during dyconnmap import: {e_import}. Using fallback.")

# --- 0. Global Configuration and Constants ---

# --- Rutas (Se pueden sobrescribir desde el Notebook) ---
BASE_PATH_AAL3 = Path('/home/diego/proyectos/betavae-xai-ad/data')

# AQUI EL CAMBIO PRINCIPAL: Apuntamos directamente a tu CSV limpio
SUBJECT_METADATA_CSV_PATH = BASE_PATH_AAL3 / 'SubjectsData_AAL3_qc.csv' 

ROI_SIGNALS_DIR_PATH_AAL3 = BASE_PATH_AAL3 / 'ROISignals_AAL3_NiftiPreprocessedAllBatchesNorm'
ROI_FILENAME_TEMPLATE = 'ROISignals_{subject_id}.mat'
AAL3_META_PATH = BASE_PATH_AAL3 / 'ROI_MNI_V7_vol.txt' 
AAL3_NIFTI_PATH = BASE_PATH_AAL3 / "AAL3v1.nii.gz"

# Parametros de Procesamiento
TR_SECONDS = 3.0 
LOW_CUT_HZ = 0.01
HIGH_CUT_HZ = 0.08
FILTER_ORDER = 2 
TAPER_ALPHA = 0.1 

RAW_DATA_EXPECTED_COLUMNS = 170 
AAL3_MISSING_INDICES_1BASED = [35, 36, 81, 82] 
EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL = RAW_DATA_EXPECTED_COLUMNS - len(AAL3_MISSING_INDICES_1BASED)
SMALL_ROI_VOXEL_THRESHOLD = 100 

N_ROIS_EXPECTED = 131 
TARGET_LEN_TS = 140 
N_NEIGHBORS_MI = 5 
DFC_WIN_POINTS = 30 
DFC_STEP = 5      
APPLY_HRF_DECONVOLUTION = False 
HRF_MODEL = 'glover' 

USE_GRANGER_CHANNEL = True
GRANGER_MAX_LAG = 1 

deconv_str = "_deconv" if APPLY_HRF_DECONVOLUTION else ""
granger_suffix_global = f"GrangerLag{GRANGER_MAX_LAG}" if USE_GRANGER_CHANNEL else "NoEffConn"
OUTPUT_CONNECTIVITY_DIR_NAME_BASE = f"AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.19_Standalone" 

POSSIBLE_ROI_KEYS = ["signals", "ROISignals", "roi_signals", "ROIsignals_AAL3", "AAL3_signals", "roi_ts"] 

USE_PEARSON_OMST_CHANNEL = True 
USE_PEARSON_FULL_SIGNED_CHANNEL = True 
USE_MI_CHANNEL_FOR_THESIS = True 
USE_DFC_ABS_DIFF_MEAN_CHANNEL = True 
USE_DFC_STDDEV_CHANNEL = True 
USE_PRECISION_CHANNEL = False          
USE_DCOR_CHANNEL      = True          

CONNECTIVITY_CHANNEL_NAMES: List[str] = [] 
N_CHANNELS = 0 

try:
    TOTAL_CPU_CORES = multiprocessing.cpu_count()
    MAX_WORKERS = max(1, TOTAL_CPU_CORES // 2 if TOTAL_CPU_CORES > 2 else 1)
except NotImplementedError:
    TOTAL_CPU_CORES = 1
    MAX_WORKERS = 1
logger.info(f"Global MAX_WORKERS: {MAX_WORKERS} (of {TOTAL_CPU_CORES} cores)")

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
def _get_aal3_network_mapping_and_order() -> Optional[Dict[str, Any]]:
    logger.info("Attempting to map AAL3 ROIs to Yeo-17 networks and reorder.")
    if not AAL3_NIFTI_PATH.exists():
        logger.error(f"AAL3 NIfTI file NOT found at: {AAL3_NIFTI_PATH}. Cannot perform ROI reordering.")
        return None
    if VALID_AAL3_ROI_INFO_DF_166 is None:
        return None

    try:
        # 1. Cargar atlas Yeo-17
        yeo_atlas_obj = fetch_atlas_yeo_2011() 
        yeo_img = nib.load(yeo_atlas_obj.thick_17) 
        yeo_data = yeo_img.get_fdata().astype(int)

        # 2. Cargar atlas AAL3 NIfTI
        aal_img_orig = nib.load(AAL3_NIFTI_PATH)

        if not np.allclose(aal_img_orig.affine, yeo_img.affine, atol=1e-3) or aal_img_orig.shape != yeo_img.shape:
            logger.warning("Resampling AAL3 to Yeo space...")
            try:
                aal_img_resampled = nli_image.resample_to_img(aal_img_orig, yeo_img, interpolation='nearest')
                aal_data = aal_img_resampled.get_fdata().astype(int)
            except Exception as e_resample:
                logger.error(f"Failed to resample: {e_resample}. ROI reordering skipped.")
                return None
        else:
            aal_data = aal_img_orig.get_fdata().astype(int)

        # 3. Mapeo
        final_aal3_rois_info_df = VALID_AAL3_ROI_INFO_DF_166.drop(INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166).reset_index(drop=True)
        
        if len(final_aal3_rois_info_df) != FINAL_N_ROIS_EXPECTED:
            return None

        original_131_aal3_colors = final_aal3_rois_info_df['color'].tolist()
        original_131_aal3_names = final_aal3_rois_info_df['nom_c'].tolist()
        
        roi_network_mapping = [] 
        for aal3_idx, aal3_color in enumerate(original_131_aal3_colors):
            aal3_name = original_131_aal3_names[aal3_idx]
            aal3_roi_mask = (aal_data == aal3_color)
            
            winner_yeo_label = 0
            yeo17_name = YEO17_LABELS_TO_NAMES[0]
            overlap_percentage = 0.0

            if np.any(aal3_roi_mask):
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
                        overlap_percentage = (counts[winner_yeo_label_idx] / total_roi_voxels) * 100 
                        yeo17_name = YEO17_LABELS_TO_NAMES.get(winner_yeo_label, f"UnknownYeo{winner_yeo_label}")
            
            roi_network_mapping.append((aal3_color, aal3_name, winner_yeo_label, yeo17_name, overlap_percentage, aal3_idx ))
        
        roi_network_mapping_sorted = sorted(roi_network_mapping, key=lambda x: (x[2] == 0, x[2], x[0]))
        new_order_indices = [item[5] for item in roi_network_mapping_sorted] 
        roi_names_new_order = [item[1] for item in roi_network_mapping_sorted]
        network_labels_new_order = [item[3] for item in roi_network_mapping_sorted]
        
        logger.info("Successfully mapped AAL3 ROIs to Yeo-17 networks.")
        
        # Guardar CSV de mapeo
        mapping_df = pd.DataFrame(roi_network_mapping_sorted, columns=['AAL3_Color', 'AAL3_Name', 'Yeo17_Label', 'Yeo17_Network', 'Overlap_Percent', 'Original_Index_0_N'])
        mapping_filename = BASE_PATH_AAL3 / f"aal3_{FINAL_N_ROIS_EXPECTED}_to_yeo17_mapping.csv"
        try:
            mapping_df.to_csv(mapping_filename, index=False)
        except Exception:
            pass

        return {
            'order_name': 'aal3_to_yeo17_overlap_sorted',
            'roi_indices_original_order': list(range(FINAL_N_ROIS_EXPECTED)), 
            'roi_names_original_order': original_131_aal3_names, 
            'roi_names_new_order': roi_names_new_order,
            'network_labels_new_order': network_labels_new_order,
            'new_order_indices': new_order_indices 
        }
    except Exception as e:
        logger.error(f"Error during ROI reordering: {e}")
        return None

def _initialize_aal3_roi_processing_info():
    global VALID_AAL3_ROI_INFO_DF_166, AAL3_MISSING_INDICES_0BASED, \
           INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166, FINAL_N_ROIS_EXPECTED, \
           N_ROIS_EXPECTED, OUTPUT_CONNECTIVITY_DIR_NAME, CONNECTIVITY_CHANNEL_NAMES, N_CHANNELS, \
           PEARSON_OMST_CHANNEL_NAME, granger_suffix_global, AAL3_ROI_ORDER_MAPPING

    omst_suffix_for_dir = "OMST_GCE_Signed" if OMST_PYTHON_LOADED and orthogonal_minimum_spanning_tree is not None and USE_PEARSON_OMST_CHANNEL else "PearsonFullSigned"
    current_pearson_channel_to_use_as_base = PEARSON_OMST_CHANNEL_NAME_PRIMARY if OMST_PYTHON_LOADED and USE_PEARSON_OMST_CHANNEL else PEARSON_OMST_FALLBACK_NAME
    
    channel_norm_suffix = "_ChNorm" 
    roi_reorder_suffix = "_ROIreorderedYeo17" 
    
    if not AAL3_META_PATH.exists():
        logger.error(f"AAL3 metadata file NOT found: {AAL3_META_PATH}.")
        FINAL_N_ROIS_EXPECTED = N_ROIS_EXPECTED 
        AAL3_ROI_ORDER_MAPPING = None 
    else:
        try:
            meta_aal3_df = pd.read_csv(AAL3_META_PATH, sep='\t')
            meta_aal3_df['color'] = pd.to_numeric(meta_aal3_df['color'], errors='coerce')
            meta_aal3_df.dropna(subset=['color'], inplace=True)
            meta_aal3_df['color'] = meta_aal3_df['color'].astype(int)
            
            AAL3_MISSING_INDICES_0BASED = [idx - 1 for idx in AAL3_MISSING_INDICES_1BASED]
            VALID_AAL3_ROI_INFO_DF_166 = meta_aal3_df[~meta_aal3_df['color'].isin(AAL3_MISSING_INDICES_1BASED)].copy()
            VALID_AAL3_ROI_INFO_DF_166.sort_values(by='color', inplace=True) 
            VALID_AAL3_ROI_INFO_DF_166.reset_index(drop=True, inplace=True)
            
            small_rois_mask_on_166 = VALID_AAL3_ROI_INFO_DF_166['vol_vox'] < SMALL_ROI_VOXEL_THRESHOLD
            INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166 = VALID_AAL3_ROI_INFO_DF_166[small_rois_mask_on_166].index.tolist()
            
            FINAL_N_ROIS_EXPECTED = EXPECTED_ROIS_AFTER_AAL3_MISSING_REMOVAL - len(INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166)
            N_ROIS_EXPECTED = FINAL_N_ROIS_EXPECTED 
            
            AAL3_ROI_ORDER_MAPPING = _get_aal3_network_mapping_and_order() 
        except Exception as e:
            logger.error(f"Error initializing ROI info: {e}")
            FINAL_N_ROIS_EXPECTED = N_ROIS_EXPECTED
            AAL3_ROI_ORDER_MAPPING = None 
    
    current_roi_order_suffix = roi_reorder_suffix if AAL3_ROI_ORDER_MAPPING and AAL3_ROI_ORDER_MAPPING.get("new_order_indices") is not None else ""
    
    # Nombre del directorio simplificado
    if FINAL_N_ROIS_EXPECTED is None or not AAL3_META_PATH.exists(): 
        OUTPUT_CONNECTIVITY_DIR_NAME = f"{OUTPUT_CONNECTIVITY_DIR_NAME_BASE}_ERR_INIT"
    else:
        OUTPUT_CONNECTIVITY_DIR_NAME = f"{OUTPUT_CONNECTIVITY_DIR_NAME_BASE}{current_roi_order_suffix}"
            
    temp_channels = []
    if USE_PEARSON_OMST_CHANNEL:
        temp_channels.append(current_pearson_channel_to_use_as_base)
    
    if USE_PEARSON_FULL_SIGNED_CHANNEL and current_pearson_channel_to_use_as_base != PEARSON_OMST_FALLBACK_NAME : 
        if PEARSON_OMST_FALLBACK_NAME not in temp_channels:
             temp_channels.append(PEARSON_OMST_FALLBACK_NAME) 
    if USE_MI_CHANNEL_FOR_THESIS: temp_channels.append("MI_KNN_Symmetric")
    if USE_DFC_ABS_DIFF_MEAN_CHANNEL: temp_channels.append("dFC_AbsDiffMean")
    if USE_DFC_STDDEV_CHANNEL: temp_channels.append("dFC_StdDev") 
    if USE_PRECISION_CHANNEL: temp_channels.append("Precision_FisherZ")
    if USE_DCOR_CHANNEL: temp_channels.append("DistanceCorr")
    
    if USE_GRANGER_CHANNEL: 
        granger_channel_name = f"Granger_F_lag{GRANGER_MAX_LAG}" 
        temp_channels.append(granger_channel_name)
    
    CONNECTIVITY_CHANNEL_NAMES = list(dict.fromkeys(temp_channels)) 
    N_CHANNELS = len(CONNECTIVITY_CHANNEL_NAMES)
    return True

# Ejecutar inicialización global
if not _initialize_aal3_roi_processing_info():
    logger.critical("CRITICAL: ROI processing info could not be initialized.")
    exit() 

# --- 1. Subject Metadata Loading (SIMPLIFICADO) ---
def load_metadata(subject_meta_csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Carga SOLAMENTE el CSV de sujetos proporcionado, asumiendo que ya está filtrado (QC passed).
    """
    logger.info(f"--- Loading Metadata from: {subject_meta_csv_path} ---")
    try:
        if not subject_meta_csv_path.exists():
            logger.critical(f"CSV file NOT found: {subject_meta_csv_path}")
            return None

        subjects_df = pd.read_csv(subject_meta_csv_path)
        
        # Limpieza y Validación
        if 'SubjectID' not in subjects_df.columns:
            logger.critical("Column 'SubjectID' missing in CSV.")
            return None
            
        subjects_df['SubjectID'] = subjects_df['SubjectID'].astype(str).str.strip() 
        
        # Manejo de TimePoints si no existe (opcional)
        if 'TimePoints' not in subjects_df.columns:
            logger.warning("'TimePoints' column missing in CSV. Assuming default behavior.")
            subjects_df['TimePoints'] = TARGET_LEN_TS 
        
        # Manejo de ResearchGroup si no existe
        if 'ResearchGroup' not in subjects_df.columns:
            subjects_df['ResearchGroup'] = 'Unknown'

        logger.info(f"Loaded {len(subjects_df)} subjects to process.")
        return subjects_df[['SubjectID', 'TimePoints', 'ResearchGroup']]

    except Exception as e:
        logger.critical(f"Error loading metadata: {e}", exc_info=True)
        return None

# --- Funciones de Procesamiento y Cálculo (Sin cambios mayores) ---
# [Mantienen la misma lógica matemática de la versión v6.5.18]

def _reorder_rois_by_network_for_timeseries(timeseries_data, new_order_indices, subject_id):
    if new_order_indices is None or len(new_order_indices) != timeseries_data.shape[1]:
        return timeseries_data
    return timeseries_data[:, new_order_indices]

def _reorder_connectivity_matrix_by_network(matrix, new_order_indices, subject_id, channel_name):
    if new_order_indices is None or len(new_order_indices) != matrix.shape[0]:
        return matrix
    return matrix[np.ix_(new_order_indices, new_order_indices)]

def _load_signals_from_mat(mat_path: Path, possible_keys: List[str]) -> Optional[np.ndarray]:
    try:
        data = sio.loadmat(mat_path)
        for key in possible_keys:
            if key in data and isinstance(data[key], np.ndarray) and data[key].ndim >= 2:
                return data[key].astype(np.float64) 
    except Exception:
        pass
    return None

def _orient_and_reduce_rois(raw_sigs, subject_id, initial_expected_cols, aal3_missing_0based, small_rois_indices_from_166, final_expected_rois):
    if raw_sigs.ndim != 2: return None
    oriented_sigs = raw_sigs.copy()
    
    if oriented_sigs.shape[0] == initial_expected_cols and oriented_sigs.shape[1] != initial_expected_cols:
        oriented_sigs = oriented_sigs.T
    
    if oriented_sigs.shape[1] != initial_expected_cols: return None

    if aal3_missing_0based is not None:
        oriented_sigs = np.delete(oriented_sigs, aal3_missing_0based, axis=1)
    
    if small_rois_indices_from_166 is not None:
        oriented_sigs = np.delete(oriented_sigs, small_rois_indices_from_166, axis=1)
        
    return oriented_sigs

def _bandpass_filter_signals(sigs, lowcut, highcut, fs, order, subject_id, taper_alpha=0.1):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = np.zeros_like(sigs)
    padlen = 3 * max(len(a), len(b))
    
    for i in range(sigs.shape[1]):
        sig = sigs[:, i]
        if len(sig) > padlen:
            win = windows.tukey(len(sig), alpha=taper_alpha)
            filtered[:, i] = filtfilt(b, a, sig * win)
        else:
            filtered[:, i] = sig
    return filtered

def _hrf_deconvolution(sigs, tr, hrf_model, subject_id):
    if hrf_model == 'glover': kernel = glover_hrf(tr, oversampling=1)
    elif hrf_model == 'spm': kernel = spm_hrf(tr, oversampling=1)
    else: return sigs
    
    deconv = np.zeros_like(sigs)
    for i in range(sigs.shape[1]):
        try:
            quot, _ = deconvolve(sigs[:, i], kernel)
            deconv[:, i] = quot[:sigs.shape[0]] if len(quot) >= sigs.shape[0] else np.pad(quot, (0, sigs.shape[0]-len(quot)))
        except:
            deconv[:, i] = sigs[:, i]
    return deconv

def _preprocess_time_series(sigs, target_len, subject_id, eff_conn_lag, tr, low, high, order, apply_deconv, hrf_model, taper):
    sigs = _bandpass_filter_signals(sigs, low, high, 1.0/tr, order, subject_id, taper)
    if apply_deconv:
        sigs = _hrf_deconvolution(sigs, tr, hrf_model, subject_id)
    
    scaler = StandardScaler()
    sigs = scaler.fit_transform(sigs)
    sigs = np.nan_to_num(sigs)
    
    current_len = sigs.shape[0]
    if current_len != target_len:
        new_sigs = np.zeros((target_len, sigs.shape[1]))
        x_old = np.linspace(0, 1, current_len)
        x_new = np.linspace(0, 1, target_len)
        for i in range(sigs.shape[1]):
            new_sigs[:, i] = interp1d(x_old, sigs[:, i], kind='linear', fill_value="extrapolate")(x_new)
        return new_sigs.astype(np.float32)
    return sigs.astype(np.float32)

def load_and_preprocess_single_subject_series(subject_id, target_len, dir_path, template, keys, lag, tr, low, high, order, deconv, hrf, taper, roi_info):
    mat_path = dir_path / template.format(subject_id=subject_id)
    if not mat_path.exists(): return None, "File not found", False
    
    raw = _load_signals_from_mat(mat_path, keys)
    if raw is None: return None, "Load error", False
    
    reduced = _orient_and_reduce_rois(raw, subject_id, RAW_DATA_EXPECTED_COLUMNS, AAL3_MISSING_INDICES_0BASED, INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166, FINAL_N_ROIS_EXPECTED)
    if reduced is None: return None, "Reduction error", False

    if roi_info and roi_info.get("new_order_indices"):
        reduced = _reorder_rois_by_network_for_timeseries(reduced, roi_info["new_order_indices"], subject_id)

    processed = _preprocess_time_series(reduced, target_len, subject_id, lag, tr, low, high, order, deconv, hrf, taper)
    return processed, "OK", True

# --- Connectivity Calculators ---
def fisher_r_to_z(r):
    return np.arctanh(np.clip(np.nan_to_num(r), -0.99999, 0.99999))

def calculate_pearson_full_fisher_z_signed(ts, sid):
    if ts.shape[0] < 2: return None
    c = np.corrcoef(ts, rowvar=False)
    z = fisher_r_to_z(c)
    np.fill_diagonal(z, 0)
    return z.astype(np.float32)

def calculate_pearson_omst_signed_weighted(ts, sid):
    if not OMST_PYTHON_LOADED: return None
    c = np.corrcoef(ts, rowvar=False)
    z = fisher_r_to_z(c)
    w = np.abs(z)
    np.fill_diagonal(w, 0)
    try:
        _, adj = orthogonal_minimum_spanning_tree(w, n_msts=None)
        res = z * (adj > 0).astype(int)
        np.fill_diagonal(res, 0)
        return res.astype(np.float32)
    except: return None

def calculate_precision_partial_corr_fisher_z(ts, sid, cv_folds=3):
    ts = RobustScaler().fit_transform(ts)
    try:
        gl = GraphicalLassoCV(cv=cv_folds, n_jobs=-1).fit(ts)
        prec = gl.precision_
        d = np.sqrt(np.diag(prec))
        pc = -prec / np.outer(d, d)
        np.fill_diagonal(pc, 1.0)
        return fisher_r_to_z(pc).astype(np.float32)
    except: return None

def calculate_distance_correlation(ts, sid):
    n = ts.shape[1]
    res = np.zeros((n, n), dtype=np.float32)
    for i, j in itertools.combinations(range(n), 2):
        val = distance_correlation(ts[:, i], ts[:, j])
        res[i, j] = res[j, i] = val
    return res

def _mi_pair(x, y, k):
    try: return mutual_info_regression(x, y, n_neighbors=k, random_state=42, discrete_features=False)[0]
    except: return 0.0

def calculate_mi_knn_connectivity(ts, k, sid):
    n = ts.shape[1]
    res = np.zeros((n, n), dtype=np.float32)
    tasks = []
    for i in range(n):
        for j in range(i+1, n):
            tasks.append((i, j, ts[:, i].reshape(-1,1), ts[:, j]))
    
    n_jobs = max(1, TOTAL_CPU_CORES // MAX_WORKERS if MAX_WORKERS > 0 else 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals = Parallel(n_jobs=n_jobs)(delayed(_mi_pair)(t[2], t[3], k) for t in tasks)
        vals_ji = Parallel(n_jobs=n_jobs)(delayed(_mi_pair)(t[3].reshape(-1,1), t[2].flatten(), k) for t in tasks)
    
    for idx, (i, j, _, _) in enumerate(tasks):
        res[i, j] = res[j, i] = (vals[idx] + vals_ji[idx]) / 2.0
    return res

def calculate_custom_dfc_abs_diff_mean(ts, win, step, sid):
    n_tp, n_roi = ts.shape
    n_win = (n_tp - win) // step + 1
    if n_win < 2: return np.zeros((n_roi, n_roi), dtype=np.float32)
    
    sum_diff = np.zeros((n_roi, n_roi))
    prev = None
    count = 0
    for i in range(n_win):
        w_ts = ts[i*step : i*step+win]
        curr = np.abs(np.nan_to_num(np.corrcoef(w_ts, rowvar=False)))
        np.fill_diagonal(curr, 0)
        if prev is not None:
            sum_diff += np.abs(curr - prev)
            count += 1
        prev = curr
    return (sum_diff / count).astype(np.float32) if count > 0 else np.zeros((n_roi, n_roi), dtype=np.float32)

def calculate_dfc_std_dev(ts, win, step, sid):
    n_tp, n_roi = ts.shape
    n_win = (n_tp - win) // step + 1
    if n_win < 2: return np.zeros((n_roi, n_roi), dtype=np.float32)
    
    mats = []
    for i in range(n_win):
        w_ts = ts[i*step : i*step+win]
        c = np.nan_to_num(np.corrcoef(w_ts, rowvar=False))
        np.fill_diagonal(c, 0)
        mats.append(c)
    
    res = np.std(np.stack(mats), axis=0)
    np.fill_diagonal(res, 0)
    return res.astype(np.float32)

def _granger_p(t1, t2, l):
    try:
        r = grangercausalitytests(np.column_stack([t2, t1]), [l], verbose=False)
        return r[l][0]['ssr_ftest'][0]
    except: return 0.0

def calculate_granger_f_matrix(ts, lag, sid):
    n = ts.shape[1]
    res = np.zeros((n, n), dtype=np.float32)
    tasks = []
    for i in range(n):
        for j in range(i+1, n):
            tasks.append((i, j, ts[:, i], ts[:, j]))
    
    n_jobs = max(1, TOTAL_CPU_CORES // MAX_WORKERS if MAX_WORKERS > 0 else 1)
    vals = Parallel(n_jobs=n_jobs)(delayed(_granger_p)(t[2], t[3], lag) for t in tasks)
    vals_ji = Parallel(n_jobs=n_jobs)(delayed(_granger_p)(t[3], t[2], lag) for t in tasks)
    
    for idx, (i, j, _, _) in enumerate(tasks):
        res[i, j] = res[j, i] = (vals[idx] + vals_ji[idx]) / 2.0
    return res

def calculate_all_connectivity_modalities_for_subject(subject_id, ts, k, win, step, lag, roi_info):
    mats = {}
    errors = {}
    timings = {}
    
    for name in CONNECTIVITY_CHANNEL_NAMES:
        t0 = time.time()
        res = None
        try:
            if name == PEARSON_OMST_CHANNEL_NAME_PRIMARY:
                if OMST_PYTHON_LOADED: res = calculate_pearson_omst_signed_weighted(ts, subject_id)
            elif name == PEARSON_OMST_FALLBACK_NAME:
                res = calculate_pearson_full_fisher_z_signed(ts, subject_id)
            elif name == "MI_KNN_Symmetric":
                res = calculate_mi_knn_connectivity(ts, k, subject_id)
            elif name == "dFC_AbsDiffMean":
                res = calculate_custom_dfc_abs_diff_mean(ts, win, step, subject_id)
            elif name == "dFC_StdDev":
                res = calculate_dfc_std_dev(ts, win, step, subject_id)
            elif name == "Precision_FisherZ":
                res = calculate_precision_partial_corr_fisher_z(ts, subject_id)
            elif name == "DistanceCorr":
                res = calculate_distance_correlation(ts, subject_id)
            elif name.startswith("Granger"):
                res = calculate_granger_f_matrix(ts, lag, subject_id)
            
            if res is not None and roi_info:
                res = _reorder_connectivity_matrix_by_network(res, roi_info.get("new_order_indices"), subject_id, name)
            
            mats[name] = res
        except Exception as e:
            errors[name] = str(e)
        timings[f"{name}_sec"] = time.time() - t0
        
    return {"matrices": mats, "errors": errors, "timings": timings}

# --- Pipeline ---
def process_single_subject_pipeline(row_tuple):
    _, row = row_tuple
    sid = str(row['SubjectID']).strip()
    res = {"id": sid, "status": "PENDING"}
    
    try:
        # Preprocess
        ts, msg, ok = load_and_preprocess_single_subject_series(
            sid, TARGET_LEN_TS, ROI_SIGNALS_DIR_PATH_AAL3, ROI_FILENAME_TEMPLATE, POSSIBLE_ROI_KEYS,
            GRANGER_MAX_LAG, TR_SECONDS, LOW_CUT_HZ, HIGH_CUT_HZ, FILTER_ORDER,
            APPLY_HRF_DECONVOLUTION, HRF_MODEL, TAPER_ALPHA, AAL3_ROI_ORDER_MAPPING
        )
        if not ok:
            res["status"] = "PREPROC_FAIL"
            res["error"] = msg
            return res
        
        # Connectivity
        conn = calculate_all_connectivity_modalities_for_subject(
            sid, ts, N_NEIGHBORS_MI, DFC_WIN_POINTS, DFC_STEP, GRANGER_MAX_LAG, AAL3_ROI_ORDER_MAPPING
        )
        
        # Stack & Save
        final_mats = []
        for name in CONNECTIVITY_CHANNEL_NAMES:
            m = conn["matrices"].get(name)
            if m is None: continue
            
            # RobustScaler per channel (off-diagonal)
            flat = m[~np.eye(m.shape[0], dtype=bool)]
            if flat.size > 0:
                m_scaled = m.copy()
                m_scaled[~np.eye(m.shape[0], dtype=bool)] = RobustScaler().fit_transform(flat.reshape(-1,1)).flatten()
                final_mats.append(m_scaled.astype(np.float32))
            else:
                final_mats.append(m.astype(np.float32))
                
        if len(final_mats) == N_CHANNELS:
            tensor = np.stack(final_mats)
            out_dir = BASE_PATH_AAL3 / OUTPUT_CONNECTIVITY_DIR_NAME / "individual_subject_tensors"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            meta = {
                'tensor_data': tensor, 'subject_id': sid, 'channel_names': CONNECTIVITY_CHANNEL_NAMES,
                'roi_order_name': AAL3_ROI_ORDER_MAPPING.get('order_name') if AAL3_ROI_ORDER_MAPPING else 'default'
            }
            if AAL3_ROI_ORDER_MAPPING:
                meta['roi_names_in_order'] = AAL3_ROI_ORDER_MAPPING.get('roi_names_new_order')
                meta['network_labels_in_order'] = AAL3_ROI_ORDER_MAPPING.get('network_labels_new_order')

            np.savez_compressed(out_dir / f"tensor_{sid}.npz", **meta)
            res["status"] = "SUCCESS"
            res["path"] = str(out_dir / f"tensor_{sid}.npz")
            logger.info(f"S {sid}: Success.")
        else:
            res["status"] = "CONN_FAIL"
            res["errors"] = conn["errors"]
            
    except Exception as e:
        res["status"] = "CRASH"
        res["error"] = str(e)
        logger.error(f"S {sid} crashed: {e}")
        
    return res

def main():
    logger.info(f"--- Pipeline v6.5.19 ---")
    
    if not BASE_PATH_AAL3.exists() or not ROI_SIGNALS_DIR_PATH_AAL3.exists():
        logger.critical("Data paths not found.")
        return

    df = load_metadata(SUBJECT_METADATA_CSV_PATH)
    if df is None: return

    output_dir = BASE_PATH_AAL3 / OUTPUT_CONNECTIVITY_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {len(df)} subjects...")
    
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_single_subject_pipeline, row): row[1]['SubjectID'] for row in df.iterrows()}
        for f in tqdm(as_completed(futs), total=len(futs)):
            results.append(f.result())

    pd.DataFrame(results).to_csv(output_dir / "processing_log.csv", index=False)
    
    # Assemble Global Tensor
    successes = [r for r in results if r["status"] == "SUCCESS"]
    if successes:
        tensors = []
        ids = []
        for s in successes:
            try:
                with np.load(s["path"]) as d:
                    tensors.append(d['tensor_data'])
                    ids.append(s["id"])
            except: pass
        
        if tensors:
            global_t = np.stack(tensors).astype(np.float32)
            logger.info(f"Global Tensor Shape: {global_t.shape}")
            
            meta = {'global_tensor_data': global_t, 'subject_ids': ids, 'channel_names': CONNECTIVITY_CHANNEL_NAMES}
            if AAL3_ROI_ORDER_MAPPING:
                 meta['roi_names_in_order'] = AAL3_ROI_ORDER_MAPPING.get('roi_names_new_order')
            
            np.savez_compressed(output_dir / "GLOBAL_TENSOR.npz", **meta)
            logger.info("Global tensor saved.")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()