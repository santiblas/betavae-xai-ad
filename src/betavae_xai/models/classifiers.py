"""
models/classifiers.py

Classical ML classifiers (scikit-learn, LightGBM, XGBoost, CatBoost) and
their Optuna search spaces for AD vs CN classification from VAE latent
representations.

Used in:
"Explainable Latent Representation Learning for Alzheimer’s Disease:
 A β-VAE and Saliency Map Framework"
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import warnings
warnings.filterwarnings("ignore")

import lightgbm
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from lightgbm import LGBMClassifier
from lightgbm.basic import _LIB, _safe_call
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


logger = logging.getLogger(__name__)

# Silenciar un poco librerías ruidosas
for noisy in ["lightgbm", "optuna", "sklearn", "xgboost", "catboost"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

# LightGBM C++ log level (si la función existe)
if hasattr(_LIB, "LGBM_SetLogLevel"):
    try:
        # 0 = fatal, 1 = error, 2 = warning, 3 = info, 4 = debug
        _safe_call(_LIB.LGBM_SetLogLevel(0))
    except Exception:
        pass

# XGBoost: esconder logs molestos
os.environ.setdefault("XGB_HIDE_LOG", "1")

# Detección de GPU opcional (cupy)
try:
    import cupy as cp

    HAS_GPU: bool = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_GPU = False


ClassifierPipelineAndGrid = Tuple[ImblearnPipeline, Dict[str, Any], int]


def get_available_classifiers() -> List[str]:
    """Devuelve la lista de tipos de clasificadores soportados."""
    return ["rf", "gb", "svm", "logreg", "mlp", "xgb", "cat"]


def _parse_hidden_layers(hidden_layers_str: str | None) -> Tuple[int, ...]:
    """Convierte un string '128,64' en una tupla (128, 64)."""
    if not hidden_layers_str:
        return (128, 64)
    return tuple(
        int(x.strip())
        for x in hidden_layers_str.split(",")
        if x.strip()
    )


def get_classifier_and_grid(
    classifier_type: str,
    *,
    seed: int = 42,
    balance: bool = False,
    use_smote: bool = False,
    tune_sampler_params: bool = False,
    mlp_hidden_layers: str = "128,64",
    calibrate: bool = False,
    use_feature_selection: bool = False,
) -> ClassifierPipelineAndGrid:
    """
    Construye un pipeline de imblearn y devuelve:
      - pipeline (ImblearnPipeline)
      - dict de distribuciones de Optuna
      - n_iter_suggested (int) para la búsqueda.

    Los features de entrada típicos son:
      - vectores latentes del VAE (z, mu, etc.)
      - opcionalmente, metadatos concatenados.
    """
    ctype = classifier_type.lower()
    if ctype not in get_available_classifiers():
        raise ValueError(f"Tipo de clasificador no soportado: {classifier_type!r}")

    class_weight = "balanced" if balance else None
    model: Any
    param_distributions: Dict[str, Any]
    n_iter_search = 150  # valor por defecto

    # ------------------------------------------------------------------
    # SVM (RBF)
    # ------------------------------------------------------------------
    if ctype == "svm":
        model = SVC(
            probability=False,
            random_state=seed,
            class_weight=class_weight,
            cache_size=500,
        )
        param_distributions = {
            "model__C": FloatDistribution(1, 1e4, log=True),
            "model__gamma": FloatDistribution(1e-7, 1e-3, log=True),
            "model__kernel": CategoricalDistribution(["rbf"]),
        }
        n_iter_search = 200

    # ------------------------------------------------------------------
    # Regresión logística
    # ------------------------------------------------------------------
    elif ctype == "logreg":
        model = LogisticRegression(
            random_state=seed,
            class_weight=class_weight,
            solver="liblinear",
            max_iter=2000,
        )
        param_distributions = {
            "model__C": FloatDistribution(1e-5, 1, log=True),
        }
        n_iter_search = 200

    # ------------------------------------------------------------------
    # Gradient Boosting con LightGBM
    # ------------------------------------------------------------------
    elif ctype == "gb":
        model = LGBMClassifier(
            random_state=seed,
            objective="binary",
            class_weight=class_weight,
            n_jobs=1,   # Optuna se encarga de la paralelización externa
            verbose=-1,
        )

        # Soporte GPU de LightGBM (si el build lo permite)
        if HAS_GPU:
            try:
                # LGBM_HasGPU devuelve 0/1
                if bool(_safe_call(_LIB.LGBM_HasGPU())):
                    model.set_params(device_type="gpu", gpu_use_dp=True)
                    logger.info("[LightGBM] ➜ GPU activada")
                else:
                    model.set_params(device_type="cpu")
                    logger.info("[LightGBM] ➜ Build sin GPU, usando CPU")
            except Exception:
                model.set_params(device_type="cpu")
                logger.info("[LightGBM] ➜ No se pudo comprobar la GPU, usando CPU")
        else:
            model.set_params(device_type="cpu")
            logger.info("[LightGBM] ➜ GPU no disponible, usando CPU")

        param_distributions = {
            # Estructura del árbol
            "model__max_depth":        IntDistribution(3, 12),
            "model__num_leaves":       IntDistribution(8, 2**10),

            # Muestras y features por árbol
            "model__bagging_fraction": FloatDistribution(0.5, 1.0),
            "model__feature_fraction": FloatDistribution(0.5, 1.0),
            "model__bagging_freq":     IntDistribution(1, 10),

            # Aprendizaje
            "model__learning_rate":    FloatDistribution(5e-4, 0.01, log=True),
            "model__n_estimators":     IntDistribution(300, 1000),

            # Regularización
            "model__min_child_samples": IntDistribution(5, 50),
            "model__min_child_weight":  FloatDistribution(1e-3, 10, log=True),
            "model__min_split_gain":    FloatDistribution(0.0, 1.0),
            "model__reg_alpha":         FloatDistribution(1e-3, 1.0, log=True),
            "model__reg_lambda":        FloatDistribution(1e-3, 1.0, log=True),
        }

        n_param = len(param_distributions)
        n_iter_search = int(round((15 * n_param) / 10.0)) * 10  # múltiplo de 10

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------
    elif ctype == "rf":
        logger.info("[RandomForest] ➜ scikit-learn (CPU).")
        model = RandomForestClassifier(
            random_state=seed,
            class_weight=class_weight,
            n_jobs=-1,
        )
        param_distributions = {
            "model__n_estimators":      IntDistribution(100, 1200),
            "model__max_features":      CategoricalDistribution(["sqrt", "log2", 0.2, 0.4]),
            "model__max_depth":         IntDistribution(8, 50),
            "model__min_samples_split": IntDistribution(2, 30),
            "model__min_samples_leaf":  IntDistribution(1, 20),
        }
        n_iter_search = 150

    # ------------------------------------------------------------------
    # MLP (scikit-learn)
    # ------------------------------------------------------------------
    elif ctype == "mlp":
        hidden = _parse_hidden_layers(mlp_hidden_layers)
        model = MLPClassifier(
            random_state=seed,
            hidden_layer_sizes=hidden,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=25,
        )
        param_distributions = {
            "model__alpha":             FloatDistribution(1e-5, 1e-1, log=True),
            "model__learning_rate_init":FloatDistribution(1e-5, 1e-2, log=True),
        }
        n_iter_search = 200

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    elif ctype == "xgb":
        device = "cuda" if HAS_GPU else "cpu"
        model = XGBClassifier(
            random_state=seed,
            eval_metric="auc",
            n_jobs=1,
            tree_method="hist",
            device=device,
            verbosity=0,
        )
        if HAS_GPU:
            logger.info("[XGBoost] ➜ se usará GPU (device=cuda)")
        else:
            logger.info("[XGBoost] ➜ GPU no disponible, usando CPU.")

        param_distributions = {
            "model__gamma":            FloatDistribution(0.0, 5.0),
            "model__n_estimators":     IntDistribution(500, 1500),
            "model__learning_rate":    FloatDistribution(1e-4, 0.1, log=True),
            "model__max_depth":        IntDistribution(4, 12),
            "model__subsample":        FloatDistribution(0.3, 1.0),
            "model__colsample_bytree": FloatDistribution(0.5, 1.0),
            "model__min_child_weight": FloatDistribution(0.5, 10.0, log=True),
        }
        n_iter_search = 200

    # ------------------------------------------------------------------
    # CatBoost
    # ------------------------------------------------------------------
    elif ctype == "cat":
        model = CatBoostClassifier(
            random_state=seed,
            eval_metric="Logloss",
            verbose=0,
            loss_function="Logloss",
            thread_count=1,
        )
        if HAS_GPU:
            model.set_params(task_type="GPU", devices="0:0")
            logger.info("[CatBoost] ➜ se usará GPU")
        else:
            logger.info("[CatBoost] ➜ GPU no disponible, usando CPU.")

        param_distributions = {
            "model__depth":             IntDistribution(4, 8),
            "model__learning_rate":     FloatDistribution(1e-3, 0.08, log=True),
            "model__l2_leaf_reg":       FloatDistribution(0.1, 20.0, log=True),
            "model__iterations":        IntDistribution(400, 1500),
            "model__bagging_temperature": FloatDistribution(0.1, 0.9),
        }
        n_iter_search = 120

    # ------------------------------------------------------------------
    # Calibración opcional (isotonic)
    # ------------------------------------------------------------------
    if calibrate and ctype in ["svm", "gb", "rf"]:
        # Envolvemos el modelo en CalibratedClassifierCV y adaptamos los nombres de los params
        base_model = model
        model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)

        _cal = CalibratedClassifierCV(
            base_model.model if hasattr(base_model, "model") else base_model
        )
        _inner = "estimator" if "estimator" in _cal.get_params() else "base_estimator"

        param_distributions = {
            f"model__{_inner}__{k.split('__', 1)[1]}": v
            for k, v in param_distributions.items()
        }

    # ------------------------------------------------------------------
    # Construcción del pipeline Imblearn
    # ------------------------------------------------------------------
    # 1) Escalado: no escalamos árboles / boosting / xgb / cat
    scaler_step = (
        ("scaler", "passthrough")
        if ctype in ["rf", "gb", "xgb", "cat"]
        else ("scaler", StandardScaler())
    )

    # 2) SMOTE opcional
    oversampler_step: tuple | None = None
    if use_smote:
        oversampler_step = ("smote", SMOTE(random_state=seed))
        logger.info("[SMOTE] ➜ aplicado sólo dentro de folds (imblearn Pipeline).")
        if tune_sampler_params:
            param_distributions["smote__k_neighbors"] = IntDistribution(3, 25)

    # 3) Selección de features opcional (para latentes + metadatos)
    feature_selector_step: tuple | None = None
    if use_feature_selection:
        feature_selector_step = ("feature_selector", SelectKBest(f_classif))
        param_distributions["feature_selector__k"] = IntDistribution(20, 256)
        logger.info("[SelectKBest] ➜ añadido al pipeline (k tunable 20–256).")

    # 4) Clasificador
    model_step = ("model", model)

    # 5) Ensamblar pasos en orden lógico
    steps: List[tuple] = [scaler_step]
    if feature_selector_step is not None:
        steps.append(feature_selector_step)
    if oversampler_step is not None:
        steps.append(oversampler_step)
    steps.append(model_step)

    full_pipeline = ImblearnPipeline(steps=steps)
    return full_pipeline, param_distributions, n_iter_search


__all__ = [
    "ClassifierPipelineAndGrid",
    "get_available_classifiers",
    "get_classifier_and_grid",
]
