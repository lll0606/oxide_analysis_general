"""
Configuration Module - Contains constants and configuration settings for oxide descriptor analysis
This module defines all constants, directory paths, color schemes,
marker styles, and default configuration parameters used throughout the analysis.
"""

import logging
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
from datetime import datetime
import os
import numpy as np
matplotlib.use('Agg')  # Use non-interactive backend

# Configure logging
log_filename = f"oxide_analysis_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Constants
RESULTS_DIR = Path('results')
MODELS_DIR = RESULTS_DIR / 'models'
FIGURES_DIR = RESULTS_DIR / 'figures'
ANALYSIS_DIR = RESULTS_DIR / 'analysis'

# Ensure directories exist
for directory in [RESULTS_DIR, MODELS_DIR, FIGURES_DIR, ANALYSIS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Color schemes for different cluster structures
COLORS = {
    'Pt07': {'3d': '#4497c4', 'quasi': '#68be93', 'flat': '#FFD700'},
    'Pt13': {'3d': '#4497c4', 'bj': '#68be93', '2l': '#ffce3f', 'rod': '#f4ae6f', 'flat': '#f28e8e'},
    'combined':{'3d': '#4497c4', 'quasi': '#68be93', 'bj': '#68be93', '2l': '#ffce3f', 'rod': '#f4ae6f', 'flat': '#f28e8e'}
}

ACTIVE_CLUSTERS = ['combined']

# Marker styles for different structures
MARKERS = {
    '3d': 'o',        # Circle
    'quasi': '^',     # Triangle
    'flat': 's',      # Square
    'bj': 'D',        # Diamond
    '2l': 'p',        # Pentagon
    'rod': '*'        # Star
}

FEATURE_NAME_MAP = {
    'ΔEads': 'ΔEads',
    'Δq': 'Δq',
    'RMSD': 'RMSD'
    ,
    'polarons': 'Polaron count',
    'surface_polarons': 'Surface polaron count',
    'ε': 'ε',
    'Ce3_dist_sum': 'Σ(Ce³⁺-Ce³⁺ dist)',
    'Ce3_dist_min': 'Min(Ce³⁺-Ce³⁺ dist)',
    'Ce3_dist_mean': 'Mean(Ce³⁺-Ce³⁺ dist)',
    'Ce3_dist_max': 'Max(Ce³⁺-Ce³⁺ dist)',
    'Ce3_dist_std': 'Std(Ce³⁺-Ce³⁺ dist)',
    'E_pol_pol': 'Epol-pol',
    'E_pol_lattice': 'Epol-lattice',
    'Pt_O_bonds': 'Pt-O bond count',
    'Pt_O_dist_sum': 'Σ(Pt-O dist)',
    'Pt_O_dist_min': 'Min(Pt-O dist)',
    'Pt_O_dist_mean': 'Mean(Pt-O dist)',
    'Pt_O_dist_max': 'Max(Pt-O dist)',
    'Pt_O_dist_std': 'Std(Pt-O dist)',
    'Ov_concentration(%)': 'O vacancy (%)'
}

FEATURE_GROUPS = {
    "Pt_cluster": ["Δq", "RMSD"],
    "interface": [
        "Pt_O_bonds", "Pt_O_dist_sum", "Pt_O_dist_min", "Pt_O_dist_mean", 
        "Pt_O_dist_max", "Pt_O_dist_std"
    ],
    "CeOx": [
        "polarons", "surface_polarons", "ε", "Ce3_dist_sum", "Ce3_dist_min", "Ce3_dist_mean", 
        "Ce3_dist_max", "Ce3_dist_std", "E_pol_pol", "E_pol_lattice",
        "Ov_concentration(%)"
    ]
}

# Default configuration
DEFAULT_CONFIG = {
    "feature_cols": [
        'RMSD', 'Δq'
        , 
        'polarons', 'surface_polarons', 'ε', 
        'Ce3_dist_sum', 'Ce3_dist_min', 'Ce3_dist_mean', 
        'Ce3_dist_max', 'Ce3_dist_std', 'E_pol_pol', 'E_pol_lattice',
        'Pt_O_bonds', 'Pt_O_dist_sum', 'Pt_O_dist_min', 'Pt_O_dist_mean', 
        'Pt_O_dist_max', 'Pt_O_dist_std', 'Ov_concentration(%)'
    ],
    "target_col": "ΔEads",
    "structure_col": "structure",
    "random_state": 42,
    "test_size": 0.15,
    "cv_folds": 5,
    "n_jobs": -1,
    "dpi": 300,
    "target_r2": 0.95,
    "max_iterations": 30,
    "fig_formats": ["png","svg"],
    "train_on_raw_features":True,
    "train_on_selected_features":True,  
    "train_on_stacking_models":True,
    "smart_physical_feature": {
    "use_smart": True,
    "top_n": 8,
    "n_clusters": 3,
    "max_total_features": 10
    },
    # advanced options
    "advanced_models": {
        "use_xgboost": True,
        "use_lightgbm": False,
        "visualize_hyperparameters": True,
        "models_to_visualize":['rf', 'gb', 'xgb'],
        "optimize_hyperparameters": True
    },
    "FEATURE_NAME_MAP": FEATURE_NAME_MAP,
}

# set param_grids
# config.py
from skopt.space import Integer, Real

# PARAM_GRIDS = {
#     'rf': {
#         'n_estimators': list(range(10, 310, 50)),  
#         'max_depth': list(range(1, 3)),
#         'max_features': [round(x, 2) for x in np.linspace(0.05, 1.0, 2)]
#     },
#     'gb': {
#         'n_estimators': list(range(50, 301, 150)),
#         'max_depth': list(range(2, 11,5)),
#         'learning_rate': [0.01, 0.05, 0.1, 0.2]
#     },
#     'xgb': {
#         'n_estimators': list(range(50, 301, 150)),
#         'max_depth': list(range(2, 11,5)),
#         'learning_rate': [0.01, 0.05, 0.1, 0.2],
#         'subsample': [0.5, 0.75, 1.0],
#         'colsample_bytree': [0.5, 0.75, 1.0]
#     },

# }
PARAM_GRIDS = {
    'rf': {
        'n_estimators': list(range(10, 310, 10)),  
        'max_depth': list(range(1, 16)),
        'max_features': [round(x, 2) for x in np.linspace(0.05, 1.0, 20)]
    },
    'gb': {
        'n_estimators': list(range(50, 301, 50)),
        'max_depth': list(range(2, 11)),
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    },
    'xgb': {
        'n_estimators': list(range(50, 301, 50)),
        'max_depth': list(range(2, 11)),
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.5, 0.75, 1.0],
        'colsample_bytree': [0.5, 0.75, 1.0]
    },

}


BAYES_GRIDS = {
        'xgb': {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(2,10),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0)
    }
}
