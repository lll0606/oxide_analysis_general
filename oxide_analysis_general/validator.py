#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
External validataion module for oxide_analysis_general package
"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt
import json

# package imports
from oxide_analysis_general.data_processor import DataProcessor
from oxide_analysis_general.visualizer import Visualizer
from oxide_analysis_general.config import DEFAULT_CONFIG, ANALYSIS_DIR

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'Arial'

## logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Validator:
    """
    External validation class for new cluster datasets
    """

    def __init__(self, config:Dict = None):
        self.config = config if config else DEFAULT_CONFIG.copy()
        self.data_processor = DataProcessor(self.config)
        self.visualizer = Visualizer(self.config)

        self.validation_dir = ANALYSIS_DIR / 'external_validation'
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Validator initialized.")
        logger.info(f"External validation directory set to {self.validation_dir}")


    # 1. load models
    def load_trained_models(self, model_dir:Path) -> Dict[str, Any]:
        """
        only load combined models for external validation
        """
        if model_dir is None:
            model_dir = Path('results/model')
        
        model = {}
        model_files = {
            'combined_gb':'combined_gb_model.pkl',
            'combined_rf':'combined_rf_model.pkl',
            'combined_xgb':'combined_xgb_model.pkl'
        }

        for name, file_name in model_files.items():
            model_path = model_dir / file_name
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model_obj = pickle.load(f)
                    if isinstance(model_obj, dict) and 'model' in model_obj:
                        model[name] = model_obj['model']
                    
                    else:
                        model[name] = model_obj
                logger.info(f"Loaded model: {name} from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")

        # load scaler and features
        combined_scaler_path = model_dir / 'combined_scaler.pkl'
        features_path = model_dir / 'combined_selected_features.pkl'

        if combined_scaler_path.exists():
            with open(combined_scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Loaded combined scaler from {combined_scaler_path}")
        else:
            self.scaler = None
            logger.warning(f"Combined scaler file not found: {combined_scaler_path}")
        
        if features_path.exists():
            with open(features_path, 'rb') as f:
                self.selected_features = pickle.load(f)
            logger.info(f"Loaded combined selected features from {features_path}")
        else:
            self.selected_features = self.config.get('feature_cols', [])
            logger.warning(f"Combined selected features file not found: {features_path}")

        return model

    # 2. prepare validation data 
    def prepare_data(self, data_file: str, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], str]:
        """
        Load and process new dataset for external validation.
        """

        dataset_name = sheet_name if sheet_name else Path(data_file).stem
        logger.info(f"Preparing data for dataset: {dataset_name}")

        data_obj = self.data_processor.load_data(data_file, sheet_name)

        if isinstance(data_obj, dict):
            if sheet_name and sheet_name in data_obj:
                df = data_obj[sheet_name]
                logger.info(f"Using specified sheet '{sheet_name}' from Excel.")
            else:
                first_key = next(iter(data_obj))
                df = data_obj[first_key]
                logger.warning(f"load_data returned dict, using first sheet '{first_key}' automatically.")
        else:
            df = data_obj

        X_raw, y, structure = self.data_processor.prepare_modeling_data(df)
        if X_raw is None:
            logger.error("No feature columns found in the dataset.")
            return None, None, None, dataset_name

        constant_cols = X_raw.columns[X_raw.nunique() <= 1].tolist()
        if constant_cols:
            logger.warning(f"Dropping constant columns before physics feature creation: {constant_cols}")
            X_raw = X_raw.drop(columns=constant_cols, errors='ignore')
            
        X_physics = self.data_processor.create_physics_features(X_raw, y, fit=False)
        logger.info(f"{dataset_name} data prepared: {X_physics.shape[0]} samples, {X_physics.shape[1]} features.")

        return X_physics, y, structure, dataset_name

    # 3. validate models
    def validate_single_model(
        self, model, model_name: str,
        X_val: pd.DataFrame, y_val: pd.Series,
        structure: Optional[pd.Series],
        dataset_name: str
        ) -> Dict:
        """
        Validate a single model on new dataset
        """
        logger.info(f"Validating model: {model_name} on dataset: {dataset_name}")

        # Handle model object (in case it's a dict)
        if isinstance(model, dict) and 'model' in model:
            model = model['model']

        # === Define model features from trained model ===
        model_features = self.selected_features or X_val.columns.tolist()

        # === Align features ===
        common = [f for f in model_features if f in X_val.columns]
        missing = [f for f in model_features if f not in X_val.columns]

        if missing:
            logger.warning(f"Missing features in validation data: {missing}")
            # Add missing columns filled with zeros to preserve alignment
            for m in missing:
                X_val[m] = 0.0

        # Reorder columns to match model training order exactly
        X_aligned = X_val[model_features].copy()

        # === Apply scaler if available ===
        if self.scaler:
            try:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X_aligned),
                    columns=X_aligned.columns,
                    index=X_aligned.index
                )
            except Exception as e:
                logger.warning(f"Scaling failed: {e}. Using unscaled features.")
                X_scaled = X_aligned
        else:
            X_scaled = X_aligned

        # === Predict ===
        y_pred = model.predict(X_scaled)

        # === Metrics ===
        y_safe = np.where(y_val == 0, np.nan, y_val)
        metrics = dict(
            r2=r2_score(y_safe, y_pred),
            rmse=np.sqrt(mean_squared_error(y_safe, y_pred)),
            mae=mean_absolute_error(y_safe, y_pred),
            mape=np.nanmean(np.abs((y_safe - y_pred) / y_safe)) * 100
        )

        residual = y_safe - y_pred
        metrics.update(
            residual_mean=np.mean(residual),
            residual_std=np.std(residual),
            residual_normality_p=stats.shapiro(residual)[1],
            residual_bias_p=stats.ttest_1samp(residual, 0)[1],
            y_true=y_val,
            y_pred=y_pred,
            structure=structure
        )

        logger.info(
            f"{model_name} on {dataset_name} - "
            f"R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, "
            f"MAE: {metrics['mae']:.4f}, MAPE: {metrics['mape']:.2f}%"
        )

        return metrics

    # 4. run validation process
    def run_validation(
                self, data_file:str, sheet_name:Optional[str]=None,
                models_to_validate: Optional[Dict] = None
        ) -> Dict:
        """
            main function to run external validation
        """

        X_val, y_val, structure, dataset_name = self.prepare_data(data_file, sheet_name)
        if X_val is None or y_val is None:
            logger.error("Data preparation failed. Validation aborted.")
            return {}
            
        if models_to_validate is None:
            models_to_validate = self.load_trained_models(Path('results/model'))
            # models_to_validate = self.load_trained_models()

        results = {}

        for name, model in models_to_validate.items():
            try:
                metrics = self.validate_single_model(
                        model, name, X_val, y_val, structure, dataset_name
                    )
                results[name] = metrics

                self.visualizer.plot_parity(
                        y_true=metrics['y_true'],
                        y_pred=metrics['y_pred'],
                        # structure=metrics['structure'],
                        title = f"{dataset_name} - {name} Validation Parity Plot",
                        stage='external_validation',
                        model_type = name.split('_')[-1],
                        cluster_name = dataset_name
                    )
            except Exception as e:
                logger.error(f"Validation failed for model {name} on dataset {dataset_name}: {e}")

        self.save_validation_results(results, dataset_name)
        self.create_comparison_plots(results, dataset_name)

        return results
    
    # 5. save results
    def save_validation_results(self, results:Dict, dataset_name:str):
        """
        save csv, txt, json results
        """

        summary = [
            {
                'Model': k,
                'R2': v['r2'],
                'RMSE (eV)': v['rmse'],
                'MAE (eV)': v['mae'],
                'MAPE (%)': v['mape'],
                'Residual Mean': v['residual_mean'],
                'Residual Std': v['residual_std'],
                'Residual Normality p-value': v['residual_normality_p'],
                'Residual Bias p-value': v['residual_bias_p']
            } for k, v in results.items()
        ]

        df = pd.DataFrame(summary)
        df.to_csv(self.validation_dir / f'{dataset_name}_validation_summary.csv', index=False)

        with open(self.validation_dir / f'{dataset_name}_validation_report.txt', 'w') as f:
            f.write(f"{'='*60} Validation Report for {dataset_name} {'='*60}\n\n")
            for name, m in results.items():
                f.write(f"{name}\n{'-'*30}\n")
                f.write(f"R2={m['r2']:.3f}, RMSE={m['rmse']:.3f} eV, MAE={m['mae']:.3f} eV, MAPE={m['mape']:.2f}%\n")
                if m['r2'] > 0.8:
                    f.write("excellent generalization performance.\n\n")
                elif m['r2'] > 0.6:
                    f.write("acceptable generalization performance.\n\n")
                else:
                    f.write("poor generalization performance.\n\n")

        with open(self.validation_dir / f'{dataset_name}_results.json', 'w') as jf:
            json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, np.float32, np.float64)) else vv
                           for kk, vv in v.items() if isinstance(vv, (float, int, str))}
                       for k, v in results.items()}, jf, indent=2)
        logger.info(f"Validation results saved for dataset: {dataset_name}")

    # 6. comparison plots
    def create_comparison_plots(self, results:Dict, dataset_name:str):
        """
        create comparison plots for different models, save as png and svg files
        """
        if not results:
            return

        model_names = list(results.keys())
        r2 = [results[m]['r2'] for m in model_names]
        rmse = [results[m]['rmse'] for m in model_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
        bars1 = ax1.bar(model_names, r2, color='skyblue')
        for b, v in zip(bars1, r2):
            ax1.text(b.get_x() + b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', va='bottom')

        ax1.set_title(f'R2 Comparison - {dataset_name}')

        bars2 = ax2.bar(model_names, rmse, color='salmon')
        for b, v in zip(bars2, rmse):
            ax2.text(b.get_x() + b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', va='bottom')

        ax2.set_title(f'RMSE Comparison - {dataset_name}')

        plt.tight_layout()
        plt.savefig(self.validation_dir / f'{dataset_name}_model_comparison.png', dpi=300, transparent=True)
        plt.savefig(self.validation_dir / f'{dataset_name}_model_comparison.svg', dpi=300, transparent=True)
        plt.close()
        logger.info(f"Comparison plots saved for dataset: {dataset_name}")



# CLI entry 

def main():
    import argparse

    parser = argparse.ArgumentParser(description="External Validation for oxide_analysis_general Models")
    parser.add_argument('--data_file', required=True, help='Path to the new dataset file (CSV or Excel)')
    parser.add_argument('--sheet_name', default=None,  help='Sheet name if Excel file')
    parser.add_argument('--model_dir', default = 'results/model', help='Directory containing trained models')
    args = parser.parse_args()

    validator = Validator()
    models = validator.load_trained_models(Path(args.model_dir))
    results = validator.run_validation(args.data_file, args.sheet_name, models)

    print ("\n" + "="*80)
    print("External Validation Completed. Summary:")
    print("="*80)
    for name, m in results.items():
        print(f"{name}: R2={m['r2']:.3f}, RMSE={m['rmse']:.3f} eV, MAE={m['mae']:.3f} eV, MAPE={m['mape']:.2f}%")


if __name__ == "__main__":
    main()

