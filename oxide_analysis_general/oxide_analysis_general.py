"""
Oxide Analysis Module - Main coordinator for oxide descriptor analysis workflow
"""

import numpy as np
import shutil
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import re
from scipy.stats import pearsonr
import time
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge


# Import local modules
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .visualizer import Visualizer
from .config import DEFAULT_CONFIG, COLORS, MARKERS, FIGURES_DIR, ANALYSIS_DIR, PARAM_GRIDS, ACTIVE_CLUSTERS, FEATURE_GROUPS,FEATURE_NAME_MAP

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'Arial'
# Set Seaborn context
sns.set_context("notebook")

# Configure logger
logger = logging.getLogger(__name__)

class OxideAnalysis:
    """Main class for oxide descriptor analysis workflow"""
    def __init__(self, config: Dict = None):
        """
        Initialize the analysis workflow
        """
        # Use provided config or default
        self.config = config if config else DEFAULT_CONFIG
        if "FEATURE_GROUPS" not in self.config:
            self.config["FEATURE_GROUPS"] = FEATURE_GROUPS
        self.data_processor = DataProcessor(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.visualizer = Visualizer(self.config)
        self.config['ACTIVE_CLUSTERS'] = ACTIVE_CLUSTERS
        logger.info("Oxide analysis initialized with configuration")
    
    def _load_data(self, data_file):
        """load data from file"""
        data_dict = {}
        
        if 'Pt07' in ACTIVE_CLUSTERS:
            data_dict['Pt07'] = self.data_processor.load_data(data_file, 'Pt07')
        else:
            data_dict['Pt07'] = pd.DataFrame()
            
        if 'Pt13' in ACTIVE_CLUSTERS:
            data_dict['Pt13'] = self.data_processor.load_data(data_file, 'Pt13')
        else:
            data_dict['Pt13'] = pd.DataFrame()
            
        if 'combined' in ACTIVE_CLUSTERS:
            data_dict['combined'] = self.data_processor.load_data(data_file, 'combine')
        else:
            data_dict['combined'] = pd.DataFrame()
            
        return data_dict['Pt07'], data_dict['Pt13'], data_dict['combined']

    def _prepare_all_modeling_data(self, pt07_data, pt13_data, combined_data):
        """prepare all modeling data"""
        modeling_data = {}
        
        if 'Pt07' in ACTIVE_CLUSTERS and not pt07_data.empty:
            X_pt07_raw, y_pt07, structures_pt07 = self.data_processor.prepare_modeling_data(pt07_data)
            if X_pt07_raw is not None:
                # X_pt07_physics = self.data_processor.create_physics_features(X_pt07_raw, y_pt07)
                X_pt07_scaled, _, pt07_scaler = self.data_processor.scale_features(X_pt07_raw, method='robust')
                modeling_data['Pt07'] = (X_pt07_raw, None, X_pt07_scaled, y_pt07, structures_pt07)
        
        if 'Pt13' in ACTIVE_CLUSTERS and not pt13_data.empty:
            X_pt13_raw, y_pt13, structures_pt13 = self.data_processor.prepare_modeling_data(pt13_data)
            if X_pt13_raw is not None:
                # X_pt13_physics = self.data_processor.create_physics_features(X_pt13_raw, y_pt13)
                X_pt13_scaled, _, pt13_scaler = self.data_processor.scale_features(X_pt13_raw, method='robust')
                modeling_data['Pt13'] = (X_pt13_raw, None, X_pt13_scaled, y_pt13, structures_pt13)
        
        if 'combined' in ACTIVE_CLUSTERS and not combined_data.empty:
            X_combined_raw, y_combined, structures_combined = self.data_processor.prepare_modeling_data(combined_data)
            if X_combined_raw is not None:
                # X_combined_physics = self.data_processor.create_physics_features(X_combined_raw, y_combined)
                X_combined_scaled, _, combined_scaler = self.data_processor.scale_features(X_combined_raw, method='robust')
                modeling_data['combined'] = (X_combined_raw, None, X_combined_scaled, y_combined, structures_combined)
        
        return modeling_data

    def _split_data(self, X, y, structures, test_size=0.15, random_state=42):
        """Split data into training and testing sets"""
        X_train, X_test, y_train, y_test, struct_train, struct_test = train_test_split(
            X, y, structures, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, struct_train, struct_test

    def _train_and_evaluate_model(self, model, model_name, X_train, y_train, X_test, y_test):
        """Train and evaluate a model"""
        # Train the model
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.squeeze()
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.squeeze()
              
        model.fit(X_train, y_train)
        # Predict on training and test sets
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        # Calculate metrics
        train_metrics = self.model_trainer._calculate_metric(y_train, train_pred)
        test_metrics = self.model_trainer._calculate_metric(y_test, test_pred)
        # Log the results
        logger.info(f"{model_name}: Train R²={train_metrics['R²']:.4f}, Test R²={test_metrics['R²']:.4f}")
        # Return the results
        return {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': test_pred
        }
    
    def _train_models_on_raw_features(self, X_train, y_train, X_test, y_test, struct_train, struct_test, cluster_name):
        """
        stage 1 - Train standard models with raw features and hyperparameter tuning
        """
        logger.info (f"[{cluster_name}] for Stage 1: train standard models with raw features and hyperparameter tuning")
        raw_features = [f.replace("-", "_") for f in self.config['feature_cols'] if f.replace("-", "_") in X_train.columns]
        if not raw_features:
            logger.warning(f"No raw features found in {X_train.columns.tolist()} for {cluster_name}")
            return {}
        
        X_train_raw = X_train[raw_features].copy()
        X_test_raw = X_test[raw_features].copy()

        baseline_results = self.model_trainer.train_baseline_models(
            X_train_raw,
            y_train,
            X_test_raw,
            y_test,
            struct_train =struct_train,
            struct_test = struct_test,
            # structures=structures,
            cluster_name=cluster_name,
        )

        return {
            'models': baseline_results.get('models', {}),
            'scores': baseline_results.get('scores', {}),
            'params': baseline_results.get('params', {}),
            'optimzation_results': baseline_results.get('optimzation_results', {})
        }
    
    def _train_models_on_selected_features(self, X_train, y_train, X_test, y_test, selected_features, struct_train, struct_test, cluster_name):
        """
        stage 2 - Train standard models with selected features and hyperparameter tuning
        """
        logger.info (f"[{cluster_name}] for Stage 2: train standard models with selected features and hyperparameter tuning")

        selected_results = self.model_trainer.train_optimized_models(
            X_train[selected_features],
            y_train,
            X_test[selected_features],
            y_test,
            struct_train = struct_train,
            struct_test = struct_test,
            cluster_name=cluster_name
        )

        return {
            'models': selected_results.get('models', {}),
            'scores': selected_results.get('scores', {}),
            'params': selected_results.get('params', {}),
            'optimzation_results': selected_results.get('optimzation_results', {}),
            'selected_features': selected_features
        }
    
    def _train_on_stacking_model(self, selected_results, X_train, y_train,X_test, y_test, 
                                 struct_train, struct_test,  cluster_name):
        """
        Stage 3: Train stacking ensemble model using base models from Stage 2.
        """
        logger.info(f"[{cluster_name}] Stage 3: Training stacking model")

        base_models_dict = selected_results.get("models", {})

        # use stacking model
        stacking_results = self.model_trainer.train_stacking_model(
            base_models=base_models_dict,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            struct_train=struct_train,
            struct_test=struct_test,
            # structures=structures,
            cluster_name=cluster_name
        )

        return stacking_results

    def _find_best_model(self, model_results):
        """Find the best model based on R²"""
        if not model_results:
            logger.warning("No models found in results")
            return {'model': None, 'name': None, 'r2': 0}
        

        best_r2 = -float('inf')
        best_model_name = None
        #best_model_info = None

        for model_name, results in model_results.items():
            test_r2 = results['test_metrics']['R²']
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_model_name = model_name
        
        if best_model_name is None:
            logger.warning("No best model found")
            return {'model': None, 'name': None, 'r2': 0}
        
            
        return {'model': model_results[best_model_name]['model'], 
                'name': best_model_name,
                'r2': best_r2,
                'metrics': model_results[best_model_name]['test_metrics']}
    def _save_results(self, results: Dict) -> None:
        """save the results to the analysis directory"""
        try:
            metrics_summary = {
                'Pt07': {},
                'Pt13': {},
                'combined': {}
            }
            
            # add every model's metrics
            for dataset in ['Pt07', 'Pt13', 'combined']:
                if dataset not in results:
                    logger.warning(f"Dataset {dataset} not found in results")
                    continue

                # check if metrics key exists
                if 'metrics' in results[dataset]:
                    for model_name, metrics in results[dataset]['metrics'].items():
                        metrics_summary[dataset][model_name] = metrics

                # safely access best_model
                if 'best_model' in results[dataset]:
                    best_model = results[dataset]['best_model']
                    if isinstance(best_model, dict) and 'name' in best_model and 'metrics' in best_model:
                        metrics_summary[dataset][f"Best ({best_model['name']})"] = best_model['metrics']

            # check if combined_models key exists
            if 'combined' in results and 'combined_models' in results['combined']:
                for model_name, metrics in results['combined']['combined_models'].items():
                    metrics_summary['combined'][model_name] = metrics

            # create DataFrame and save
            metrics_df = pd.DataFrame()
            
            for dataset in metrics_summary:
                for model_name, metrics in metrics_summary[dataset].items():
                    row = {
                        'Dataset': dataset,
                        'Model': model_name,
                        'R²': metrics.get('R²', None),
                        'RMSE (eV)': metrics.get('RMSE', None)
                    }
                    metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)

            # only save if there is data
            if not metrics_df.empty:
                metrics_df.to_csv(ANALYSIS_DIR / 'model_metrics_summary.csv', index=False)

            # Write Markdown report with error checking
            with open(ANALYSIS_DIR / 'analysis_summary.md', 'w') as f:
                f.write(f"# Oxide Descriptor Analysis Summary\n\n")
                f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Performance section
                f.write("## Overall Performance\n\n")
                f.write("| Dataset | Model | R² | RMSE (eV) |\n")
                f.write("|---------|-------|------|----------|\n")
                
                for dataset in metrics_summary:
                    for model_name, metrics in metrics_summary[dataset].items():
                        r2 = metrics.get('R²', 'N/A')
                        rmse = metrics.get('RMSE', 'N/A')

                        # Format values
                        r2_formatted = f"{r2:.4f}" if isinstance(r2, (float, int)) else r2
                        rmse_formatted = f"{rmse:.4f}" if isinstance(rmse, (float, int)) else rmse
                        
                        f.write(f"| {dataset} | {model_name} | {r2_formatted} | {rmse_formatted} |\n")

                # Feature Importance section
                f.write("## Important Features\n\n")
                for dataset in ['Pt07', 'Pt13', 'combined']:
                    if dataset in results and 'best_model' in results[dataset]:
                        best_model = results[dataset]['best_model']
                        if 'model' in best_model and 'name' in best_model:
                            model = best_model['model']
                            model_name = best_model['name']
                            
                            if hasattr(model, 'feature_importances_') and 'selected_features' in results[dataset]:
                                features = results[dataset]['selected_features']
                                importances = model.feature_importances_

                                # Ensure consistent length
                                if len(importances) > len(features):
                                    importances = importances[:len(features)]
                                elif len(features) > len(importances):
                                    features = features[:len(importances)]
                                    
                                if len(features) > 0:
                                    f.write(f"### {dataset} Top Features ({model_name})\n\n")
                                    f.write("| Feature | Importance |\n")
                                    f.write("|---------|------------|\n")

                                    # Get feature importances
                                    indices = np.argsort(importances)[::-1]

                                    # Show top 10 features
                                    for i in range(min(10, len(features))):
                                        if i < len(indices):
                                            idx = indices[i]
                                            if idx < len(features):
                                                f.write(f"| {features[idx]} | {importances[idx]:.4f} |\n")

                # Conclusion section
                f.write("\n## Conclusion\n\n")
                f.write("Based on the analysis, the following observations can be made:\n\n")

                # Check if models met target performance
                best_model_info = []
                for dataset in ['Pt07', 'Pt13', 'combined']:
                    if dataset in results and 'best_model' in results[dataset]:
                        best_model = results[dataset]['best_model']
                        if 'metrics' in best_model:
                            best_r2 = best_model['metrics'].get('R²', 0)
                            best_model_info.append((dataset, best_r2))
                
                # order as r2
                best_model_info.sort(key=lambda x: x[1], reverse=True)
                
                #  add more conclustion based on r2
                for dataset, r2 in best_model_info:
                    model_name = results[dataset]['best_model']['name']
                    if r2 >= 0.95:
                        f.write(f"- {dataset}: Achieved target performance (R²={r2:.4f}) using {model_name}.\n")
                    elif r2 >= 0.8:
                        f.write(f"- {dataset}: Good performance (R²={r2:.4f}) using {model_name}, close to target.\n")
                    elif r2 >= 0.5:
                        f.write(f"- {dataset}: Moderate performance (R²={r2:.4f}) using {model_name}, more optimization needed.\n")
                    else:
                        f.write(f"- {dataset}: Insufficient performance (R²={r2:.4f}) using {model_name}, significant improvement required.\n")
            
            logger.info("Results saved to analysis directory")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def _organize_figures(self):
        """Organize saved figures into baseline/optimized/stacking subfolders"""
        base_dir = FIGURES_DIR
        if not base_dir.exists():
            logger.warning(f"Figure directory {base_dir} not found.")
            return

        subfolders = ['baseline', 'optimized', 'stacking']
        for sub in subfolders:
            (base_dir / sub).mkdir(parents=True, exist_ok=True)

        for file in base_dir.glob("*.*"):
            if file.suffix.lower() not in [".png", ".svg"]:
                continue

            fname = file.name.lower()
            moved = False
            for sub in subfolders:
                if sub in fname:
                    shutil.move(str(file), str(base_dir / sub / file.name))
                    logger.info(f"Moved {file.name} -> {sub}/")
                    moved = True
                    break

            if not moved:
                logger.debug(f"Kept {file.name} in figures/ (not baseline/optimized/stacking)")

    def _analyze_cluster(self, X_raw, X_physics, X_scaled, y, structures, cluster_name):
        """
        Analyze a single cluster: (all need to do hyperparameter tuning fisrt training)
        stage 1 - use raw features
        stage 2 - use constructed and selected features
        stage 3 - use advanced models
        """

        logger.info(f"Analyzing {cluster_name}, targeting R²=0.95")
        logger.info(f"X_raw type: {type(X_raw)}, shape: {X_raw.shape}")
        # logger.info(f"X_physics type: {type(X_physics)}, shape: {X_physics.shape}")
        logger.info(f"X_scaled type: {type(X_scaled)}, shape: {X_scaled.shape}")
        logger.info(f"y type: {type(y)}, shape: {y.shape}")
        logger.info(f"structures type: {type(structures)}, shape: {structures.shape}")

        # basic data process for step 1
        X_raw_train, X_raw_test, y_train, y_test, struct_train, struct_test = self._split_data(
            X_raw, y, structures,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        logger.info(f"Splitting data for {cluster_name} cluster: {len(X_raw_train)} train samples, {len(X_raw_test)} test samples")
        results = {}
        # === stage 1 === Train standard models with raw features and hyperparameter tuning
        if self.config.get("train_on_raw_features", True):
            logger.info(f"[{cluster_name}] for Stage 1: train standard models with raw features and hyperparameter tuning")
            results['raw'] = self._train_models_on_raw_features(
                X_raw_train, y_train, X_raw_test, y_test,
                struct_train,struct_test, cluster_name,
            )
            # Check if the model is empty
            target_col = self.config['target_col']
            available_cols = [col for col in X_raw_train.columns if col != target_col]
            Xy_filtered = X_raw_train.assign(**{target_col: y_train})[[target_col] + available_cols].copy()

            correlation_df = self.model_trainer.compute_group_correlation_with_pvalues(
                Xy_filtered, self.config["FEATURE_GROUPS"])
            if not correlation_df.empty and correlation_df["significant"].any():
                logger.info(f"Found {correlation_df['significant'].sum()} significant correlations in {cluster_name}")
                self.visualizer.plot_correlation_network(
                    correlation_df, cluster_name, 
                    group_assignments=None
                )
            else:
                logger.warning(f"No significant correlations found in {cluster_name} for raw features.")
        # === stage 2 === Train standard models with selected features and hyperparameter tuning
        if self.config.get("train_on_selected_features", True):
            logger.info(f"[{cluster_name}] for Stage 2: train standard models with constructed+raw features and hyperparameter tuning")
            X_physics_train = self.data_processor.create_physics_features(X_raw_train, y_train,fit=True)
            X_physics_test = self.data_processor.create_physics_features(X_raw_test, y_test,fit=False)
            logger.info(f"X_physics_train_columns: {X_physics_train.columns.tolist()}")
            logger.info(f"X_physics_test_columns: {X_physics_test.columns.tolist()}")
            # Ensure the physical features are consistent between train and test sets
            common_features = sorted(list(set(X_physics_train.columns) & set(X_physics_test.columns)))
            logger.info(f"[{cluster_name}] Common physical features used: {common_features}")
            X_physics_train = X_physics_train[common_features]
            X_physics_test = X_physics_test[common_features]

            #plot the pearson heatmap with updated feature names
            #add correlation network for physical features with dynamic group inference
            Xy_physics = X_physics_train.assign(**{self.config['target_col']: y_train})
            group_assignments = {}
            for col in Xy_physics.columns:
                if col!= self.config['target_col']:
                    group_assignments[col] = self.visualizer.infer_group_from_feature_name(col)
            
            dynamic_feature_groups = {}
            for col,group in group_assignments.items():
                if group not in dynamic_feature_groups:
                    dynamic_feature_groups[group] = []
                dynamic_feature_groups[group].append(col)
            
            for group, feats in dynamic_feature_groups.items():
                logger.info (f"Dynamic group '{group}' contains features: {feats}")

            corr_df_physics = self.model_trainer.compute_group_correlation_with_pvalues(
                Xy_physics, feature_groups=dynamic_feature_groups
            )
            if not corr_df_physics.empty and corr_df_physics["significant"].any():
                # only plot if significant correlations exist
                self.visualizer.plot_correlation_network(
                    corr_df_physics, cluster_name + "_physics",
                    group_assignments=group_assignments
                )
            else:
                logger.warning(f"No significant correlations found in {cluster_name} for physical features.")
            self.visualizer.plot_pearson_correlation(Xy_physics, cluster_name + "_physics", stage='stage2')

            results['selected'] = self._train_models_on_selected_features(
                X_physics_train, y_train, X_physics_test, y_test,
                  common_features, struct_train, struct_test, cluster_name
            )
            logger.info(f"[DEBUG] structures type: {type(struct_train)}, length: {len(struct_train) if struct_train is not None else 'None'}")
            logger.info(f"[DEBUG] structures type: {type(struct_test)}, length: {len(struct_test) if struct_test is not None else 'None'}")
  

        # === stage 3 === Train advanced models with selected features and hyperparameter tuning only for the meta model or the model has not been hyperparameter tuning
        if self.config.get("train_on_stacking_models", True):  
            logger.info(f"[{cluster_name}] for Stage 3: train stacking model")
            results['stacking'] = self._train_on_stacking_model(
                selected_results = results ['selected'],
                X_train = X_physics_train,
                y_train = y_train, 
                X_test = X_physics_test, 
                y_test = y_test,  
                struct_train = struct_train,
                struct_test = struct_test,
                cluster_name = cluster_name
            )  
        return results

    def run_analysis(self, data_file: str) -> Dict:
        """Run the complete analysis workflow"""
        logger.info(f"Starting analysis with {data_file}")
        start_time = time.time()
        
        # Load and prepare data
        pt07_data, pt13_data, combined_data = self._load_data(data_file)
        
        # Create correlation visualizations for all available clusters
        data_map = {
            'Pt07': pt07_data,
            'Pt13': pt13_data,
            'combined': combined_data
        }
        for cluster_name in ACTIVE_CLUSTERS:
            if cluster_name in data_map and not data_map[cluster_name].empty:
                self.visualizer.plot_pearson_correlation(data_map[cluster_name], cluster_name,stage='stage1' )
        
        # Prepare modeling data
        modeling_data = self._prepare_all_modeling_data(pt07_data, pt13_data, combined_data)

        # Run the complete building and training process for active clusters
        results = {}
        for name in ACTIVE_CLUSTERS:
            if name in modeling_data:
                logger.info(f"Starting the complete training analysis for {name} cluster...")
                cluster_results = self._analyze_cluster(
                    *modeling_data[name], cluster_name=name)
                results[name] = cluster_results
                # cluster_results = self._analyze_cluster(
                ####     *modeling_data[name], name)
                # results[name] = cluster_results
            else:
                logger.warning(f"Cluster {name} not found in modeling data")

        # Prepare arguments for comparison visualization
        cluster_args = {}
        for cluster in ['Pt07', 'Pt13', 'combined']:
            if cluster in ACTIVE_CLUSTERS:
                cluster_args[f'{cluster.lower()}_data'] = data_map[cluster]
                cluster_args[f'{cluster.lower()}_results'] = results.get(cluster, {})
            else:
                cluster_args[f'{cluster.lower()}_data'] = pd.DataFrame()
                cluster_args[f'{cluster.lower()}_results'] = {}
        
        # Create comparison visualizations
        self.visualizer.create_comparison_visualizations(**cluster_args)
        
        # Save results (only for active clusters)
        filtered_results = {cluster: results.get(cluster, {}) for cluster in ACTIVE_CLUSTERS}
        self._save_results(filtered_results)

        # Check all performances of the models
        for dataset_name, dataset_result in filtered_results.items():
            logger.info(f"==== Performance Summary for {dataset_name} ====")
            if 'models' in dataset_result:
                for model_name, model in dataset_result['models'].items():
                    if 'scores' in dataset_result and model_name in dataset_result['scores']:
                        score = dataset_result['scores'][model_name]
                        if 'R²' in score:
                            logger.info(f"Model: {model_name}, R²: {score['R²']:.4f}")
                            # Check for high-performance models
                            if score['R²'] > 0.85:
                                logger.info(f"HIGH PERFORMANCE MODEL FOUND: {dataset_name}/{model_name}")
                                logger.info(f"Parameters: {model.get_params()}")
        
        logger.info("Analysis completed!")
        self._organize_figures()
        end_time = time.time()
        total_time = end_time - start_time
        total_minutes = total_time / 60
        logger.info(f"Total time taken: {total_minutes:.2f} minutes")
        logger.info(f"Total time taken: {total_time:.2f} seconds")

        # save the modeling results in case needed to validate later
        model_save_path = Path ('results/model') 
        model_save_path.mkdir(parents=True, exist_ok=True)

        if 'combined' in filtered_results:
            if 'selected' in filtered_results['combined']:
                selected_data = filtered_results['combined'].get('selected', {})

                if 'models' in selected_data:
                    for model_name, model_obj in selected_data['models'].items():
                        if model_obj is not None:
                            metrics = selected_data['scores'].get(model_name, {})
                            logger.info(f"Combined model: {model_name}, Test Metrics: {metrics}")


                            save_path = model_save_path / f'combined_{model_name}_model.pkl'
                            with open(save_path, 'wb') as f:
                                pickle.dump(model_obj,f)
                                # pickle.dump(model_obj['model'], f)
                            logger.info(f"Saved combined model {model_name} to {save_path}")
                
                ## save the features
                if 'selected_features' in selected_data:
                    save_path = model_save_path / f'combined_selected_features.pkl'
                    with open(save_path, 'wb') as f:
                        pickle.dump(selected_data['selected_features'],f)
                    logger.info(f"Saved combined selected features to {save_path}")
            
                ## save scaler
                if hasattr(self.data_processor, 'combined_scaler'):
                    save_path = model_save_path / f'combined_scaler.pkl'
                    with open(save_path, 'wb') as f:
                        pickle.dump(self.data_processor.combined_scaler,f)
                    logger.info(f"Saved combined scaler to {save_path}")

        logger.info("Model saving completed.")

        return filtered_results
    


