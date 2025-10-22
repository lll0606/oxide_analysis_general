"""
Model Training Module for Oxide Descriptor Analysis

This module provides functionality for training, evaluating, and optimizing
various machine learning models for predicting adsorption energies.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from .config import PARAM_GRIDS,BAYES_GRIDS
from .visualizer import Visualizer
from .data_processor import DataProcessor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import (
    cross_val_score, cross_val_predict, GridSearchCV
)
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    VotingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.base import clone
from sklearn.inspection import permutation_importance


# Configure logger
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class for training and evaluating machine learning models"""
    
    def __init__(self, config: Dict):
        """
        Initialize the model trainer with configuration
        """
        self.config = config
        self.data_processor = DataProcessor(config)
        self.visualizer = Visualizer(config)
        self.random_state = config["random_state"]
        self.test_size = config["test_size"]
        self.cv_folds = config["cv_folds"]
        self.n_jobs = config["n_jobs"]
        self.advanced_models = config.get("advanced_models", {})
        logger.info("Model trainer initialized")

    def _calculate_metric(self,y_true,y_pred):
        """
        Calculate R², RMSE, and MAE metrics
        
        """
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        return {
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae
        }

    def compute_group_correlation_with_pvalues(self, df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Compute Pearson correlation and Bonferroni-corrected p-values between feature groups.
        If feature_groups is empty, it will infer group for each feature using infer_group_from_feature_name().
        """
        from scipy.stats import pearsonr
        from statsmodels.stats.multitest import multipletests

        logger.info(f"[DEBUG] Starting correlation analysis")
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        df = df.select_dtypes(include=[np.number])
        logger.debug(f"Feature groups: {feature_groups}")

        # Automatically infer group assignments if not provided
        if not feature_groups:
            logger.info("No feature_groups provided, inferring groups using infer_group_from_feature_name")
            feature_groups = {}
            for feat in df.columns:
                if feat == self.config['target_col']:
                    continue
                group = self.visualizer.infer_group_from_feature_name(feat)
                if group not in feature_groups:
                    feature_groups[group] = []
                feature_groups[group].append(feat)
            for group, feats in feature_groups.items():
                logger.info(f"Infferred group '{group}' with {len(feats)} features: {feats}")

        results = []

        # Loop through group pairs
        groups = list(feature_groups.keys())
        for i, group_a in enumerate(groups):
            for j, group_b in enumerate(groups):
                if i >= j:    
                    continue

                logger.debug(f"---- Comparing '{group_a}' with '{group_b}'-----")

                features_a = feature_groups[group_a]
                features_b = feature_groups[group_b]


                for feat_a in features_a:
                    for feat_b in features_b:
                        if feat_a == feat_b:
                            continue
                        if feat_a not in df.columns or feat_b not in df.columns:
                            logger.warning(f"[SKIP] Missing feature: {feat_a} or {feat_b} in DataFrame")
                            continue
                        x = df[feat_a]
                        y = df[feat_b]
                        if x.isnull().any() or y.isnull().any():
                            logger.warning(f"[SKIP] NaN values found in {feat_a} or {feat_b}, skipping correlation")
                            continue
                        try:
                            corr, pval = pearsonr(x, y)
                            results.append({
                                "group_a": group_a,
                                "group_b": group_b,
                                "feature_a": feat_a,
                                "feature_b": feat_b,
                                "correlation": corr,
                                "p_value": pval,
                            })
                        except Exception as e:
                            logger.warning(f"Correlation failed for {feat_a} vs {feat_b}: {e}")
        df_result = pd.DataFrame(results)
        if df_result.empty:
            logger.warning("No valid feature pairs found for correlation or p-value correction.")
            return pd.DataFrame(columns=[
                "group_a", "group_b", "feature_a", "feature_b", "correlation", "p_value",
                "p_value_corrected", "significant"
            ])
        logger.info(f"[INFO] Total valid correlation or p-value pairs: {len(df_result)}")
        logger.debug(f"Applying Bonferroni correction to {len(df_result)} p-value pairs")

        # Apply Bonferroni correction
        corrected = multipletests(df_result["p_value"], method="bonferroni")
        df_result["p_value_corrected"] = corrected[1]
        df_result["significant"] = corrected[0]

        sig_count = df_result["significant"].sum()
        logger.info(f"Found {sig_count} significant correlations after Bonferroni correction")

        return df_result

    def analyze_feature_importance(self, model, feature_names, X=None, y=None):
        """
        Analyze feature importance for a model
        """
        feature_names = [f for f in feature_names if f in self.config.get("feature_names", [])]
        importance_data = {}
        
        # Check if model has feature_importances_ attribute (tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Create sorted importance data
            importance_data['feature_importances'] = {
                'features': [feature_names[i] for i in indices],
                'importance': importances[indices],
                'indices': indices
            }
            
            # Log top features
            logger.info("Top 10 features by importance:")
            for i in range(min(10, len(indices))):
                logger.info(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # If data is provided, calculate permutation importance
        if X is not None and y is not None:
            try:
                # Permutation importance is model-agnostic
                perm_importance = permutation_importance(
                    model, X, y, n_repeats=10, random_state=self.random_state
                )
                
                # Sort by mean importance
                perm_indices = perm_importance.importances_mean.argsort()[::-1]
                
                importance_data['permutation_importance'] = {
                    'features': [feature_names[i] for i in perm_indices],
                    'importance_mean': perm_importance.importances_mean[perm_indices],
                    'importance_std': perm_importance.importances_std[perm_indices],
                    'indices': perm_indices
                }
                
                # Log top features by permutation importance
                logger.info("Top 10 features by permutation importance:")
                for i in range(min(10, len(perm_indices))):
                    logger.info(
                        f"{i+1}. {feature_names[perm_indices[i]]}: "
                        f"{perm_importance.importances_mean[perm_indices[i]]:.4f} ± "
                        f"{perm_importance.importances_std[perm_indices[i]]:.4f}"
                    )
            except Exception as e:
                logger.warning(f"Failed to calculate permutation importance: {str(e)}")
        
        return importance_data

    def feature_contribution_analysis(self, model, X, feature_names):
        """
        Analyze how individual features contribute to predictions
        """
        # This works best with tree-based models
        feature_names = [f for f in feature_names if f in self.config.get("feature_names", [])]
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute, analysis may be limited")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)

        results = {}
        
        # Get baseline predictions
        baseline_pred = model.predict(X)
        
        # Analyze each feature by permutation
        for i, feature in enumerate(feature_names):
            # Create a shuffled version of this feature
            X_shuffled = X.copy()
            X_shuffled[feature] = np.random.permutation(X_shuffled[feature].values)
            
            # Get predictions with shuffled feature
            shuffled_pred = model.predict(X_shuffled)
            
            # Calculate impact of shuffling (higher difference = more important)
            impact = np.mean(np.abs(baseline_pred - shuffled_pred))
            results[feature] = impact
        
        # Sort features by impact
        sorted_features = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        # Log top features by impact
        logger.info("Top features by contribution impact:")
        for i, (feature, impact) in enumerate(sorted_features[:10]):
            logger.info(f"{i+1}. {feature}: {impact:.4f}")
        
        return {
            'feature_impact': dict(sorted_features),
            'top_features': [f for f, _ in sorted_features[:min(10, len(sorted_features))]]
        }

    def optimize_model_hyperparameters(self, X, y, model_type='rf', param_grid=None):
        """
        Optimize model hyperparameters using grid search
        """
        logger.info(f"Optimizing hyperparameters for {model_type} model")
        
        #dataframe preprocessing
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).copy()

        too_large = (X.abs() > 1e10).any(axis=1)
        if too_large.any():
            logger.warning(f"Dropping {too_large.sum()} samples due to NaN/Inf in X or y.")
            X = X.loc[~too_large].reset_index(drop=True)
            y = y.loc[~too_large].reset_index(drop=True)

        X = X.reset_index(drop=True).astype(np.float64)
        y = y.reset_index(drop=True).astype(np.float64)

        from sklearn.model_selection import GridSearchCV
        if model_type == 'rf':
            model = RandomForestRegressor(random_state=self.random_state)
        elif model_type == 'gb':
            model = GradientBoostingRegressor(random_state=self.random_state)
        elif model_type == 'xgb':
            X = X.to_numpy()
            y = y.to_numpy()
            try:
                from xgboost import XGBRegressor
                model = XGBRegressor(random_state=self.random_state)
            except ImportError:
                logger.error("XGBoost not available")
                return None, None, None
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None, None, None

        # choose parameter grid from the config
        if param_grid is None:
            param_grid = PARAM_GRIDS.get(model_type)
            if param_grid is None:
                logger.error(f"No parameter grid found for {model_type} model")
                return None, None
        logger.info(f"PARAM_GRID used in grid search for {model_type}: {param_grid}")
        # covert  to numpy if xgb to avoid dtype error
        if model_type == 'xgb':
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.to_numpy()
        
        # Set up cross-validation
        cv = min(self.cv_folds, len(X) // 3)
        if cv < 2:
            logger.warning("Not enough samples for cross-validation, using 2 folds")
            cv = 2 

        # grid search
        try:
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='r2',
                n_jobs=self.n_jobs, verbose=1, return_train_score=True
            )
            grid_search.fit(X, y)
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best cross-validation score: {best_score:.4f}")
            legal_keys = model.get_params().keys()
            filtered_params = {k: v for k, v in best_params.items() if k in legal_keys}
            optimized_model = clone(model).set_params(**filtered_params)
            optimized_model.fit(X, y)    
            logger.info(f"[DEBUG] Returned grid_search type: {type(grid_search)}")
            return optimized_model, best_params, grid_search
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            return None, None, None

    def bayesian_optimize_hyperparameters(self, X, y, model_type='rf', n_calls=30, param_space=None):
        """
        Optimize model hyperparameters using Bayesian optimization 
        """
        try:
            from skopt import BayesSearchCV
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            logger.error("Scikit-Optimize not available")
            return None, None, None
        
        logger.info(f"Bayesian hyperparameter optimization for {model_type} model")

        # --- Sanitize data ---
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).copy()

        invalid_mask = (
            X.isna().any(axis=1) |
            ~np.isfinite(X).all(axis=1) |
            (X.abs() > 1e10).any(axis=1) |
            y.isna() |
            ~np.isfinite(y)
        )
        if invalid_mask.any():
            logger.warning(f"[{model_type}] Dropping {invalid_mask.sum()} rows with NaN / Inf / too-large values.")
            X = X.loc[~invalid_mask].reset_index(drop=True)
            y = y.loc[~invalid_mask].reset_index(drop=True)

        # Convert to float64 for safety
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        # For xgboost, ensure numpy type
        if model_type == 'xgb':
            X = X.to_numpy()
            y = y.to_numpy()

        # handle DataFrame input
        if model_type == 'xgb':
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            if isinstance(y, pd.DataFrame):
                y = y.to_numpy()

        # define model first, before using it
        if model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=self.random_state)
        elif model_type == 'gb':
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(random_state=self.random_state)
        elif model_type == 'xgb':
            try:
                from xgboost import XGBRegressor
                model = XGBRegressor(random_state=self.random_state)
            except ImportError:
                logger.error("XGBoost not available")
                return None, None, None
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None, None, None
        
        # Define parameter spaces for different model types
        if param_space is None:
            logger.warning("There is no parameter space provided, Using default parameter space")
            if model_type == 'rf':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=self.random_state)
                param_space = {
                    'n_estimators': Integer(50, 500),
                    'max_depth': Integer(1, 30),
                    'max_features': Real(0.1, 1.0),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 10)
                }
            elif model_type == 'gb':
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=self.random_state)
                param_space = {
                    'n_estimators': Integer(50, 500),
                    'learning_rate': Real(0.01, 0.3, 'log-uniform'),
                    'max_depth': Integer(1, 10),
                    'subsample': Real(0.5, 1.0),
                    'min_samples_split': Integer(2, 20),
                    'max_features': Real(0.1, 0.9)
                }
            elif model_type == 'xgb':
                try:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(random_state=self.random_state)
                    param_space = {
                        'n_estimators': Integer(50, 500),
                        'learning_rate': Real(0.01, 0.3, 'log-uniform'),
                        'max_depth': Integer(1, 10),
                        'subsample': Real(0.5, 1.0),
                        'colsample_bytree': Real(0.5, 1.0),
                        'min_child_weight': Integer(1, 10)
                    }
                except ImportError:
                    logger.error("XGBoost not available")
                    return None, None, None
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        # calculate suitable cross-validation
        cv = min(self.cv_folds, len(X) // 3)
        if cv < 2:
            cv = 2
            
        # create bayesian optimization object
        opt = BayesSearchCV(
                model,
                param_space,
                n_iter=n_calls,
                cv=cv,
                scoring='r2',
                # n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        
        logger.info(f"Starting bayesain optimization for {model_type} model")
        opt.fit(X, y)

        best_params = opt.best_params_
        best_score = opt.best_score_
        logger.info(f"Best parameters for {model_type} parameters: {best_params}")
        logger.info(f"Best cross-validation score for {model_type}: {best_score:.4f}")

        # clone the model and set parameters
        from sklearn.base import clone 
        leagal_keys = model.get_params().keys()
        filtered_params = {k: v for k, v in best_params.items() if k in leagal_keys}
        final_model = clone(model).set_params(**filtered_params)
        final_model.fit(X, y)

        return final_model, best_params, opt

    def optimize_with_grid_and_bayes(self, X, y, model_type='rf'):
        """
        2-steps methods: Optimize model hyperparameters using grid search first and Bayesian optimization second
        """
        from skopt.space import Real, Integer, Categorical
        import numpy as np
        from .config import PARAM_GRIDS, BAYES_GRIDS

        # first step: grid search
        logger.info(f"First step: grid search to find {model_type} model the best hyperparameters")
        grid_model, grid_params, opt_results  = self.optimize_model_hyperparameters(
            X,y, model_type=model_type, param_grid=PARAM_GRIDS.get(model_type)
            )

        if grid_model is None or grid_params is None:
            logger.error("Grid search failed, stopping optimization")
            return None, None, None

        logger.info(f"grid search best parameters: {grid_params}")
        # based on grid search results, define Bayesian optimization space
        logger.info("based on grid search results, define Bayesian optimization space")

        param_space = None
        if model_type == 'rf':
            # make sure lower and uppper bounds are not the same
            # make sure none values are properly handled
            max_depth = grid_params.get('max_depth')
            n_estimators = grid_params.get('n_estimators')
            min_samples_split = grid_params.get('min_samples_split', 2)
            min_samples_leaf = grid_params.get('min_samples_leaf', 1)

            max_depth_lower = 3 if max_depth is None else max(3, max_depth - 3)
            max_depth_upper = max(max_depth_lower + 2, 25 if max_depth is None else max_depth + 5)

            n_estimators_lower = 50 if n_estimators is None else max(50, n_estimators - 50)
            n_estimators_upper = max(n_estimators_lower + 50, 500 if n_estimators is None else n_estimators + 100)
            
            min_samples_split_lower = 2 if min_samples_split is None else max(2, min_samples_split - 2)
            min_samples_split_upper = max(min_samples_split_lower + 3, 15 if min_samples_split is None else min_samples_split + 5)

            min_samples_leaf_lower = 1 if min_samples_leaf is None else max(1, min_samples_leaf - 1)
            min_samples_leaf_upper = max(min_samples_leaf_lower + 2, 10 if min_samples_leaf is None else min_samples_leaf + 3)

            param_space = {
                'n_estimators': Integer(n_estimators_lower, n_estimators_upper),
                'max_depth': Integer(max_depth_lower, max_depth_upper),
                'max_features': Real(0.1,1.0),
                'min_samples_split': Integer(min_samples_split_lower, min_samples_split_upper),
                'min_samples_leaf': Integer(min_samples_leaf_lower, min_samples_leaf_upper)
            }

        elif model_type == 'gb':
            # gradient boosting parameters space
            # same as random forest, make sure lower and upper bounds are not the same
            # make sure none value are properly handled
            max_depth = grid_params.get('max_depth')
            n_estimators = grid_params.get('n_estimators')
            learning_rate = grid_params.get('learning_rate')
            subsample = grid_params.get('subsample', 0.5)

            max_depth_lower = 3 if max_depth is None else max(3, max_depth - 3)
            max_depth_upper = max(max_depth_lower + 2, 25 if max_depth is None else max_depth + 5)
                                  
            n_estimators_lower = 50 if n_estimators is None else max(50, n_estimators - 50)
            n_estimators_upper = max(n_estimators_lower + 50, 500 if n_estimators is None else n_estimators + 100)

            learning_rate_lower = 0.01 if learning_rate is None else max(0.01, learning_rate / 2)
            learning_rate_upper = min(0.3, learning_rate * 2)
    

            param_space = {
                'n_estimators': Integer(n_estimators_lower, n_estimators_upper),
                'learning_rate': Real(learning_rate_lower, learning_rate_upper, prior='log-uniform'),
                'max_depth': Integer(max_depth_lower, max_depth_upper),
                'subsample':Real(0.1,1.0),
                'min_samples_split': Integer(2, 10)
            }
            
        elif model_type == 'xgb':
            try:
            # XGBoost parameters space
            # same as random forest, make sure lower and upper bounds are not the same
                # make sure none values are properly handled
                max_depth = grid_params.get('max_depth')
                n_estimators = grid_params.get('n_estimators')
                learning_rate = grid_params.get('learning_rate')
                subsample = grid_params.get('subsample')
                colsample_bytree = grid_params.get('colsample_bytree')

                n_estimators_lower = 50 if n_estimators is None else max(50, n_estimators - 50)
                n_estimators_upper = max(n_estimators_lower + 50, 500 if n_estimators is None else n_estimators + 100)
                
                learning_rate_lower = 0.01 if learning_rate is None else max(0.01, learning_rate / 2)
                learning_rate_upper = min(0.3, learning_rate * 2)

                max_depth_lower = 3 if max_depth is None else max(3, max_depth - 3)
                max_depth_upper = max(max_depth_lower + 2, 25 if max_depth is None else max_depth + 5)

                colsample_bytree_lower = 0.5 if colsample_bytree  is None else max(0.5, colsample_bytree  - 0.1)
                colsample_bytree_upper = min(1.0, colsample_bytree  + 0.1)
                
                subsample_lower = 0.5 if subsample is None else max(0.5, subsample - 0.1)
                subsample_upper = min(1.0, subsample + 0.1)
                
                param_space = {
                    'n_estimators': Integer(n_estimators_lower, n_estimators_upper),
                    'learning_rate': Real(learning_rate_lower, learning_rate_upper, prior='log-uniform'),
                    'max_depth': Integer(max_depth_lower, max_depth_upper),
                    'subsample': Real(subsample_lower, subsample_upper),
                    'colsample_bytree': Real(colsample_bytree_lower, colsample_bytree_upper),
                    'min_child_weight': Integer(1, 10)
                }

            except ImportError:
                logger.error("XGBoost not available")
                param_space = None
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            return grid_model, grid_params, opt_results
        
        # bayesian hyperparameter optimization
        logger.info("Second step: Bayesian optimization to refine hyperparameters")
        bayes_model, bayes_params, opt_results = self.bayesian_optimize_hyperparameters(
            X, y, model_type=model_type, n_calls=30, param_space=param_space)
        
        # if Bayesian optimization failed, return grid search results
        if bayes_model is None:
            logger.warning("Bayesian optimization failed, using grid search results")
            return grid_model, grid_params, opt_results
        
        # we use simple cross-validation to compare grid search and Bayesian optimization results
        from sklearn.model_selection import cross_val_score
        cv = min(self.cv_folds, len(X) // 3)
        if cv < 2:
            cv = 2
        
        grid_scores = cross_val_score(grid_model, X, y, cv=cv, scoring='r2', n_jobs=self.n_jobs)
        bayes_scores = cross_val_score(bayes_model, X, y, cv=cv, scoring='r2')

        grid_mean = np.mean(grid_scores)
        bayes_mean = np.mean(bayes_scores)

        logger.info(f"Grid search mean R²: {grid_mean:.4f}")
        logger.info(f"Bayesian optimization mean R²: {bayes_mean:.4f}")

        # return the best model based on cross-validation
        if bayes_mean > grid_mean:
            logger.info("Bayesian optimization improved model performance")
            logger.info(f"[DEBUG] Final opt_results type by bayes: {type(opt_results)}")
            return bayes_model, bayes_params, opt_results
        else:
            logger.info("Grid search model is more sufficient")
            logger.info(f"[DEBUG] Final opt_results type by grid: {type(opt_results)}")
            return grid_model, grid_params, opt_results

    def train_baseline_models(self, X_train, y_train, X_test=None, y_test=None,
                              struct_train=None, struct_test=None, cluster_name = None):
        """
        Train standard models (rf/gb/xgb) using original feature set only.
        Includes hyperparameter optimization and re-training.
        """
        from .config import PARAM_GRIDS
        from sklearn.metrics import r2_score
        from pathlib import Path

        model_types = ['rf', 'gb', 'xgb']
        results = {'models': {}, 'params': {}, 'scores': {}, 'optimization_results': {}}

        logger.info(f"[DEBUG] train_baseline_models: input X_train.columns: {X_train.columns.tolist()}")

        if struct_train is not None and struct_test is not None:
            structures_all = np.concatenate([struct_train, struct_test])
        else:
            structures_all = None
        
        train_indices = np.arange(len(y_train))
        test_indices = np.arange(len(y_train), len(y_train) + len(y_test)) if X_test is not None else None

        for model_type in model_types:
            try:
                X_train = X_train.select_dtypes(include=[np.number])
                if X_test is not None:
                    X_test = X_test.select_dtypes(include=[np.number])
                model, best_params, opt_results = self.optimize_with_grid_and_bayes(
                    X_train, y_train, model_type=model_type
                )
                logger.info(f"Best parameters for {model_type}: {best_params}")
                if opt_results is not None and hasattr(opt_results, 'best_score_'):
                    logger.info(f"Best cross-validation score for {model_type}: {opt_results.best_score_:.4f}")
                else:
                    logger.warning(f"{model_type} returned no valid optimizer result (opt_results type: {type(opt_results)})")
                results['params'][model_type] = best_params
                results['optimization_results'][model_type] = opt_results
                try:
                    # plot the grid search results
                    self.visualizer.visualize_hyperparameter_tuning(
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        model_type=model_type,
                        cluster_name=cluster_name,
                        stage='baseline'
                    )
                except Exception as e:
                    logger.warning(f"Failed to visualize gridsearch hyperparameter tuning for {model_type} in baseline model training: {str(e)}")
                # plot the bayesian optimization results
                try:
                    opt_data = getattr(opt_results, 'optimizer_results_', [None])[0]
                    if opt_data is not None:
                        save_path = self.visualizer.get_figure_save_path(
                            cluster_name=cluster_name,
                            model_type=model_type,
                            stage='baseline',
                            plot_type='bayesian_optimization',
                            fmt='png'

                        )
                        self.visualizer.visualize_bayesian_optimization(
                            optimization_results=opt_data,
                            model_type=model_type,
                            cluster_name=cluster_name,
                            stage='baseline',
                            save_path=save_path
                        )
                except Exception as e:
                    logger.warning(f"Failed to visualize Bayesian optimization results for {model_type}: {str(e)}")

                legal_params = model.get_params().keys()
                filtered_params = {k: v for k, v in best_params.items() if k in legal_params}
                model.set_params(**filtered_params)

                if isinstance(y_train, pd.DataFrame):
                    y_train = y_train.squeeze()
                model.fit(X_train, y_train)
                results['models'][model_type] = model

                if X_test is not None and y_test is not None:
                    if isinstance(y_test, pd.DataFrame):
                        y_test = y_test.squeeze()
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    results['scores'][model_type] = {'R²': r2}
                    logger.info(f"{model_type} model R² on test set: {r2:.4f}")
                else:
                    y_pred = None
                          
                # plot parity
                # combine y_test and y_pred
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                y_true_all = np.concatenate([y_train, y_test])
                y_pred_all = np.concatenate([train_pred, test_pred])

                try:
                    self.visualizer.log_errors_by_threshold(
                        y_true=y_true_all,
                        y_pred=y_pred_all,
                        X=pd.concat([X_train, X_test], axis=0).reset_index(drop=True),
                        structures=structures_all,
                        cluster_name=cluster_name,
                        stage='baseline',
                        model_type=model_type,
                        threshold=1.5
                    )
                except Exception as e:
                    logger.warning(f"[{cluster_name}-{model_type}] Failed to log large error points in baseline stage: {str(e)}")

                self.visualizer.plot_parity(
                    y_true = y_true_all, 
                    y_pred = y_pred_all, 
                    structures = structures_all,
                    train_indices= train_indices,
                    test_indices=test_indices,
                    title = f"{cluster_name} {model_type} (baseline)",
                    stage = "baseline",
                    model_type=model_type,
                    cluster_name=cluster_name,
                )
                # plot the feature importance
                try:
                    feature_names = X_train.columns.tolist()
                    allowed_features = [f.replace("-", "_") for f in self.config["feature_cols"]]
                    feature_names = [f for f in feature_names if f in allowed_features]
                    group_assignment = {f: g for g, feats in self.config.get("FEATURE_GROUPS", {}).items() for f in feats}
                    self.visualizer.plot_feature_importance(
                        model=model,
                        feature_names=feature_names,
                        cluster_name=cluster_name,
                        model_type=model_type,
                        stage='baseline',
                        group_assignment=group_assignment,
                        title=f"{cluster_name} {model_type} (baseline)",
                    )
                except Exception as e:
                    logger.warning(f"Failed to plot feature importance for {model_type} in baseline model training: {str(e)}")

            except Exception as e:
                logger.warning(f"{model_type} failed during baseline training: {str(e)}")
        return results

    def train_optimized_models(self, X_train, y_train,  X_test = None, y_test = None, 
                               struct_train = None, struct_test = None, cluster_name=None):
        """train optimized model to achieve high R²"""
        import pandas as pd
        import numpy as np
        from sklearn.metrics import r2_score
        
        logger.info("start training optimized model")

        if struct_train is not None and struct_test is not None:
            structures_all = np.concatenate([struct_train, struct_test])
        else:
            structures_all = None

        train_indices = np.arange(len(y_train))
        test_indices = np.arange(len(y_train), len(y_train) + len(y_test)) if X_test is not None else None

        # check if y is a 
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.squeeze()
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.squeeze()  

        # 1. create physical features
        X_train_selected = X_train.copy()
        X_test_selected = X_test.copy() if X_test is not None else None
        selected_features = X_train_selected.columns.tolist()
        logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape if X_test is not None else None}")
        
        
        results = {
            'feature_importance': {},
            'feature_contribution': {},
            'selected_features': selected_features,
            'train_data': X_train_selected,
            'train_labels': y_train,
            'test_data': X_test_selected,
            'test_labels': y_test,
            'models': {}
        }

        df_all = pd.concat([X_train_selected, X_test_selected], axis=0) if X_test_selected is not None else X_train_selected
        group_assignment = self.visualizer.assign_feature_to_group_by_correlation(df_all)

        model_types = ['rf', 'gb', 'xgb']
        for model_type in model_types:
            model = None
            best_params = None
            opt_results = None

            try:
                model, best_params, opt_results = self.optimize_with_grid_and_bayes(
                    X_train_selected, y_train, model_type=model_type
                )
                logger.info(f"Best parameters after grid_and_bayes for {model_type} in optimized model: {best_params}")
            except Exception as e:
                logger.warning(f"{model_type} optimization failed with grid_and_bayes method in optimzied model: {str(e)}, back to grid search")
                model, best_params, opt_results = self.optimize_model_hyperparameters(
                    X_train_selected, y_train, model_type=model_type
                )
            
            results['models'][model_type] = model
            results[f"{model_type}_params"] = best_params
            results[f"{model_type}_opt_results"] = opt_results

            # plot the hyperparameter optimization tuning results
            #plot the grid search results
            try:
                self.visualizer.visualize_hyperparameter_tuning(
                    X_train = X_train_selected,
                    y_train = y_train,
                    X_test = X_test_selected,
                    y_test = y_test,
                    model_type= model_type,
                    cluster_name= cluster_name,
                    stage='optimized'
                )
            except Exception as e:
                logger.warning(f"Failed to visualize gridsearch hyperparameter tuning for {model_type} in optimized model: {str(e)}")
            # plot the bayesian optimization results
            try:
                opt_data = getattr(opt_results, 'optimizer_results_', [None])[0]
                if opt_data is not None:
                    save_path = self.visualizer.get_figure_save_path(
                        cluster_name=cluster_name,
                        model_type=model_type,
                        stage='optimized',
                        plot_type='bayesian_optimization',
                        fmt='png'
                    )
                    self.visualizer.visualize_bayesian_optimization(
                        optimization_results=opt_data,
                        model_type=model_type,
                        cluster_name=cluster_name,
                        stage='optimized',
                        save_path=save_path
                    )
            except Exception as e:
                logger.warning(f"Failed to visualize Bayesian optimization results for {model_type} in optimized model: {str(e)}")
            # fit the model
            if X_test_selected is not None and y_test is not None:
                try:
                    # plot parity
                    train_pred = model.predict(X_train_selected)
                    test_pred = model.predict(X_test_selected)

                    y_true_all = np.concatenate([y_train, y_test])
                    y_pred_all = np.concatenate([train_pred, test_pred])

                    try:
                        self.visualizer.log_errors_by_threshold(
                            y_true=y_true_all,
                            y_pred=y_pred_all,
                            X=pd.concat([X_train_selected, X_test_selected], axis=0).reset_index(drop=True),
                            structures=structures_all,
                            cluster_name=cluster_name,
                            stage='optimized',
                            model_type=model_type,
                            threshold=1.5
                        )
                    except Exception as e:
                        logger.warning(f"[{cluster_name}-{model_type}] Failed to log large error points in optimized stage: {str(e)}")

                    self.visualizer.plot_parity(
                        y_true = y_true_all,
                        y_pred = y_pred_all,
                        structures = structures_all,
                        train_indices= train_indices,
                        test_indices=test_indices, 
                        title = f"{cluster_name} {model_type} (optimized)",
                        stage = "optimized",
                        model_type=model_type,
                        cluster_name=cluster_name,
                    )
                    # plot the feature importance
                    try:
                        feature_names = selected_features
                        self.visualizer.plot_feature_importance(
                            model=model,
                            feature_names=feature_names,
                            cluster_name=cluster_name,
                            model_type=model_type,
                            stage='optimized',
                            group_assignment=group_assignment 
                        )
                    except Exception as e:
                        logger.warning(f"Failed to plot feature importance for {model_type} in optimized model: {str(e)}")
                except Exception as e:
                    logger.warning(f"Failed to evaluate or visualize hyperparameter tuning for {model_type} in optimized model: {str(e)}")
        # train stacked model
        base_models = {k: v for k, v in results['models'].items() if v is not None}
        if base_models:
            stack_results = self.train_stacking_model(
                base_models=base_models,
                X_train=X_train_selected,
                y_train=y_train,
                X_test=X_test_selected,
                y_test=y_test,
                struct_train=struct_train,
                struct_test=struct_test,
                cluster_name=cluster_name
            )
            results['stacked_ensemble'] = stack_results
        else:
            results['stacked_ensemble'] = {}

        logger.info("optimized model training completed")
        return results
  
    def train_stacking_model(self, base_models, X_train, y_train, X_test=None, y_test=None, 
                             struct_train=None, struct_test=None, cluster_name=None):
        """
        Train a stacking ensemble model using pre-trained base models.
        """
        import numpy as np
        from sklearn.model_selection import KFold, GridSearchCV
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        logger.info(f"[{cluster_name}] Training stacking ensemble model")

        if struct_train is not None and struct_test is not None:
            structures_all = np.concatenate([struct_train, struct_test])
        else:
            structures_all = None
        train_indices = np.arange(len(y_train))
        test_indices = np.arange(len(y_train), len(y_train) + len(y_test)) if X_test is not None else None

        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.squeeze()
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.squeeze()  

        base_model_list = list(base_models.items())
        meta_features = np.zeros((X_train.shape[0], len(base_model_list)))

        # Cross-validated meta-feature generation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for i, (name, model) in enumerate(base_model_list):
            if model is None:
                logger.warning(f"Base model {name} is None, skipping")
                continue
            logger.info(f"[{cluster_name}] Generating meta features for base model: {name}")
            oof_preds = np.zeros(X_train.shape[0])
            for train_idx, val_idx in kf.split(X_train):
                model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                oof_preds[val_idx] = model.predict(X_train.iloc[val_idx])
            meta_features[:, i] = oof_preds

        # gridsearchcv on meta feature (ridge regression)
        logger.info(f"[{cluster_name}] Training meta model (Ridge regression)")
        alpha_values = [0.1, 1.0, 10.0, 20.0]
        param_grid = {'alpha': alpha_values}
        grid = GridSearchCV(
            Ridge(),param_grid,scoring='r2',
            cv=5, return_train_score=True
        )
        grid.fit(meta_features, y_train)
        
        # filter parameters to avoid invalid values (e.g. n_jobs )
        meta_model = Ridge()
        legal_params = meta_model.get_params().keys()
        filtered_params = {k: v for k, v in grid.best_params_.items() if k in legal_params}
        meta_model.set_params(**filtered_params)
        meta_model.fit(meta_features, y_train)

        train_preds = meta_model.predict(meta_features)
        logger.info(f"[{cluster_name}] Best alpha for Ridge (meta model): {grid.best_params_['alpha']}")
        # get the results
        results = {
            'final_model': meta_model,
            'base_models': base_models,
            'train_meta_features': meta_features,
            'train_preds': train_preds
        }
        try:
            save_path = self.visualizer.get_figure_save_path(
                stage='stacking',
                model_type='ridge',
                cluster_name=cluster_name,
                plot_type='meta_model_alpha_tuning',
                fmt ='png'
            )
            self.visualizer.visualize_meta_model_hyperparameters(
                cv_results=grid.cv_results_,
                param_name='alpha',
                param_values=alpha_values,
                cluster_name=cluster_name,
                title=f"{cluster_name} Ridge Meta Model Hyperparameter Tuning",
                save_path=save_path
            )
        except Exception as e:
            logger.warning(f"[{cluster_name}] Failed to visualize meta model hyperparameter tuning: {str(e)}")
        
        # refit base models on the entire training set
        refit_base_models = {}
        for name, model in base_model_list:
            try:
                model.fit(X_train, y_train)
                refit_base_models[name] = model
            except Exception as e:
                logger.warning(f"[{cluster_name}] Failed to refit base model {name} in the stacking stage: {str(e)}")

        test_meta_features_list = []
        for name, model in refit_base_models.items():
            try:
                test_meta_features_list.append(model.predict(X_test))
                logger.info(f"[{cluster_name}] Generated test meta features for base model: {name}")
            except Exception as e:
                logger.warning(f"[{cluster_name}] Failed test prediction for {name}: {str(e)}")

        if test_meta_features_list:
            test_meta_features = np.column_stack(test_meta_features_list)
            test_preds = meta_model.predict(test_meta_features)
            logger.info(f"[{cluster_name}] Stacked model predictions on test set complete")

            results['test_meta_features'] = test_meta_features
            results['metrics'] = {
                'R²': r2_score(y_test, test_preds),
                'RMSE': np.sqrt(mean_squared_error(y_test, test_preds)),
                'MAE': mean_absolute_error(y_test, test_preds)
            }
            try:
                if structures_all is not None:
                    y_true_all = np.concatenate([y_train, y_test])
                    y_pred_all = np.concatenate([train_preds, test_preds])

                    try:
                        self.visualizer.log_errors_by_threshold(
                            y_true=y_true_all,
                            y_pred=y_pred_all,
                            X=pd.concat([X_train, X_test], axis=0).reset_index(drop=True),
                            structures=structures_all,
                            cluster_name=cluster_name,
                            stage='stacking',
                            model_type='ridge',
                            threshold=1.5
                        )
                    except Exception as e:
                        logger.warning(f"[{cluster_name}-ridge Failed to log large error points in stacking stage: {str(e)}")

                    # plot the meta model parity
                    self.visualizer.plot_parity(
                        y_true=y_true_all,
                        y_pred=y_pred_all,
                        structures=structures_all,
                        train_indices=train_indices,
                        test_indices=test_indices,
                        title=f"{cluster_name} Stacked Model Parity Plot",
                        stage='stacking',
                        model_type='ridge',
                        cluster_name=cluster_name,
                    )

                    self.visualizer._visualize_ensemble_model(
                        ensemble_result={
                            'model': meta_model,
                            'train_predictions': train_preds,
                            'predictions': test_preds,
                            'meta_features': meta_features,
                            'base_model': refit_base_models
                        },
                        X_train=X_train,
                        y_train=y_train,
                        struct_train=struct_train,
                        X_test=X_test,
                        y_test=y_test,
                        struct_test=struct_test,
                        cluster_name=cluster_name
                    )
                else:
                    logger.warning(f"[{cluster_name}] Structures are None, skipping parity plot and ensemble model visualization")
            except Exception as e:
                logger.warning(f"[{cluster_name}] Failed to visualize stacking ensemble model: {str(e)}")
        logger.info(f"[{cluster_name}] Stacking model evaluation on test set complete")
        return results
    

               