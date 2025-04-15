"""
Visualization Module for Oxide Descriptor Analysis

This module provides a Visualizer class for creating plots and visualizations
of model results, correlations, and feature importance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Import from local config
from .config import DEFAULT_CONFIG, COLORS, MARKERS, FIGURES_DIR, ANALYSIS_DIR, PARAM_GRIDS

# Configure logger
logger = logging.getLogger(__name__)

class Visualizer:
    """Class for creating visualizations of model results"""
    
    def __init__(self, config: Dict):
        """
        Initialize the visualizer with configuration
        """
        self.config = config
        self.dpi = config["dpi"]
        self.fig_format = config["fig_formats"]
        self.set_plot_style()
        logger.info("Visualizer initialized")
    
    def set_plot_style(self):
        """Set the default plot style"""
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 18,                       
            'axes.labelsize': 18,       
            'axes.titlesize': 20,       
            'xtick.labelsize': 18,      
            'ytick.labelsize': 18,      
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'text.color': 'black'
        })

    def get_figure_save_path(self, stage=None, 
                             model_type=None, 
                             plot_type=None, cluster_name=None, fmt="png", base_dir=FIGURES_DIR):
        """
        generate a consistent path to save figures based on stage, cluster, model_type, and plot_type
        """
        parts = []

        if stage:
            parts.append(stage.lower())
        if model_type:
            parts.append(model_type.lower())
        if plot_type:
            parts.append(plot_type.lower())
        if cluster_name:
            parts.append(cluster_name.lower())
        
        filename = "_".join(parts) + f".{fmt}"
        path = base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def plot_pearson_correlation(self, data: pd.DataFrame, cluster_name: str, 
                                save_dir: Path = FIGURES_DIR) -> None:
        """
        Plot Pearson correlation heatmap 
        Args:
            data: DataFrame with features
            cluster_name: Name of the cluster for the plot title
            save_dir: Directory to save the plot
        """
        logger.info(f"Creating Pearson correlation heatmap for {cluster_name}")

        target_col = self.config["target_col"]
        physical_feature_prefixes = ['_over_', '_times_', 'diff_', 'inv_']

        # Decide which features to use
        if "physical" in cluster_name.lower():
            constructed_features = [col for col in data.columns 
                                if any(p in col for p in physical_feature_prefixes)]
            original_features = [col for col in self.config.get("feature_cols", []) if col in data.columns]
            feature_cols = constructed_features + original_features
            logger.info(f"Detected 'physical' cluster, using physical + original features: {feature_cols}")
        else:
            feature_cols = [col for col in self.config.get("feature_cols", []) if col in data.columns]
            logger.info(f"Using original features from config: {feature_cols}")
        
        if target_col not in data.columns:
            logger.error(f"Target column '{target_col}' not found in data.")
            return

        cols = [target_col] + feature_cols
        
        # Drop rows with NaN values
        data_clean = data.dropna(subset=cols)
        
        # Calculate correlation matrix
        corr_matrix = data_clean[cols].corr(method='pearson')
        
        # Create correlation heatmap
        plt.figure(figsize=(14, 12))
        #decrease the corelation matrix font size
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5,
                    annot_kws={"size": 10})
        plt.title(f"Pearson Correlation Matrix - {cluster_name}", fontsize=18)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        for fmt in self.config.get("fig_formats", ["png"]):
            save_path = self.get_figure_save_path(
                cluster_name=cluster_name, plot_type='Pearson_correlation', fmt=fmt
                )
            plt.savefig(save_path, dpi=self.dpi, transparent=True)
            logger.info(f"Saved Pearson correlation heatmap to {save_path}")
        plt.close()
        
        # Create scatter plots for key relationships
        self.plot_key_correlations(data_clean, cluster_name, save_dir)

    def plot_key_correlations(self, data: pd.DataFrame, cluster_name: str, save_dir: Path = FIGURES_DIR) -> None:
        """
        Plot all features vs target Pearson correlation as a horizontal bar plot.
        Automatically detects whether to use original or physical (constructed) features.
        """
        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr

        target_col = self.config["target_col"]
        physical_feature_prefixes = ['_over_', '_times_', 'diff_', 'inv_']

        if "physical" in cluster_name.lower():
            # 识别物理特征 + 原始特征
            constructed_features = [col for col in data.columns if any(p in col for p in physical_feature_prefixes)]
            original_features = [col for col in self.config.get("feature_cols", []) if col in data.columns]
            feature_cols = constructed_features + original_features
            logger.info(f"[{cluster_name}] Using physical + original features: {feature_cols}")
        else:
            # 只用 config 中定义的原始特征
            feature_cols = [col for col in self.config.get("feature_cols", []) if col in data.columns]
            logger.info(f"[{cluster_name}] Using original config features: {feature_cols}")

        # filter valid features
        available_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(data[col])]
        data_clean = data.dropna(subset=available_cols + [target_col])

        # calculate Pearson r
        correlations = []
        for col in available_cols:
            try:
                r, p = pearsonr(data_clean[col], data_clean[target_col])
                correlations.append((col, r, p))
            except Exception as e:
                logger.warning(f"correlation failed for {col}: {e}")
        
        # sort and prepare
        correlations = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)
        if not correlations:
            logger.warning("No valid correlations found.")
            return
        
        features = [x[0] for x in correlations]
        r_values = [x[1] for x in correlations]

        # plot
        plt.figure(figsize=(10, len(features) * 0.5 + 2))
        bars = plt.barh(features, r_values, color='steelblue', edgecolor='black')
        plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
        plt.xlabel('Pearson correlation coefficient', fontsize=14)
        plt.title(f"Key Correlations with {target_col} - {cluster_name}", fontsize=16)
        plt.gca().invert_yaxis()
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)

        for bar, r in zip(bars, r_values):
            width = bar.get_width()
            plt.text(width + 0.02 * np.sign(width), bar.get_y() + bar.get_height() / 2,
                    f"{r:.2f}", va='center', ha='left', fontsize=10, color='black')

        # save
        for fmt in self.config.get("fig_formats", ["png"]):
            save_path = self.get_figure_save_path(
                cluster_name=cluster_name, plot_type='feature_target_correlation', fmt=fmt
            )
            plt.savefig(save_path, dpi=self.dpi, transparent=True)
            logger.info(f"Saved feature-target correlation plot to {save_path}")
        plt.close()

    def plot_parity(self, y_true: np.ndarray, y_pred: np.ndarray, 
                   structures: np.ndarray = None, 
                   title: str = "Model Performance", 
                   stage: Optional[str] = None,
                   cluster_name: Optional[str] = None,
                   model_type: Optional[str] = None,
                   train_indices:np.ndarray=None,
                   test_indices:np.ndarray=None,
                   save_path = None) -> None:
        """
        Create parity plot (actual vs predicted)
        Args:
            y_true: True target values
            y_pred: Predicted target values
            structures: Array of structure types for coloring points
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate metrics
        # only calculate test data metrics, and record in the log
        if train_indices is not None and test_indices is not None:
            #r2_train = r2_score(y_true[train_indices], y_pred[train_indices])
            r2_test = r2_score(y_true[test_indices], y_pred[test_indices])
            rmse_test = np.sqrt(mean_squared_error(y_true[test_indices], y_pred[test_indices]))
            mae_test = mean_absolute_error(y_true[test_indices], y_pred[test_indices])
            logger.info(f"Parity plot (Test): R²={r2_test:.4f}, RMSE= {rmse_test:.4f}, MAE={mae_test:.4f})")
        else:
            r2_test = r2_score(y_true, y_pred)
            rmse_test = np.sqrt(mean_squared_error(y_true, y_pred))
            mae_test = mean_absolute_error(y_true, y_pred)
            logger.info(f"Parity plot (Full data): R²={r2_test:.4f}, RMSE= {rmse_test:.4f}, MAE={mae_test:.4f})")
        
        # plot the data 
        structure_colors = COLORS.get(cluster_name, {}) if cluster_name in COLORS else {}
        if structures is not None and train_indices is not None and test_indices is not None:
            for struct in np.unique(structures):
                color = structure_colors.get(struct, '#777777')
                marker = MARKERS.get(struct, 'o')
                structure_mask = structures ==  struct
                train_mask = structure_mask & np.isin(np.arange(len(y_true)), train_indices)
                test_mask = structure_mask & np.isin(np.arange(len(y_true)), test_indices)
                plt.scatter(y_true[train_mask], y_pred[train_mask], 
                            label=f"{struct} (Train)", 
                            alpha=0.5, s=100, color=color, edgecolors='k', marker=marker) 
                            
                plt.scatter(y_true[test_mask], y_pred[test_mask],
                            label=f"{struct} (Test)", 
                            alpha=1.0, s=100, color=color, edgecolors='k', marker=marker)
        else:
            plt.scatter(y_true, y_pred, alpha=0.7, s=100, edgecolors='k')
            logger.info("No structures provided, using default color scheme")

        # Plot perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred)) - 0.5
        max_val = max(np.max(y_true), np.max(y_pred)) + 0.5
        # add a margin to the plot
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
        plt.xlabel("DFT ΔEads (eV)", fontsize=16)
        plt.ylabel("Predicted ΔEads (eV)", fontsize=16)
        plt.title(title, fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        plt.text (0.05, 0.95, 
                    f"R² = {r2_test:.4f}\nRMSE = {rmse_test:.4f} eV\nMAE = {mae_test:.4f} eV",
                    transform=plt.gca().transAxes,
                  fontsize = 14, verticalalignment = 'top',
                  bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.8))
        handles, labels = plt.gca().get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda x: labels[x])
        plt.legend([handles[i] for i in order], 
                   [labels[i] for i in order], 
                   loc='lower right', fontsize=12)

        for fmt in self.fig_format:
            if save_path is None:
                path = self.get_figure_save_path(
                    stage=stage, 
                    cluster_name=cluster_name, 
                    model_type=model_type, 
                    plot_type='parity', 
                    fmt=fmt
                    )
            else:
                path = str(save_path).replace('.png', f'.{fmt}')
            plt.savefig(path, dpi=self.dpi, transparent=True)
            logger.info(f"Saved parity plot to {path}")
        
        plt.close()
    
    def plot_scatter_contour(self, x_data: np.ndarray, y_data: np.ndarray, 
                            z_data: np.ndarray, structures: np.ndarray = None,
                            cluster_name: str = None,
                            xlabel: str = 'X', ylabel: str = 'Y', zlabel: str = 'Z',
                            title: str = 'Contour Plot', save_path: Path = None) -> None:
        """
        Create scatter plot with contour background
        
        Args:
            x_data: X-axis data
            y_data: Y-axis data
            z_data: Z-axis data for contour colors
            structures: Array of structure types
            xlabel, ylabel, zlabel: Axis labels
            title: Plot title
            save_path: Path to save the plot
        """
        from scipy.interpolate import griddata
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(10, 8))
        
        # Calculate overall data range
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        
        # Add margins to ranges
        x_range = (x_max - x_min) * 0.1
        y_range = (y_max - y_min) * 0.1
        
        x_min -= x_range
        x_max += x_range
        y_min -= y_range
        y_max += y_range
        
        # Create grid for interpolation
        x_grid = np.linspace(x_min, x_max, 500)
        y_grid = np.linspace(y_min, y_max, 500)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Use griddata for interpolation
        Z = griddata((x_data, y_data), z_data, (X, Y), method='linear')
        # For cubic interpolation NaN values, fill with linear interpolation
        if np.any(np.isnan(Z)):
            Z_linear = griddata((x_data, y_data), z_data, (X, Y), method='linear')
            mask = np.isnan(Z)
            Z[mask] = Z_linear[mask]
        
        # Plot contour
        z_min, z_max = np.floor(np.min(z_data)), np.ceil(np.max(z_data))
        contour_levels = np.linspace(z_min, z_max, 20)
        contour = plt.contourf(X, Y, Z, levels=contour_levels, cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(contour)
        cbar.set_label(zlabel, fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        
        # Plot data points with structure colors if available
        if structures is not None and cluster_name is not None:
            from .config import COLORS, MARKERS
            cluster_colors = COLORS.get(cluster_name, {})
            for struct in np.unique(structures):
                mask = structures == struct
                color = cluster_colors.get(struct, '#777777')
                structure_marker = MARKERS.get(struct, 'o')
                plt.scatter(x_data[mask], y_data[mask], 
                           label=f'{struct}', 
                           color=color,
                           marker=structure_marker,
                           alpha=0.8, s=80, edgecolors='k')
        else:
            plt.scatter(x_data, y_data, alpha=0.8, s=80, edgecolors='k')
            logger.info("No structures provided, using default color scheme")
        
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.title(title, fontsize=22)
        plt.tight_layout()
        
        if structures is not None:
            plt.legend(fontsize=15, loc='best')
            
        if save_path: 
            base_path = save_path.with_suffix('')
            for fmt in self.config.get("fig_formats", ["png"]):
                full_path = base_path.with_suffix(f'.{fmt}')
                plt.savefig(full_path, dpi=self.config["dpi"], transparent=True)
                logger.info(f"Saved contour plot to {full_path}")
        plt.close()

    def plot_meta_model_coefficients(self, model, feature_names, cluster_name, model_type, stage = None):
        """
        Plot coefficients of a linear meta-model (e.g., Ridge in stacking)
        """
        if not hasattr(model, 'coef_'):
            logger.warning("Meta-model has no coefficients to plot.")
            return

        importances = model.coef_
        plt.figure(figsize=(8, 6))
        plt.barh(feature_names, importances)
        plt.xlabel("Coefficient Weight", fontsize=14)
        plt.title(f"{cluster_name} {model_type} meta-model coefficient", fontsize=16)
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()

        for fmt in self.fig_format:
            save_path = self.get_figure_save_path(
                stage=stage, 
                cluster_name=cluster_name, 
                model_type=model_type, 
                plot_type='meta_model_coefficients', 
                fmt=fmt
                )
            plt.savefig(save_path, dpi=self.dpi, transparent=True)
            logger.info(f"Saved meta-model coefficients plot to {save_path}")

        plt.close()

    def plot_meta_feature_matrix(self, meta_features, base_model_names, y_true, cluster_name, model_type, stage = None, save_path = None):
        """Heatmap of base model predictions (meta-features) vs true values """
        import seaborn as sns
        import pandas as pd
        
        # make sure base_model_names is correct
        if base_model_names is None or len(base_model_names) != meta_features.shape[1]:
            base_model_names = [f"Model {i+1}" for i in range(meta_features.shape[1])]

        df = pd.DataFrame(meta_features, columns=base_model_names)
        df["True"] = y_true

        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title(f"{cluster_name} Meta-feature Correlation Heatmap", fontsize=14)
        plt.tight_layout()

        for fmt in self.fig_format:
            if save_path is None:
                path = self.get_figure_save_path(
                    stage=stage, 
                    cluster_name=cluster_name, 
                    model_type=model_type, 
                    plot_type='meta_feature_matrix', 
                    fmt=fmt
                )
            else:
                path = str(save_path).replace('.png', f'.{fmt}')
            
            plt.savefig(path, dpi=self.dpi, transparent=True)
            logger.info(f"Saved meta-feature matrix plot to {path}")
        plt.close()

    def _scan_and_plot_hyperparam(self, model_class, param_grid, fixed_params,
                                X_train, y_train, X_test, y_test,
                                param_name, title_prefix, save_path):
        """
        Scan one hyperparameter and plot its effect on performance.
        Args:
            model_class: Estimator class (e.g., GradientBoostingRegressor)
            param_grid: List of values for the hyperparameter to scan
            fixed_params: Other fixed hyperparameters
            param_name: Name of hyperparameter to scan (str)
            title_prefix: Title prefix for the plot
            save_path: Where to save the figure
        """
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        results = {'train_r2': [], 'test_r2': [],
                'train_rmse': [], 'test_rmse': [],
                'train_mae': [], 'test_mae': []}

        for val in param_grid:
            params = fixed_params.copy()
            params[param_name] = val
            model = model_class(**params)
            
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.squeeze()
            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.squeeze()  

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            results['train_r2'].append(r2_score(y_train, y_train_pred))
            results['test_r2'].append(r2_score(y_test, y_test_pred))
            results['train_rmse'].append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
            results['test_rmse'].append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
            results['train_mae'].append(mean_absolute_error(y_train, y_train_pred))
            results['test_mae'].append(mean_absolute_error(y_test, y_test_pred))

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(param_grid, results['train_rmse'], 'b-', label='RMSE (train)')
        plt.plot(param_grid, results['test_rmse'], 'b--', label='RMSE (test)')
        plt.plot(param_grid, results['train_mae'], 'g-', label='MAE (train)')
        plt.plot(param_grid, results['test_mae'], 'g--', label='MAE (test)')

        ax2 = plt.gca().twinx()
        ax2.plot(param_grid, results['train_r2'], 'r-', label='R² (train)')
        ax2.plot(param_grid, results['test_r2'], 'r--', label='R² (test)')
        ax2.set_ylabel('R²', fontsize=12)

        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

        plt.xlabel(param_name, fontsize=12)
        plt.ylabel('RMSE / MAE (eV)', fontsize=12)
        plt.title(f"{title_prefix}: {param_name}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        base_path = save_path.with_suffix('')
        for fmt in self.config.get("fig_formats", ["png"]):
            plt.savefig(f"{base_path}.{fmt}", dpi=self.config["dpi"], transparent=True)
        
        #plt.savefig(save_path, dpi=self.config['dpi'], transparent=True)
        plt.close()
        logger.info(f"Saved hyperparameter tuning plot: {save_path}")

    def visualize_hyperparameter_tuning(self, X_train, y_train, X_test, y_test,
                                    model_type='rf', cluster_name=None,stage=None):
        """
        Unified hyperparameter tuning visualizer for RF, GB, XGB.
        """
        logger.info(f"Hyperparameter tuning for {model_type.upper()}")
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from xgboost import XGBRegressor  
        
        model_map = {
        'rf': RandomForestRegressor,
        'gb': GradientBoostingRegressor,
        'xgb': XGBRegressor
        }

        param_grid = PARAM_GRIDS.get(model_type)
        if param_grid is None:
            logger.warning(f"Unsupported hyperparameter grid for: {model_type}")
            return
        model_class = model_map.get(model_type)
        candidate_params = {
            'random_state': self.config.get('random_state', 42),
            'n_jobs': self.config.get('n_jobs', -1)
        }
        legal_keys = model_class().get_params().keys()
        fixed_params = {k: v for k, v in candidate_params.items() if k in legal_keys}
        title_prefix = f"{model_type.upper()} - {cluster_name}"
        for param_name, param_sets in param_grid.items():
            for fmt in self.config.get("fig_formats", ["png"]):
                save_path = self.get_figure_save_path(
                    stage=stage, 
                    cluster_name=cluster_name, 
                    model_type=model_type, 
                    plot_type=f"{param_name} hyperparameter_tuning", 
                    fmt=fmt
                )
            self._scan_and_plot_hyperparam(
                model_class=model_map[model_type],
                param_grid=param_sets,
                fixed_params=fixed_params,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                param_name=param_name,
                title_prefix=title_prefix,
                save_path=save_path
            )

    def visualize_bayesian_optimization(self, optimization_results, model_type='rf', 
                                        cluster_name=None, stage=None, save_path=None):
        """
        Visualize the results of Bayesian optimization
        Args:
            optimization_results: Results from Bayesian optimization
            model_type: Type of model ('rf', 'gb', 'xgb')
            cluster_name: Name of the cluster for the plot title
            save_path: Path to save the figure
        """
        from skopt.plots import plot_convergence
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        # check if opt_results is a dictionary of results by model type
        if optimization_results is None:
            logger.warning("No optimization results available to plot.")
            return
        
        if save_path is None:
            save_path = self.get_figure_save_path(
                stage=stage,
                cluster_name=cluster_name, 
                model_type=model_type, 
                plot_type='bayesian_optimization', 
                fmt=None
            )
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_convergence(optimization_results,ax=ax)
            ax.set_title(f"{model_type.upper()} Bayesian Optimization Convergence for {cluster_name}", fontsize=16)
            plt.tight_layout()
            for fmt in self.config.get("fig_formats", ["png", "svg"]):
                path = Path(save_path).with_suffix(f".{fmt}")
                plt.savefig(path, dpi=self.config["dpi"], transparent=True)
                logger.info(f"Saved Bayesian optimization plot to {save_path}_convergence.{fmt}")
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot Bayesian optimization results: {e}")

    def visualize_meta_model_hyperparameters(self, cv_results, param_name, param_values, 
                                            title, cluster_name, save_path):
        """
        Visualize meta-model hyperparameter tuning results
        
        Args:
            cv_results: Cross-validation results from GridSearchCV
            param_name: Name of the parameter being tuned (e.g., 'alpha')
            param_values: List of parameter values tested
            title: Plot title
            cluster_name: Name of the cluster
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(10, 6))
        
        # try to get the parameter key from cv_results
        param_key = f'param_{param_name}'
        
        if cv_results is not None and isinstance(cv_results, dict):
            # check if all the required keys are present
            if 'mean_test_score' in cv_results and 'mean_train_score' in cv_results:
                test_scores = cv_results['mean_test_score']
                train_scores = cv_results.get('mean_train_score', [])
                
                # 如果param_key存在，使用它来排序
                if param_key in cv_results:
                    # 将参数值转换为数值，并获取排序索引
                    param_values = [float(val) for val in cv_results[param_key]]
                    sorted_indices = np.argsort(param_values)
                    
                    param_values = np.array(param_values)[sorted_indices]
                    test_scores = np.array(test_scores)[sorted_indices]
                    
                    if len(train_scores) > 0:
                        train_scores = np.array(train_scores)[sorted_indices]
                
                # 绘制测试集和训练集得分
                plt.plot(param_values, test_scores, 'o-', color='red', label='Validation score')
                
                if len(train_scores) > 0:
                    plt.plot(param_values, train_scores, 'o-', color='blue', label='Training score')
                
                # 标记最佳点
                best_idx = np.argmax(test_scores)
                best_param = param_values[best_idx]
                best_score = test_scores[best_idx]
                
                plt.scatter([best_param], [best_score], color='gold', edgecolors='black', 
                        s=100, zorder=5, label=f'Best: {best_param:.4f}')
                
                # 添加注释
                plt.annotate(f'Best value: {best_param:.4f}\nScore: {best_score:.4f}',
                            xy=(best_param, best_score), xytext=(0, 20),
                            textcoords='offset points', ha='center',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            else:
                # 如果缺少所需键，使用提供的param_values
                plt.text(0.5, 0.5, "Insufficient CV results data", 
                        ha='center', va='center', transform=plt.gca().transAxes)
        else:
            # 如果cv_results无效，使用提供的param_values
            plt.text(0.5, 0.5, "No CV results available", 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # 设置图表标签和标题
        plt.xlabel(f'{param_name} value', fontsize=12)
        plt.ylabel('Score (R²)', fontsize=12)
        plt.title(f'{title} - {cluster_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 保存图表
        if save_path:
            base_path = save_path.with_suffix('')
            for fmt in self.config.get("fig_formats", ["png"]):
                full_path = f"{base_path}.{fmt}"
                plt.savefig(full_path, dpi=self.config["dpi"], transparent=True)
                logger.info(f"Saved meta-model hyperparameter tuning plot to {full_path}")
        
        plt.close()

    def _visualize_ensemble_model(self, ensemble_result, X_train, y_train, struct_train,
                                X_test, y_test, struct_test, cluster_name):
        """
        Visualize the ensemble model predictions and feature importances
        """
        if 'model' not in ensemble_result:
            logger.error("No model found for visualization.")
            return
        ensemble_model = ensemble_result['model']

        # meta model coefficients visualization
        if ensemble_model is not None and hasattr(ensemble_model, 'coef_'):
            base_models = ensemble_result.get('base_model', {})
            base_model_names = list(base_models.keys()) if base_models else \
                [f"Model {i+1}" for i in range(len(ensemble_model.coef_))]
            self.plot_meta_model_coefficients(
                model=ensemble_model,
                feature_names=base_model_names,
                cluster_name=cluster_name,
                stage='stacking',
                model_type='ridge'
            )
        # meta feature matrix visualization
        if "meta_features" in ensemble_result:
            base_models = ensemble_result.get('base_model', {})
            base_model_names = list(base_models.keys()) if base_models else \
                [f"Model {i+1}" for i in range(len(ensemble_model["meta_features"].shape[1]))]
            base_path = FIGURES_DIR / f"{cluster_name.lower()}_meta_feature_heatmap"
            for fmt in self.config.get("fig_formats", ["png"]):
                save_path = base_path.with_suffix(f'.{fmt}')
                self.plot_meta_feature_matrix(
                    meta_features=ensemble_result["meta_features"],
                    base_model_names=base_model_names,
                    y_true=y_train,
                    model_type='ridge',
                    stage='stacking',
                    cluster_name=cluster_name,  
                    save_path=save_path
                )


    def create_comparison_visualizations(self, pt07_data: pd.DataFrame, pt13_data: pd.DataFrame, combined_data: pd.DataFrame,
                                        pt07_results: Dict, pt13_results: Dict, combined_results: Dict) -> None:
        """
        Create visualizations comparing different aspects of the analysis
        Args:
            pt07_data, pt13_data: Raw data DataFrames
            pt07_results, pt13_results: Results dictionaries
            combined_results: Combined analysis results
        """
        # 1. RMSD vs Charge with ΔEads contour plot
        if all(col in pt07_data.columns for col in ['RMSD', 'Δq', 'ΔEads']) and \
           all(col in pt13_data.columns for col in ['RMSD', 'Δq', 'ΔEads']) and \
           all(col in combined_data.columns for col in ['RMSD', 'Δq', 'ΔEads']):
            
            # Extract data
            pt07_rmsd = pt07_data['RMSD'].values
            pt07_charge = pt07_data['Δq'].values
            pt07_eads = pt07_data['ΔEads'].values
            pt07_structs = pt07_data['structure'].values
            
            pt13_rmsd = pt13_data['RMSD'].values
            pt13_charge = pt13_data['Δq'].values
            pt13_eads = pt13_data['ΔEads'].values
            pt13_structs = pt13_data['structure'].values

            combined_rmsd = combined_data['RMSD'].values
            combined_charge = combined_data['Δq'].values
            combined_eads = combined_data['ΔEads'].values
            combined_structs = combined_data['structure'].values
            
            # Combine data for contour plot
            all_rmsd = np.concatenate([pt07_rmsd, pt13_rmsd, combined_rmsd])
            all_charge = np.concatenate([pt07_charge, pt13_charge, combined_charge])
            all_eads = np.concatenate([pt07_eads, pt13_eads, combined_eads])
            
            # Create custom structure labels with cluster prefix
            pt07_structs_labeled = np.array([f'Pt07-{s}' for s in pt07_structs])
            pt13_structs_labeled = np.array([f'Pt13-{s}' for s in pt13_structs])
            combined_structs_labeled = np.array([f'combined-{s}' for s in combined_structs])
            all_structs = np.concatenate([pt07_structs_labeled, pt13_structs_labeled, combined_structs_labeled])
            
            # Create contour plot
            save_path = self.get_figure_save_path(
             cluster_name='combined',
            stage=None,
            model_type=None,
            plot_type='contour_summary',
            fmt='png'
            )
            # combined
            self.plot_scatter_contour(
                x_data=all_charge,
                y_data=all_rmsd,
                z_data=all_eads,
                structures=all_structs,
                cluster_name='combined',
                xlabel='Δq (e-)',
                ylabel='RMSD (Å)',
                zlabel='ΔEads (eV)',
                title='Δq vs RMSD with ΔEads (Combined)',
                save_path=save_path
                )
            # Pt07
            self.plot_scatter_contour(
                x_data=pt07_charge,
                y_data=pt07_rmsd,
                z_data=pt07_eads,
                structures=pt07_structs,
                cluster_name='Pt07',
                xlabel='Δq (e-)',
                ylabel='RMSD (Å)',
                zlabel='ΔEads (eV)',
                title='Δq vs RMSD with ΔEads (Pt07)',
                save_path=self.get_figure_save_path(
                    cluster_name='Pt07',
                    stage='summary',
                    model_type=None,
                    plot_type='contour',
                    fmt='png'
                )
            )
            # Pt13
            self.plot_scatter_contour(
                x_data=pt13_charge,
                y_data=pt13_rmsd,
                z_data=pt13_eads,
                structures=pt13_structs,
                cluster_name='Pt13',
                xlabel='Δq (e-)',
                ylabel='RMSD (Å)',
                zlabel='ΔEads (eV)',
                title='Δq vs RMSD with ΔEads (Pt13)',
                save_path=self.get_figure_save_path(
                    cluster_name='Pt13',
                    stage='summary',
                    model_type=None,
                    plot_type='contour',
                    fmt='png'
                )
            )
            logger.info("Contour plots created for RMSD vs Charge with ΔEads")

        # 2. Model performance comparison across clusters
        #check if all the metrics are present
        if 'metrics' in pt07_results and 'metrics' in pt13_results and 'metrics' in combined_results:
            # Initialize lists to collect data for the DataFrame
            models = []
            datasets = []
            r2_values = []
            rmse_values = []
            
            # Get PT07 metrics if available
            if 'Random Forest' in pt07_results['metrics']:
                models.append('Random Forest')
                datasets.append('Pt07')
                r2_values.append(pt07_results['metrics']['Random Forest'].get('R²', np.nan))
                rmse_values.append(pt07_results['metrics']['Random Forest'].get('RMSE', np.nan))
            
            if 'Gradient Boosting' in pt07_results['metrics']:
                models.append('Gradient Boosting')
                datasets.append('Pt07')
                r2_values.append(pt07_results['metrics']['Gradient Boosting'].get('R²', np.nan))
                rmse_values.append(pt07_results['metrics']['Gradient Boosting'].get('RMSE', np.nan))
            
            if 'Domain-Aware Ensemble' in pt07_results['metrics']:
                models.append('Domain-Aware Ensemble')
                datasets.append('Pt07')
                r2_values.append(pt07_results['metrics']['Domain-Aware Ensemble'].get('R²', np.nan))
                rmse_values.append(pt07_results['metrics']['Domain-Aware Ensemble'].get('RMSE', np.nan))
            
            # Get PT13 metrics if available
            if 'Random Forest' in pt13_results['metrics']:
                models.append('Random Forest')
                datasets.append('Pt13')
                r2_values.append(pt13_results['metrics']['Random Forest'].get('R²', np.nan))
                rmse_values.append(pt13_results['metrics']['Random Forest'].get('RMSE', np.nan))
            
            if 'Gradient Boosting' in pt13_results['metrics']:
                models.append('Gradient Boosting')
                datasets.append('Pt13')
                r2_values.append(pt13_results['metrics']['Gradient Boosting'].get('R²', np.nan))
                rmse_values.append(pt13_results['metrics']['Gradient Boosting'].get('RMSE', np.nan))
            
            if 'Domain-Aware Ensemble' in pt13_results['metrics']:
                models.append('Domain-Aware Ensemble')
                datasets.append('Pt13')
                r2_values.append(pt13_results['metrics']['Domain-Aware Ensemble'].get('R²', np.nan))
                rmse_values.append(pt13_results['metrics']['Domain-Aware Ensemble'].get('RMSE', np.nan))
            
            # Get Combined metrics if available
            if 'Random Forest' in combined_results['metrics']:
                models.append('Random Forest')
                datasets.append('combined')
                r2_values.append(combined_results['metrics']['Random Forest'].get('R²', np.nan))
                rmse_values.append(combined_results['metrics']['Random Forest'].get('RMSE', np.nan))
            
            if 'Gradient Boosting' in combined_results['metrics']:
                models.append('Gradient Boosting')
                datasets.append('combined')
                r2_values.append(combined_results['metrics']['Gradient Boosting'].get('R²', np.nan))
                rmse_values.append(combined_results['metrics']['Gradient Boosting'].get('RMSE', np.nan))
            
            # Create DataFrame only with available metrics
            if models:
                metrics_df = pd.DataFrame({
                    'Model': models,
                    'Dataset': datasets,
                    'R²': r2_values,
                    'RMSE': rmse_values
                })
                
                # Remove any rows with NaN values
                metrics_df = metrics_df.dropna()
                
                if not metrics_df.empty:
                    # Plot R² comparison
                    plt.figure(figsize=(12, 6))
                    sns.barplot(x='Model', y='R²', hue='Dataset', data=metrics_df)
                    plt.title('Model Performance Comparison (R²)', fontsize=16)
                    plt.xlabel('Model', fontsize=14)
                    plt.ylabel('R² Score', fontsize=14)
                    plt.ylim(0, 1)
                    plt.legend(title='Dataset')
                    plt.tight_layout()
                    base_path = FIGURES_DIR / f"model_r2_comparison"
                    for fmt in self.config.get("fig_formats", ["png"]):
                        full_path = base_path.with_suffix(f'.{fmt}')
                        plt.savefig(full_path, dpi=self.config["dpi"], transparent=True)
                        logger.info(f"Saved R² comparison plot to {full_path}")
                    plt.close()
                    
                    # Plot RMSE comparison
                    plt.figure(figsize=(12, 6))
                    sns.barplot(x='Model', y='RMSE', hue='Dataset', data=metrics_df)
                    plt.title('Model Performance Comparison (RMSE)', fontsize=16)
                    plt.xlabel('Model', fontsize=14)
                    plt.ylabel('RMSE (eV)', fontsize=14)
                    plt.legend(title='Dataset')
                    plt.tight_layout()
                    base_path= FIGURES_DIR / f"model_rmse_comparison"
                    for fmt in self.config.get("fig_formats", ["png"]):
                        full_path = base_path.with_suffix(f'.{fmt}')
                        plt.savefig(full_path, dpi=self.config["dpi"], transparent=True)
                        logger.info(f"Saved RMSE comparison plot to {full_path}")
                    plt.close()
                else:
                    logger.warning("No valid metrics data available for visualization after filtering NaN values")
            else:
                logger.warning("No metrics data available for visualization")
        else:
            logger.warning("Missing metrics dictionaries for comparison visualization")

    def plot_feature_importance(self, model, feature_names: List[str], cluster_name: str, model_type: str,
                                stage: Optional[str] = None, title: str = "Feature Importance", save_path: Optional[str] = None
                               ) -> None:
        """
        Plot feature importance for a model  
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            title: Plot title
            save_path: Path to save the plot
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {type(model).__name__} does not have feature_importances_ attribute")
            return
            
        importances = model.feature_importances_

        # make sure the length of feature names matches the length of importances
        if len(feature_names) != len(importances):
            logger.warning(f"Feature names length ({len(feature_names)}) does not match "
                     f"importances length ({len(importances)})")
            feature_names = feature_names[:len(importances)]


        # Sort features by importance
        indices = np.argsort(importances)[::-1]

        #plot
        plt.figure(figsize=(12, 8))
        plt.title(f"{cluster_name} {model_type} feature importance", fontsize=16)
        plt.bar(range(len(feature_names)), importances[indices], align='center')
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.tight_layout()
        
        for fmt in self.fig_format:
            path = self.get_figure_save_path(
                stage=stage,
                cluster_name=cluster_name,
                model_type=model_type,
                plot_type='feature_importance',
                fmt=fmt
            )
            plt.savefig(path, dpi=self.dpi, transparent=True)
            logger.info(f"Saved feature importance plot to {path}") 
        plt.close()



 