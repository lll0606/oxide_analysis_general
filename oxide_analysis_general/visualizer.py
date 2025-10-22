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
from .config import DEFAULT_CONFIG, COLORS, MARKERS, FIGURES_DIR, ANALYSIS_DIR, PARAM_GRIDS, FEATURE_NAME_MAP

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
    
    def _map_feature_names(self, feature_list: List[str]) -> List[str]:
        """Map raw and constructed feature names to standardized display names."""
        name_map = self.config.get("FEATURE_NAME_MAP", {})
        mapped_names = []
        for feat in feature_list:
            feat_fixed = feat.replace("-", "_") 
            if feat_fixed.startswith("inv_"):
                base_feat = feat_fixed[4:]
                mapped_base = name_map.get(base_feat, base_feat)
                mapped_feat = f"Inverse {mapped_base}"
            elif '_x_' in feat_fixed:
                parts = [name_map.get(p.strip(), p.strip())
                        for p in feat_fixed.split('_x_')]
                mapped_feat = ' x '.join(parts)
            # elif '_x_' in feat_fixed:
            #     sep = '_x_' if '_x_' in feat_fixed else 'x'
            #     parts = feat_fixed.split(sep)
            #     mapped_parts = []
            #     for part in parts:
            #         part = part.strip()
            #         mapped_part = name_map.get(part, part)
            #         mapped_parts.append(mapped_part)
            #     mapped_feat = " x ".join(mapped_parts) 
            elif '/' in feat_fixed:
                parts = feat_fixed.split('/')
                mapped_parts = []
                for part in parts:
                    part = part.strip()
                    mapped_part = name_map.get(part, part)
                    mapped_parts.append(mapped_part)
                mapped_feat = " / ".join(mapped_parts)
            elif '-' in feat_fixed and not feat_fixed.startswith('-'):
                parts = feat_fixed.split('-')
                mapped_parts = []
                for part in parts:
                    part = part.strip()
                    mapped_part = name_map.get(part, part)
                    mapped_parts.append(mapped_part)
                mapped_feat = " - ".join(mapped_parts)
            elif '+' in feat_fixed:
                parts = feat_fixed.split('+')
                mapped_parts = []
                for part in parts:
                    part = part.strip()
                    mapped_part = name_map.get(part, part)
                    mapped_parts.append(mapped_part)
                mapped_feat = " + ".join(mapped_parts)

            else:
                mapped_feat = name_map.get(feat_fixed, feat_fixed)
            mapped_names.append(mapped_feat)
        return mapped_names

    def set_plot_style(self):
        """Set unified matplotlib + seaborn plot style for all figures"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Base style
        plt.style.use('default')

        # Matplotlib rcParams
        plt.rcParams.update({
            'svg.fonttype': 'none',         # Make SVG text editable
            'text.usetex': False,           # Avoid requiring LaTeX installation
            'font.family': 'Arial',         # Use Arial everywhere
            'font.size': 18,
            'axes.labelsize': 18,
            'axes.titlesize': 20,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'text.color': 'black',
            'figure.dpi': self.dpi          # Use configured dpi for consistency
        })

        # Seaborn context
        sns.set_context("notebook")         # Options: paper, notebook, talk, poster
        
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

    def plot_pearson_correlation(self, data: pd.DataFrame, cluster_name: str, stage: str = 'stage1',
                                save_dir: Path = FIGURES_DIR) -> None:
        """
        Plot Pearson correlation heatmap for specified features (Stage 1) 
        or physics-informed combined features (Stage 2).
        """
        logger.info(f"Creating Pearson correlation heatmap for {cluster_name}, {stage}")

        target_col = self.config["target_col"]
        feature_cols = self.config["feature_cols"]

        if target_col not in data.columns:
            logger.error(f"Target column '{target_col}' not found in data.")
            return
        
        candidate_features = [col for col in data.columns if col != target_col]
        if set(feature_cols).issubset(set(candidate_features)):
            feature_cols = feature_cols
        else:
            feature_cols = [col for col in candidate_features if any(base in col for base in feature_cols)]

        if stage == 'stage1':
            # Stage 1: use original top features only
            feature_cols = [
                col for col in self.config["feature_cols"]
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col])
            ]
        elif stage == 'stage2':
            # Stage 2: use top features and new combined features (explicitly consider combination rules in visualizer)
            combination_symbols = ['_x_', '/', '-']
            top_features = [
                col for col in self.config["feature_cols"]
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col])
            ]
            new_combined_features = [
                col for col in data.columns 
                if col != target_col and pd.api.types.is_numeric_dtype(data[col])
                and any(sym in col for sym in combination_symbols)
            ]
            feature_cols = top_features + new_combined_features
        else:
            logger.error(f"Invalid stage '{stage}' specified. Choose 'stage1' or 'stage2'.")
            return

        if not feature_cols:
            logger.error("No valid numeric features available for plotting.")
            return

        data_clean = data.dropna(subset=[target_col] + feature_cols)
        mapped_feature_cols = self._map_feature_names(feature_cols)
        mapped_target_col = self._map_feature_names([target_col])[0]

        data_corr = data_clean[[target_col] + feature_cols].copy()
        data_corr.columns = [mapped_target_col] + mapped_feature_cols
        corr_matrix = data_corr.corr(method='pearson')

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        figsize_base = 10
        num_features = len(feature_cols)
        dynamic_size = figsize_base + num_features * 0.5
        fig, ax = plt.subplots(figsize=(dynamic_size, dynamic_size))

        cmap = plt.cm.viridis
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            mask=mask, 
            vmin=-1,
            vmax=1,
            cmap=cmap,
            square=True,
            fmt=".2f", 
            linewidths=0.5,
            annot_kws={"size": 10},
            ax=ax)

        plt.title(f"Pearson Correlation Matrix - {cluster_name} ({stage})", fontsize=18)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        for fmt in self.config.get("fig_formats", ["png"]):
            save_path = self.get_figure_save_path(
                cluster_name=cluster_name, plot_type=f'Pearson_correlation_{stage}', fmt=fmt
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
        logger.info(f"Creating key correlations plot for {cluster_name}")  

        target_col = self.config["target_col"]
        allowed_cols = self.config["feature_cols"]
        feature_cols = [col for col in data.columns if col in allowed_cols and col != target_col]

        data_clean = data.dropna(subset = feature_cols + [self.config["target_col"]])
        available_cols = feature_cols
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

        name_map = self.config.get("FEATURE_NAME_MAP", {})
        mapped_features = self._map_feature_names(features)

        # plot
        plt.figure(figsize=(10, len(features) * 0.5 + 2))
        bars = plt.barh(mapped_features, r_values, color='steelblue', edgecolor='black')

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
                            alpha=0.5, s=240, color=color, edgecolors='k', marker=marker) 
                            
                plt.scatter(y_true[test_mask], y_pred[test_mask],
                            label=f"{struct} (Test)", 
                            alpha=1.0, s=240, color=color, edgecolors='k', marker=marker)
        else:
            plt.scatter(y_true, y_pred, alpha=0.7, s=240, edgecolors='k')
            logger.info("No structures provided, using default color scheme")

        # Plot perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred)) - 0.5
        max_val = max(np.max(y_true), np.max(y_pred)) + 0.5
        # add a margin to the plot
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
        target_label = self.config.get("target_col", "target")
        mapped_target = self.config.get("FEATURE_NAME_MAP", {}).get(target_label, target_label)
        plt.xlabel(f"DFT {mapped_target} (eV)", fontsize=24)
        plt.ylabel(f"Predicted {mapped_target} (eV)", fontsize=24)
        plt.title(title, fontsize=24)
        plt.tick_params(axis='both', labelsize=20)  
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
                        z_data: np.ndarray,
                        clusters: np.ndarray,
                        structures: np.ndarray,
                        xlabel: str = 'X', ylabel: str = 'Y', zlabel: str = 'Z',
                        title: str = 'Contour Plot', save_path: Path = None) -> None:
        from scipy.interpolate import griddata
        import matplotlib.pyplot as plt
        import numpy as np
        # Set Matplotlib parameters
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'Arial'
        # Set Seaborn context
        sns.set_context("notebook")

        # 固定颜色与 marker 映射（仅此改动）
        COLOR_BY_CLUSTER = {
            'Pt07': '#1f77b4',
            'Pt13': '#7dc0a7ff'
        }

        MARKER_BY_STRUCTURE = {
            'flat': 's',
            'rod': '*',
            '3d': 'o',
            'bj': 'D',
            'quasi': '^',
            '2l': 'p'
        }

        plt.figure(figsize=(12, 8))

        # generate grid data for contour
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        x_grid = np.linspace(x_min - x_margin, x_max + x_margin, 500)
        y_grid = np.linspace(y_min - y_margin, y_max + y_margin, 500)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Interpolate Z values
        Z = griddata((x_data, y_data), z_data, (X, Y), method='linear')

        # Draw contour background
        z_min, z_max = np.floor(np.min(z_data)), np.ceil(np.max(z_data))
        contour_levels = np.linspace(z_min, z_max, 20)
        contour = plt.contourf(X, Y, Z, levels=contour_levels, cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(contour)
        cbar.set_label(zlabel, fontsize=18)
        cbar.ax.tick_params(labelsize=18)

        # only change this part: color = cluster, marker = structure
        unique_pairs = sorted(set(zip(clusters, structures)))
        for cluster, structure in unique_pairs:
            mask = (clusters == cluster) & (structures == structure)
            label = f"{structure} ({cluster})"
            color = COLOR_BY_CLUSTER.get(cluster, '#777777')
            marker = MARKER_BY_STRUCTURE.get(structure, 'o')
            plt.scatter(x_data[mask], y_data[mask],
                        label=label,
                        color=color,
                        marker=marker,
                        alpha=0.8, s=80, edgecolors='k')

        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.title(title, fontsize=22)
        plt.legend(fontsize=14, loc='best')
        plt.tight_layout()

        if save_path:
            base_path = save_path.with_suffix('')
            for fmt in self.config.get("fig_formats", ["png"]):
                full_path = base_path.with_suffix(f'.{fmt}')
                plt.savefig(full_path, dpi=self.config.get("dpi", 300), transparent=True)
                logger.info(f"Saved contour plot to {full_path}")
        plt.close()
        
    def create_comparison_visualizations(self, pt07_data: pd.DataFrame = pd.DataFrame(), 
                                        pt13_data: pd.DataFrame = pd.DataFrame(), 
                                        combined_data: pd.DataFrame = pd.DataFrame(),
                                        pt07_results: Dict = {}, 
                                        pt13_results: Dict = {}, 
                                        combined_results: Dict = {}) -> None:
        active_clusters = []
        if not pt07_data.empty and pt07_results:
            active_clusters.append('Pt07')
        if not pt13_data.empty and pt13_results:
            active_clusters.append('Pt13')
        if not combined_data.empty and combined_results:
            active_clusters.append('combined')

        if not active_clusters:
            logger.warning("No valid data available for comparison visualization")
            return

        data_map = {
            'Pt07': pt07_data,
            'Pt13': pt13_data,
            'combined': combined_data
        }

        results_map = {
            'Pt07': pt07_results,
            'Pt13': pt13_results,
            'combined': combined_results
        }

        valid_contour_clusters = []
        for cluster in active_clusters:
            data = data_map[cluster]
            if all(col in data.columns for col in ['RMSD', 'Δq', 'ΔEads', 'structure', 'cluster']):
                valid_contour_clusters.append(cluster)

        if valid_contour_clusters:
            for cluster in valid_contour_clusters:
                data = data_map[cluster]
                self.plot_scatter_contour(
                    x_data=data['Δq'].values,
                    y_data=data['RMSD'].values,
                    z_data=data['ΔEads'].values,
                    clusters=data['cluster'].values,
                    structures=data['structure'].values,
                    xlabel='Δq (e-)',
                    ylabel='RMSD (Å)',
                    zlabel='ΔEads (eV)',
                    title=f'Δq vs RMSD with ΔEads ({cluster})',
                    save_path=self.get_figure_save_path(
                        cluster_name=cluster,
                        stage='summary',
                        model_type=None,
                        plot_type='contour',
                        fmt='png'
                    )
                )
                logger.info(f"Contour plot created for {cluster}")

            if len(valid_contour_clusters) > 1:
                all_charge, all_rmsd, all_eads, all_structs, all_clusters = [], [], [], [], []
                for cluster in valid_contour_clusters:
                    data = data_map[cluster]
                    all_charge.append(data['Δq'].values)
                    all_rmsd.append(data['RMSD'].values)
                    all_eads.append(data['ΔEads'].values)
                    all_structs.append(data['structure'].values)
                    all_clusters.append(data['cluster'].values)

                self.plot_scatter_contour(
                    x_data=np.concatenate(all_charge),
                    y_data=np.concatenate(all_rmsd),
                    z_data=np.concatenate(all_eads),
                    clusters=np.concatenate(all_clusters),
                    structures=np.concatenate(all_structs),
                    xlabel='Δq (e-)',
                    ylabel='RMSD (Å)',
                    zlabel='ΔEads (eV)',
                    title='Δq vs RMSD with ΔEads (Combined Comparison)',
                    save_path=self.get_figure_save_path(
                        cluster_name='combined_comparison',
                        stage=None,
                        model_type=None,
                        plot_type='contour_summary',
                        fmt='png'
                    )
                )
                logger.info("Combined contour plot created for comparison")

    def plot_meta_model_coefficients(self, model, feature_names, cluster_name, model_type, stage = None):
        """
        Plot coefficients of a linear meta-model (e.g., Ridge in stacking)
        """
        if not hasattr(model, 'coef_'):
            logger.warning("Meta-model has no coefficients to plot.")
            return

        importances = model.coef_
        plt.figure(figsize=(8, 6))
        mapped_names = self._map_feature_names(feature_names)
        plt.barh(mapped_names, importances)
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
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="viridis", cbar=True)
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
            save_path = self.get_figure_save_path(
                stage=stage, 
                cluster_name=cluster_name, 
                model_type=model_type, 
                plot_type=f"{param_name} hyperparameter_tuning", 
                fmt=self.config.get("fig_formats", ["png"])[0]
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
        """
        plt.figure(figsize=(10, 6))
        
        # try to get the parameter key from cv_results
        param_key = f'param_{param_name}'
        
        if cv_results is not None and isinstance(cv_results, dict):
            # check if all the required keys are present
            if 'mean_test_score' in cv_results and 'mean_train_score' in cv_results:
                test_scores = cv_results['mean_test_score']
                train_scores = cv_results.get('mean_train_score', [])
                
                # if param_key exists, use it to order the values
                if param_key in cv_results:
                    # change values to float for sorting
                    param_values = [float(val) for val in cv_results[param_key]]
                    sorted_indices = np.argsort(param_values)
                    
                    param_values = np.array(param_values)[sorted_indices]
                    test_scores = np.array(test_scores)[sorted_indices]
                    
                    if len(train_scores) > 0:
                        train_scores = np.array(train_scores)[sorted_indices]

                # plot the score of test and training sets
                plt.plot(param_values, test_scores, 'o-', color='red', label='Validation score')
                
                if len(train_scores) > 0:
                    plt.plot(param_values, train_scores, 'o-', color='blue', label='Training score')
                
                # mark the best point
                best_idx = np.argmax(test_scores)
                best_param = param_values[best_idx]
                best_score = test_scores[best_idx]
                
                plt.scatter([best_param], [best_score], color='gold', edgecolors='black', 
                        s=100, zorder=5, label=f'Best: {best_param:.4f}')
                
                # annotate the best point
                plt.annotate(f'Best value: {best_param:.4f}\nScore: {best_score:.4f}',
                            xy=(best_param, best_score), xytext=(0, 20),
                            textcoords='offset points', ha='center',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            else:
                # if required keys are missing, use provided param_values
                plt.text(0.5, 0.5, "Insufficient CV results data", 
                        ha='center', va='center', transform=plt.gca().transAxes)
        else:
            # if cv_results is invalid, use provided param_values
            plt.text(0.5, 0.5, "No CV results available", 
                    ha='center', va='center', transform=plt.gca().transAxes)

        # set chart labels and title
        plt.xlabel(f'{param_name} value', fontsize=12)
        plt.ylabel('Score (R²)', fontsize=12)
        plt.title(f'{title} - {cluster_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # save the figure
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

    def infer_group_from_feature_name(self, feature_name: str) -> str:
        """
        Infer the group (Pt cluster, interface, CeOx) of a feature name using the config FEATURE_GROUPS.
        Used when no group_assignment is explicitly provided.
        """

        feature_groups = self.config.get("FEATURE_GROUPS", {})
        logger.debug(f"Available feature groups: {feature_groups}")

        flat_map = {}
        for group, feats in feature_groups.items():
            for feat in feats:
                flat_map[feat] = group
                flat_map[feat.replace("-", "_")] = group

        logger.debug(f"Flat map of features to groups: {flat_map}")
        
        if feature_name in flat_map:
            return flat_map[feature_name]
        
        feature_normalized = feature_name.replace("-", "_")
        if feature_normalized in flat_map:
            return flat_map[feature_normalized]
        
        if feature_normalized.startswith("inv_"):
            best_feat = feature_normalized[4:]
            if best_feat in flat_map:
                return flat_map[best_feat]
            
        for sep in ['_x_', ' x ', '/', '-', '+', '*', '^']:
            if sep in feature_name:
                parts = feature_name.split(sep)
                matched_groups = []

                for part in parts:
                    part = part.strip()
                    if part in flat_map:
                        matched_groups.append(flat_map[part])
                    elif part.replace("-", "_") in flat_map:
                        matched_groups.append(flat_map[part.replace("-", "_")])
                
                if matched_groups:
                    from collections import Counter
                    return Counter(matched_groups).most_common(1)[0][0]
                
        for base_feat, group in flat_map.items():
            if base_feat in feature_name or feature_name in base_feat:
                return group
            
        logger.warning(f"Feature '{feature_name}' not found in any group. Defaulting to 'unknown'.")
        return "unknown"
    
    def assign_feature_to_group_by_correlation(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Assign each feature to a group based on Pearson correlation with known base features.
        """
        from scipy.stats import pearsonr
        base_groups = self.config.get("FEATURE_GROUPS", {})
        assignment = {}

        # Flatten base features
        base_features = {feat: group for group, feats in base_groups.items() for feat in feats}

        for feat in df.columns:
            if feat == self.config['target_col']:
                continue

            max_corr = -1
            assigned_group = "unknown"

            for base_feat, group in base_features.items():
                if base_feat not in df.columns:
                    continue
                try:
                    corr, _ = pearsonr(df[feat], df[base_feat])
                    if abs(corr) > max_corr:
                        max_corr = abs(corr)
                        assigned_group = group
                except Exception:
                    continue
            
            assignment[feat] = assigned_group

        return assignment

    def plot_feature_importance(self, model, feature_names: List[str], cluster_name: str, model_type: str,
                                stage: Optional[str] = None, title: str = "Feature Importance",
                                save_path: Optional[str] = None,
                                group_assignment: Optional[Dict[str, str]] = None) -> None:
        """
        Plot feature importance for a model, using group-based color mapping (from group_assignment if provided).
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {type(model).__name__} does not have feature_importances_ attribute")
            return

        importances = model.feature_importances_

        if len(feature_names) != len(importances):
            logger.warning(f"Feature names length ({len(feature_names)}) does not match importances length ({len(importances)})")
            feature_names = feature_names[:len(importances)]

        mapped_names = self._map_feature_names(feature_names)

        inferred_groups = {}

        # prefer group_assignment if provided
        if group_assignment is not None:
            for feat in feature_names:
                if feat in group_assignment:
                    inferred_groups[feat] = group_assignment[feat]
                else:
                    inferred_groups[feat] = self.infer_group_from_feature_name(feat)
        else:
            for feat in feature_names:
                inferred_groups[feat] = self.infer_group_from_feature_name(feat)

        indices = np.argsort(importances)[::-1]
        sorted_importances = importances[indices]
        sorted_feature_names = [feature_names[i] for i in indices]
        sorted_mapped_names = [mapped_names[i] for i in indices]
        sorted_groups = [inferred_groups[feature_names[i]] for i in indices]

        group_colors = {'Pt cluster': '#3498dbff', 'interface': '#ed936bff', 'CeOx': '#7dc0a7ff'}
        bar_colors = [group_colors.get(g, '#7f7f7f') for g in sorted_groups]

        plt.figure(figsize=(16, 8))
        plt.rcParams['font.family'] = 'Arial'
        plt.title(f"{cluster_name} {model_type} feature importance", fontsize=16)
        plt.bar(range(len(sorted_mapped_names)), sorted_importances, color=bar_colors)
        plt.xticks(range(len(sorted_mapped_names)), sorted_mapped_names, rotation=45, ha='right', fontsize=14)
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

        mask = sorted_importances > 0.05
        if np.any(mask):
            filtered_importances = sorted_importances[mask]
            filtered_mapped_names = [sorted_mapped_names[i] for i in range(len(sorted_mapped_names)) if mask[i]]
            filtered_groups = [sorted_groups[i] for i in range(len(sorted_groups)) if mask[i]]
            filtered_colors = [group_colors.get(g, '#7f7f7f') for g in filtered_groups]

            plt.figure(figsize=(12, 6))
            plt.rcParams['font.family'] = 'Arial'
            plt.title(f"{cluster_name} {model_type} feature importance (>0.05)", fontsize=16)
            plt.bar(range(len(filtered_mapped_names)), filtered_importances, color=filtered_colors)
            plt.xticks(range(len(filtered_mapped_names)), filtered_mapped_names, rotation=45, ha='right', fontsize=14)
            plt.tight_layout()

            for fmt in self.fig_format:
                path = self.get_figure_save_path(
                    stage=stage,
                    cluster_name=cluster_name,
                    model_type=model_type,
                    plot_type='feature_importance_filtered',
                    fmt=fmt
                )
                plt.savefig(path, dpi=self.dpi, transparent=True)
                logger.info(f"Saved filtered feature importance plot to {path}")
            plt.close()
        else:
            logger.info(f"No features with importance > 0.05 for {cluster_name} {model_type} in {stage}.")

    def plot_correlation_network(self, correlation_df: pd.DataFrame, cluster_name: str,
                                 stage: Optional[str] = None,
                                 group_assignments: Optional[Dict[str, str]] = None) -> None:
        """
        Plot a correlation network graph from the correlation DataFrame.
        Only statistically significant correlations (after Bonferroni correction) are shown.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import numpy as np

        logger.info(f"Creating correlation network plot for {cluster_name}")

        if correlation_df.empty:
            logger.warning("Empty correlation DataFrame. Skipping network plot.")
            return

        df_sig = correlation_df[correlation_df["significant"]].copy()
        if df_sig.empty:
            logger.warning("No significant correlations to plot.")
            return
        
        if group_assignments is None:
            group_assignments = {}
            all_features = {}
            for _, row in df_sig.iterrows():
                all_features[row['feature_a']] = row['feature_a']
                all_features[row['feature_b']] = row['feature_b']
            for feat in all_features:
                group_assignments[feat] = self.infer_group_from_feature_name(feat)

        for _, row in df_sig.iterrows():
            for feat in [row['feature_a'], row['feature_b']]:
                if feat not in group_assignments:
                    group_assignments[feat] = self.infer_group_from_feature_name(feat)

        group_colors = {'Pt cluster': '#3498dbff', 'interface': '#ed936bff', 'CeOx': '#7dc0a7ff'}

        G = nx.Graph()
        for _, row in df_sig.iterrows():
            group_a = group_assignments.get(row['feature_a'], 'unknown')
            group_b = group_assignments.get(row['feature_b'], 'unknown') 
            mapped_a = self._map_feature_names([row['feature_a']])[0]
            mapped_b = self._map_feature_names([row['feature_b']])[0]

            node_a = f"{group_a}:{mapped_a}"
            node_b = f"{group_b}:{mapped_b}"
            G.add_node(node_a, group=group_a)
            G.add_node(node_b, group=group_b)
            G.add_edge(node_a, node_b, weight=abs(row["correlation"]), corr=row["correlation"])

        if len(G.nodes) == 0:
            logger.warning(f"No nodes in the graph for {cluster_name}. Skipping network plot.")
            return

        pos = nx.spring_layout(G, seed=42)
        plt.rcParams["font.family"] = "Arial"
        fig, ax = plt.subplots(figsize=(12, 10))

        edges = G.edges(data=True)
        corr_values = [data["corr"] for _, _, data in edges]
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        cmap = cm.get_cmap("viridis")
        edge_colors = [cmap(norm(corr)) for corr in corr_values]
        edge_widths = [3 + 4 * abs(data["corr"]) for _, _, data in edges]

        node_colors = [group_colors.get(node[1]['group'], '#7f7f7f') for node in G.nodes(data=True)]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=600, edgecolors='k', alpha=1.0)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths)

        label_pos = {k: (v[0], v[1] + 0.06) for k, v in pos.items()}
        nx.draw_networkx_labels(G, label_pos, ax=ax, font_size=14, font_color='black',
                                verticalalignment='bottom', horizontalalignment='center')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label('Pearson Correlation Coefficient', fontsize=16)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=group) for group, color in group_colors.items()
                            if any(node[1]['group'] in group_colors for node in G.nodes(data=True)) ]
        legend_elements = [Patch(facecolor=color, label=group) for group, color in group_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=14, frameon=False)
        plt.title(f"Significant Feature Correlation Network ({cluster_name})", fontsize=16)
        plt.axis("off")
        plt.tight_layout()

        for fmt in self.config.get("fig_formats", ["png"]):
            save_path = self.get_figure_save_path(
                cluster_name=cluster_name,
                stage=stage,
                plot_type="correlation_network",
                fmt=fmt
            )
            plt.savefig(save_path, dpi=self.dpi, transparent=True)
            logger.info(f"Saved correlation network to {save_path}")
        plt.close()

    def log_errors_by_threshold(
            self, y_true, y_pred, X, structures, cluster_name, model_type, stage, threshold=2.0, save_dir="results/errors"):
        """record errors by threshold"""
        import pandas as pd
        import os
        import numpy as np
        from pathlib import Path

        errors = np.abs(y_true - y_pred)
        mask = errors > threshold

        if mask.sum()==0:
            logger.info(f"No points with errors above threshold {threshold} for {cluster_name} {model_type} in {stage} are found.")
            return
        
        df_info = pd.DataFrame(X)
        df_info["true"] = y_true
        df_info["pred"] = y_pred
        df_info["error"] = errors
        df_info["structure"] = structures if structures is not None else ["N/A"] * len(y_true)

        df_filtered = df_info[mask].copy()
        df_filtered = df_filtered.sort_values(by="error", ascending=False)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{cluster_name}_{model_type}_{stage}_large_errors.csv")
        df_filtered.to_csv(save_path, index=False)

        print(f"[{cluster_name}][{model_type}] Saved {len(df_filtered)} error points to: {save_path}")
        logger.info(f"[{cluster_name}][{model_type}] Saved {len(df_filtered)} error points to: {save_path}")
            




 