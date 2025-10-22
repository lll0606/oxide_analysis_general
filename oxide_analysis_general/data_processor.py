"""
Data Processing Module for Oxide Descriptor Analysis
This module provides functionality for loading, preprocessing, and creating
physics-informed features for oxide descriptor data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler

from .config import DEFAULT_CONFIG

# Configure logger
logger = logging.getLogger(__name__)


class DataProcessor:
    """Class for loading and preprocessing oxide descriptor data"""
    
    def __init__(self, config: Dict):
        """
        Initialize the data processor with configuration
        """
        self.config = config
        self.feature_cols = config["feature_cols"]
        self.target_col = config["target_col"]
        self.structure_col = config["structure_col"]
        self.smart_feature_rules = None
        logger.info("Data processor initialized")
    
    def load_data(self, filename: str, sheet_name: str) -> pd.DataFrame:
        """
        Load and process Excel data
        """
        logger.info(f"Processing {sheet_name} data from {filename}")
        df = pd.read_excel(filename, sheet_name=sheet_name)

        if isinstance(df, dict):
            first_key = next(iter(df))
            logger.warning(f"Excel file returned multiple sheets; using first sheet '{first_key}' by default.")
            df = df[first_key]
        
        # Ensure data types are correct
        numeric_cols = [col for col in self.feature_cols + [self.target_col] 
                        if col in df.columns]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN in essential columns
        available_features = [col for col in self.feature_cols if col in df.columns]
        df = df.dropna(subset=available_features + [self.target_col])

        logger.info(f"Data shape after processing: {df.shape}")
        logger.info(f"Processed {sheet_name} data points: {len(df)}")
        
        return df
    
    def get_available_features(self, data: pd.DataFrame) -> List[str]:
        """
        Get available features from the configuration list that exist in the data
        """
        return [col for col in self.feature_cols if col in data.columns]
    

    def prepare_modeling_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Prepare data for modeling by extracting features and target.
        Args:
            data: DataFrame containing the data
        Returns:
            Tuple of (features, target, structures)
        """
        if data is None:
            logger.error("input data is None")
            return None, None, None

        logger.info(f"original data shape: {data.shape}")
        logger.info(f"original data.columns: {data.columns.tolist()}")
        logger.info(f"original config['feature_cols']: {self.feature_cols}")

        # check target column existence
        if self.target_col not in data.columns:
            logger.error(f"Target column '{self.target_col}' not found in data")
            return None, None, None

        # Check which features are available
        available_features = self.get_available_features(data)
        logger.info(f"the feature_cols in config: {self.feature_cols}")
        logger.info(f" Available feature_cols in DataFrame: {available_features}")

        # Check which features are numeric
        numeric_features = [col for col in available_features if pd.api.types.is_numeric_dtype(data[col])]
        logger.info(f" Numeric feature columns: {numeric_features}")

        # Check for missing values
        missing_matrix = data[numeric_features + [self.target_col]].isna()
        missing_counts = missing_matrix.sum().to_dict()
        logger.info(f"Missing values count: {missing_counts}")

        # Drop rows with missing values
        data_clean = data.dropna(subset=numeric_features + [self.target_col])
        logger.info(f"Rows after dropna: {len(data_clean)} (total {len(data)} rows)")

        if len(data_clean) == 0:
            logger.error("No data left after removing rows with missing values!!!")
            return None, None, None

        # Extract features and target
        X = data_clean[numeric_features].copy()
        logger.info(f"Extracted feature columns: {X.columns.tolist()}")

        y = data_clean[self.target_col]

        # Extract structure information (if exists)
        if self.structure_col in data_clean.columns:
            structures = data_clean[self.structure_col]
            logger.info(f"Structure column '{self.structure_col}' extracted successfully, containing {len(np.unique(structures))} unique structures")
        else:
            logger.warning(f"Structure column '{self.structure_col}' not found, current columns are: {data_clean.columns.tolist()}")
            structures = None

        return X, y, structures

    def preprocess_features(self, X_physics):
        """preprocess features to remove invalid values and features"""
        bad_columns = []

        # check the NaN
        nan_cols = X_physics.columns[X_physics.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"NaN columns: {nan_cols} will be removed")
            bad_columns.extend(nan_cols)

        # check the infinite values
        inf_mask = ~np.isfinite(X_physics.values)
        inf_cols = X_physics.columns[inf_mask.any(axis=0)].tolist()
        if inf_cols:
            logger.warning(f"Infinite columns: {inf_cols} will be removed")
            bad_columns.extend(inf_cols)
        
        # remove bad columns
        good_columns = [col for col in X_physics.columns if col not in bad_columns]
        if bad_columns:
            logger.warning(f"Removing from {len(X_physics.columns)} with {len(bad_columns)} problematic columns")
        
        return X_physics[good_columns]

    def create_physics_features_manual(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create physics-informed features from base descriptors
        Args:
            data: DataFrame containing base descriptors
        Returns:
            DataFrame with added physics-informed features
        """
        features = data.copy()
        # Check if the required columns are present
        logger.info("Creating physics-informed features")
        X = data[self.get_available_features(data)].copy()
        physics_features = pd.DataFrame(index=X.index)
        
        # Metal-oxygen interaction features
        if all(col in X.columns for col in ['polarons', 'Ce3_dist_min']):
            physics_features['polaron_proximity'] = X['polarons'] / (X['Ce3_dist_min'] + 0.1)
            if 'Ce3_dist_sum' in X.columns:
                physics_features['polaron_density'] = X['polarons'] / (X['Ce3_dist_sum'] + 0.1)
        
        # Charge transfer and structural deformation relationship
        if all(col in X.columns for col in ['Δq', 'RMSD']):
            physics_features['charge_deformation_ratio'] = X['Δq'] / (X['RMSD'] + 0.01)
            physics_features['charge_deformation_product'] = X['Δq'] * X['RMSD']
        
        # Electronic structure factors
        if 'ε' in X.columns:
            if 'Ce3_dist_mean' in X.columns:
                physics_features['electronic_structure_factor'] = X['ε'] * X['Ce3_dist_mean']
            if 'polarons' in X.columns:
                physics_features['polaron_electronic_coupling'] = X['ε'] * X['polarons']

        # Metal-support binding energy related
        if all(col in X.columns for col in ['Δq', 'polarons']):
            physics_features['charge_transfer_per_polaron'] = X['Δq'] / (X['polarons'] + 0.1)
        
        # Pt-O bond related
        if 'Pt-O_bonds' in X.columns:
            physics_features['Pt-O_bond_mean_squared'] = X['Pt-O_bonds'] ** 2
            #cross with other features
            if 'polarons' in X.columns:
                physics_features['Pt-O_bond_polaron_ratio'] = X['Pt-O_bonds'] / (X['polarons'] + 0.1)
            if 'Ce3_dist_min' in X.columns:
                physics_features['Pt-O_bond_Ce3_dist_min'] = X['Pt-O_bonds'] * (X['Ce3_dist_min'] + 0.01)
        
        # Combine electronic and structural features 
        if all(col in X.columns for col in ['Δq', 'RMSD', 'polarons']):
            # Comprehensive descriptor combining charge, structure, and polarons
            physics_features['charge_structure_polaron_index'] = (X['Δq'] * X['polarons']) / (X['RMSD'] + 0.01)
        
        # add combined binding energy descriptors
        if all(col in X.columns for col in ['RMSD', 'Δq', 'polarons', 'ε']):
            physics_features['binding_energy_combination'] = X['RMSD'] * X['Δq'] / ( X['polarons'] * X['ε'])

        # add clustering-based features
        if X.shape[0] >= 10 and X.shape[1] >= 3:
            try:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                X_scaled = StandardScaler().fit_transform(X.select_dtypes(include=[np.number]))
                kmeans = KMeans(n_clusters=3, random_state=42,n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                physics_features['cluster_labels'] = cluster_labels
            except Exception as e:
                logger.warning(f"KMeans clustering failed: {e}")

        # Combine original and physics features
        X_physics = pd.concat([X, physics_features], axis=1)
        logger.info(f"Extended feature set shape: {X_physics.shape}")
        logger.info(f"New physics features: {physics_features.columns.tolist()}") 
        return X_physics

    def create_physics_features_smart(self, X: pd.DataFrame, y: pd.Series, fit: bool = True) -> pd.DataFrame:
        """
        Construct physics-informed features using smart selection + clustering.
        Parameters taken from config["smart_physical_feature"]
        """
        from scipy.stats import pearsonr
        from sklearn.cluster import AgglomerativeClustering
        import numpy as np

        smart_config = self.config.get("smart_physical_feature", {})
        top_n = smart_config.get("top_n", 8)
        n_clusters = smart_config.get("n_clusters", 3)
        max_total_features = smart_config.get("max_total_features",20)

        if fit or self.smart_feature_rules is None:
            logger.info("Using SMART physical feature construction (Pearson + Clustering)")

            # Step 1: choose top-N features based on correlation with the target variable
            scores = []
            for col in X.columns:
                try:
                    r, _ = pearsonr(X[col], y)
                    scores.append((col, abs(r)))
                except:
                    continue
            top_features = [col for col, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]]
            logger.info(f"Top-{top_n} correlated features: {top_features}")

            # Step 2: cluster the top features based on their pairwise correlation
            corr_matrix = X[top_features].corr().abs()
            dist_matrix = 1 - corr_matrix
            n_features = dist_matrix.shape[0]
            n_clusters_safe = min(n_clusters, n_features)
            # if the feature number is too small,then skip the cluster just use the original features
            if n_clusters_safe < 2:
                logger.warning(f"[create_physics_features_smart] Not enough features {n_features} to cluster, using original features")
                feature_groups = {f: 0 for f in top_features}
            else:
                clustering = AgglomerativeClustering(n_clusters=n_clusters_safe, metric='precomputed', linkage='average')
                labels = clustering.fit_predict(dist_matrix)
                feature_groups = {f: int(l) for f, l in zip(top_features, labels)}
                logger.info(f"Feature groups after clustering: {feature_groups}")
            # save the feature groups
            self.smart_feature_rules = {
                "top_features": top_features,
                "feature_groups": feature_groups,
                "max_total_features": max_total_features
            }
        else:
            logger.info("Applying previously fitted SMART feature rules")
            top_features = self.smart_feature_rules['top_features']
            feature_groups = self.smart_feature_rules['feature_groups']
            max_total_features = self.smart_feature_rules['max_total_features']
        # Step 3: combine features from different clusters
        new_features = pd.DataFrame(index=X.index)
        combination_count = 0

        for i, f1 in enumerate(top_features):
            for j, f2 in enumerate(top_features):
                if i < j and feature_groups[f1] != feature_groups[f2]:
                    if combination_count + 3 > max_total_features:
                        logger.warning(f"Reached maximum allowed physical features: {max_total_features}")
                        break
                    new_features[f"{f1}_x_{f2}"] = X[f1] * X[f2]
                    new_features[f"{f1}/{f2}"] = X[f1] / (X[f2] + 1e-6)
                    new_features[f"{f1}-{f2}"] = X[f1] - X[f2]
                    combination_count += 3

        # combine original features with new features
        X_combined = pd.concat([X[top_features], new_features], axis=1)
        logger.info(f"Final SMART physical features: {X_combined.shape[1]} columns created")
        return X_combined

    def create_physics_features(self, X: pd.DataFrame, y: pd.Series, fit: bool = True) -> pd.DataFrame:
        smart_config = self.config.get("smart_physical_feature", {})
        use_smart = smart_config.get("use_smart", True)
        if use_smart:
            return self.create_physics_features_smart(X, y,fit=fit)
        else:
            return self.create_physics_features_manual(X)

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
                      method: str = 'standard') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Scale features using specified method
        Args:
            X_train: Training features
            X_test: Test features (optional)
            method: Scaling method ('standard', 'robust')
        Returns:
            Tuple of (scaled_X_train, scaled_X_test)
        """
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        if X_test is not None:
            X_test = X_test.replace([np.inf, -np.inf], np.nan)

        # record original shape
        orgin_train_shape = X_train.shape[0]
        orgin_test_shape = X_test.shape[0] if X_test is not None else 0

        # Remove rows with NaN in essential columns
        X_train = X_train.dropna()
        if X_test is not None:
            X_test = X_test.dropna()
        
        #log how many rows are dropped
        logger.info(f"[Scaling] X_train rows: {orgin_test_shape} -> {X_train.shape[0]} after dropna")
        if X_test is not None:
            logger.info(f"[Scaling] X_test rows: {orgin_test_shape} -> {X_test.shape[0]} after dropna")
        # record in the log
        bad_cols = X_train.columns[X_train.isna().any()].tolist()
        if len(bad_cols) > 0:
            logger.warning(f"NaN columns in training data: {list(bad_cols)}")
            
        if method == 'robust':
            scaler = RobustScaler()
        else:  # default to standard
            scaler = StandardScaler()
            
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled, scaler
        
        return X_train_scaled, None, scaler
    