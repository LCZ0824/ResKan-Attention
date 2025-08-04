import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

MASTER_SEED = 515

class DataPreprocessor:
    def __init__(self, random_seed=MASTER_SEED):
        self.random_seed = random_seed
        self.set_random_seeds()
        self.processed_data = None
        self.skewed_features = ['NT-proBNP', 'CRP']

    def set_random_seeds(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def load_data(self, file_name='standardized_medical_data_final.csv', target_variable='AF_Occurrence'):
        df = pd.read_csv(file_name)
        X_orig = df.drop(columns=[target_variable])
        y_orig = df[target_variable]
        return X_orig, y_orig

    def create_cv_splits(self, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        cv_splits = list(skf.split(X, y))
        return cv_splits

    def preprocess_fold(self, X_train_raw, y_train_raw, X_val_raw, y_val_raw, fold_id):
        X_train_log = X_train_raw.copy()
        X_val_log = X_val_raw.copy()
        for col in self.skewed_features:
            if col in X_train_log.columns:
                X_train_log[col] = np.log1p(X_train_log[col])
                X_val_log[col] = np.log1p(X_val_log[col])

        imputer = IterativeImputer(max_iter=10, random_state=self.random_seed)
        X_train_imputed = imputer.fit_transform(X_train_log)
        X_val_imputed = imputer.transform(X_val_log)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)

        smote = SMOTE(random_state=self.random_seed, k_neighbors=3)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_raw)

        fold_data = {
            'fold_id': fold_id,
            'X_train_processed': X_train_resampled,
            'y_train_processed': y_train_resampled,
            'X_val_processed': X_val_scaled,
            'y_val_processed': y_val_raw.values,
            'imputer': imputer,
            'scaler': scaler
        }
        return fold_data

    def process_all_folds(self, file_name='standardized_medical_data_final.csv',
                          target_variable='AF_Occurrence', n_splits=5):
        X_orig, y_orig = self.load_data(file_name, target_variable)
        cv_splits = self.create_cv_splits(X_orig, y_orig, n_splits)
        
        processed_folds = []
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train_raw = X_orig.iloc[train_idx]
            y_train_raw = y_orig.iloc[train_idx]
            X_val_raw = X_orig.iloc[val_idx]
            y_val_raw = y_orig.iloc[val_idx]
            
            fold_data = self.preprocess_fold(X_train_raw, y_train_raw, X_val_raw, y_val_raw, fold + 1)
            fold_data['train_indices'] = train_idx
            fold_data['val_indices'] = val_idx
            processed_folds.append(fold_data)
            
        self.processed_data = {
            'folds': processed_folds,
            'feature_names': list(X_orig.columns),
            'target_name': target_variable
        }
        return self.processed_data

    def get_fold_data(self, fold_id):
        if self.processed_data is None:
            raise ValueError("Data not processed yet. Run process_all_folds() first.")
        
        for fold_data in self.processed_data['folds']:
            if fold_data['fold_id'] == fold_id:
                return fold_data
        
        raise ValueError(f"Fold {fold_id} not found")

    def get_summary(self):
        if self.processed_data is None:
            raise ValueError("Data not processed yet. Run process_all_folds() first.")
            
        summary = {
            'n_folds': len(self.processed_data['folds']),
            'feature_names': self.processed_data['feature_names'],
            'n_features': len(self.processed_data['feature_names']),
            'random_seed': self.random_seed
        }
        
        fold_summaries = []
        for fold_data in self.processed_data['folds']:
            fold_summary = {
                'fold_id': fold_data['fold_id'],
                'train_samples': fold_data['X_train_processed'].shape[0],
                'val_samples': fold_data['X_val_processed'].shape[0],
                'train_target_ratio': np.mean(fold_data['y_train_processed']),
                'val_target_ratio': np.mean(fold_data['y_val_processed'])
            }
            fold_summaries.append(fold_summary)
        
        summary['folds'] = fold_summaries
        return summary

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_all_folds()
    summary = preprocessor.get_summary()
