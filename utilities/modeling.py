import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

AMENITY_COLS = ['air_conditioning', 'elevator', 'parking', 'fridge']
FINAL_FEATURES = [
    'area', 'dist_to_q1_km', 'amenity_score', 
    'street_val', 'district_val', 'amenity_ratio'
]

class MathFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, include_log=True, include_sqrt=True, include_inv=True, include_cube=True):
        self.include_log = include_log
        self.include_sqrt = include_sqrt
        self.include_inv = include_inv
        self.include_cube = include_cube
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = pd.DataFrame(X).copy()
        cols = X_new.columns
        for col in cols:
            vals = X_new[col].values
            vals_safe = np.abs(vals)
            if self.include_log: X_new[f'{col}_log'] = np.log1p(vals_safe)
            if self.include_sqrt: X_new[f'{col}_sqrt'] = np.sqrt(vals_safe)
            if self.include_inv: X_new[f'{col}_inv'] = 1 / (vals_safe + 0.1)
            if self.include_cube: X_new[f'{col}_cube'] = vals ** 3
        return X_new.values

def robust_outlier_removal(df, group_col='district', price_col='price'):
    def filter_group(g):
        if len(g) < 10: 
            return g
        
        low, high = g[price_col].quantile(0.05), g[price_col].quantile(0.95)
        return g[(g[price_col] >= low) & (g[price_col] <= high)]

    df_filtered = df.groupby(group_col, group_keys=False).apply(filter_group)

    return df_filtered.reset_index(drop=True)

def get_target_encoding_map(df, col, target, smooth=10):
    global_mean = df[target].mean()
    
    agg = df.groupby(col)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    
    smooth_means = (counts * means + smooth * global_mean) / (counts + smooth)
    
    return smooth_means, global_mean

def apply_target_encoding(df, col, encoding_map, global_mean):
    return df[col].map(encoding_map).fillna(global_mean)


def calculate_amenity_score(df: pd.DataFrame, amenity_cols: list) -> pd.DataFrame:
    """
    Tính điểm tiện ích dựa trên tổng các cột tiện ích nhị phân.
    Kiểm tra sự tồn tại của cột để tránh lỗi KeyError.
    """
    df_out = df.copy()
    valid_cols = [c for c in amenity_cols if c in df_out.columns]
    
    if valid_cols:
        df_out['amenity_score'] = df_out[valid_cols].sum(axis=1)
    else:
        df_out['amenity_score'] = 0
    return df_out

def preprocess_distance(df: pd.DataFrame, col_name: str = 'dist_to_q1_km') -> pd.DataFrame:
    """
    Xử lý các giá trị khoảng cách (ví dụ: chặn dưới để tránh chia cho 0 hoặc log lỗi).
    """
    df_out = df.copy()
    if col_name in df_out.columns:
        df_out[col_name] = df_out[col_name].clip(lower=0.1)
    return df_out

def select_model_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Lọc và chỉ giữ lại các cột cần thiết cho mô hình.
    """
    available_feats = [f for f in feature_list if f in df.columns]
    return df[available_feats]


def transform_data_pipeline(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm wrapper thực thi toàn bộ quy trình Feature Engineering.
    """
    df_processed = calculate_amenity_score(df_in, AMENITY_COLS)
    
    df_processed = preprocess_distance(df_processed)
    
    df_final = select_model_features(df_processed, FINAL_FEATURES)
    
    return df_final

def get_pipeline_feature_names(pipeline, input_features):
    math_gen = pipeline.named_steps['math_gen']
    math_feats = []
    for col in input_features:
        vals = [f"{col}_log" if math_gen.include_log else None,
                f"{col}_sqrt" if math_gen.include_sqrt else None,
                f"{col}_inv" if math_gen.include_inv else None,
                f"{col}_cube" if math_gen.include_cube else None]
        math_feats.extend([v for v in vals if v is not None])

    current_features = list(input_features) + math_feats
    
    if 'poly' in pipeline.named_steps and pipeline.named_steps['poly'] != 'passthrough':
        poly = pipeline.named_steps['poly']
        try:
            current_features = poly.get_feature_names_out(current_features)
        except:
            print("Warning: Không lấy được tên Poly features, dùng index.")
            current_features = [f"feat_{i}" for i in range(poly.n_output_features_)]
            
    return current_features