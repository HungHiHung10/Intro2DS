# 1. System & Config
import warnings
warnings.filterwarnings('ignore')

# 2. Data Manipulation & Math
import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform

# 3. Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# 4. Machine Learning (External Libraries)
import shap
import xgboost as xgb # Gom import xgboost và import xgboost as xgb làm một

# 5. Scikit-learn: Base & Preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

# 6. Scikit-learn: Model Selection
from sklearn.model_selection import (
    KFold, learning_curve
)

# 7. Scikit-learn: Models
# from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import IsolationForest

# 8. Scikit-learn: Metrics
from sklearn.metrics import (
    r2_score, mean_absolute_error, 
    mean_squared_error, mean_absolute_percentage_error
)

# 9. Setup Visualization Style (Optional)
sns.set_theme(style="whitegrid")


def evaluate_metrics(y_ground_truth, y_predict, name="Model"):
    """Hàm đánh giá đầy đủ các chỉ số theo yêu cầu đề bài"""

    r2 = r2_score(y_ground_truth, y_predict)
    mae = mean_absolute_error(y_ground_truth, y_predict)
    mse = mean_squared_error(y_ground_truth, y_predict)
    rmse = np.sqrt(mse)
    
    return {
        'Model': name,
        'R2': r2,
        'MAE (Triệu VND)': mae,
        'MSE': mse,
        'RMSE': rmse
    }

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, area_strategy='log', dist_strategy='inv', score_strategy='log', include_ratios=True):
        self.area_strategy = area_strategy
        self.dist_strategy = dist_strategy
        self.score_strategy = score_strategy
        self.include_ratios = include_ratios
        self.feature_names_in_ = None
        
    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])])
        return self
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            df = X.copy()
            
        rename_map = {}
        
        # 1. AREA STRATEGY (Xử lý Diện tích)
        if 'area' in df.columns:
            val = np.abs(df['area'])
            if self.area_strategy == 'log':
                df['area'] = np.log1p(val)
                rename_map['area'] = 'area_log'
            elif self.area_strategy == 'sqrt':
                df['area'] = np.sqrt(val)
                rename_map['area'] = 'area_sqrt'

        # 2. DISTANCE STRATEGY (Xử lý Khoảng cách)
        if 'dist_to_q1_km' in df.columns:
            val = np.clip(np.abs(df['dist_to_q1_km']), 0.1, None)
            if self.dist_strategy == 'inv':
                df['dist_to_q1_km'] = 1 / val
                rename_map['dist_to_q1_km'] = 'dist_to_q1_inv'
            elif self.dist_strategy == 'log':
                df['dist_to_q1_km'] = np.log1p(val)
                rename_map['dist_to_q1_km'] = 'dist_to_q1_log'
            elif self.dist_strategy == 'bin':
                conditions = [val < 3, (val >= 3) & (val < 7), val >= 7]
                choices = [3, 2, 1] 
                df['dist_to_q1_km'] = np.select(conditions, choices, default=1)
                rename_map['dist_to_q1_km'] = 'dist_score_bin'

        # 3. AMENITY SCORE STRATEGY (Xử lý điểm tiện ích)
        score_cols = ['amenity_score', 'district_val', 'amenity_ratio']
        for col in score_cols:
            if col in df.columns:
                val = df[col]
                if self.score_strategy == 'log':
                    df[col] = np.log1p(val)
                    rename_map[col] = f'{col}_log'
                elif self.score_strategy == 'cube':
                    df[col] = val ** 3
                    rename_map[col] = f'{col}_cube'

        # 4. RATIO & INTERACTION FEATURES (Tạo biến tương tác)
        if self.include_ratios:
            if 'floors' in df.columns and 'area' in df.columns:
                df['area_per_floor'] = df['area'] / df['floors'].clip(lower=1)
            
            if 'area' in df.columns and 'district_val' in df.columns:
                df['area_x_district'] = df['area'] * df['district_val']

        df = df.rename(columns=rename_map)
        return df

    def get_feature_names_out(self, input_features=None):
        return None

def clean_and_remove_outliers(df):
    """Quy trình làm sạch dữ liệu và loại bỏ ngoại lai"""
    df_out = df.copy()
    
    # 1. Lọc theo Domain Knowledge (Kiến thức nghiệp vụ)
    df_out = df_out[(df_out['area'] >= 10) & (df_out['area'] <= 500)] 
    df_out = df_out[df_out['price'] > 0.5] 
    
    # 2. Isolation Forest (Lọc nhiễu đa chiều)
    numeric_cols = ['price', 'area', 'dist_to_q1_km']
    valid_cols = [c for c in numeric_cols if c in df_out.columns]
    
    if valid_cols:
        iso = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso.fit_predict(df_out[valid_cols])
        df_out = df_out[outliers == 1]
        print(f"[CLEAN] IsolationForest removed {len(df) - len(df_out)} outliers.")
        
    return df_out.reset_index(drop=True)

def get_target_encoding_map(df, col, target, smooth=20):
    """Tạo map Target Encoding với kỹ thuật Smoothing"""
    global_mean = df[target].mean()
    agg = df.groupby(col)[target].agg(['count', 'mean'])
    smooth_means = (agg['count'] * agg['mean'] + smooth * global_mean) / (agg['count'] + smooth)
    return smooth_means, global_mean

def apply_target_encoding(df, col, encoding_map, global_mean):
    return df[col].map(encoding_map).fillna(global_mean)

def create_massive_features(df):
    Df = df.copy()
    
    # --- A. Các biến cơ bản ---
    # 1. Amenity Score
    amenities = ['air_conditioning', 'elevator', 'parking', 'fridge', 
                 'washing_machine', 'mezzanine', 'kitchen', 'wardrobe', 
                 'bed', 'balcony', 'free_time']
    valid_cols = [col for col in amenities if col in Df.columns]
    Df['amenity_score'] = Df[valid_cols].sum(axis=1) if valid_cols else 0
    
    # --- B. Biến đổi Toán học (Mathematical Transforms) ---
    # Logarit (giảm độ lệch cho các biến phân phối đuôi dài)
    Df['area_log'] = np.log1p(Df['area'])
    Df['dist_log'] = np.log1p(Df['dist_to_q1_km'])
    
    # Căn bậc 2 (Sqrt)
    Df['area_sqrt'] = np.sqrt(Df['area'])
    
    # Nghịch đảo (Inverse) - Ý nghĩa: "Độ gần" thay vì "Khoảng cách"
    Df['dist_inv'] = 1 / (Df['dist_to_q1_km'] + 0.1)
    
    # --- C. Biến Tương tác (Interactions) - QUAN TRỌNG NHẤT ---
    # Mật độ tiện ích trên diện tích
    Df['amenity_density'] = Df['amenity_score'] / (Df['area'] + 1)
    
    # Diện tích nhân với độ gần trung tâm (Nhà to + Gần trung tâm = Giá trị cực đại)
    Df['area_x_dist_inv'] = Df['area'] * Df['dist_inv']
    
    # Diện tích nhân tiện ích
    Df['area_x_amenity'] = Df['area'] * Df['amenity_score']
    
    return Df

def robust_outlier_removal(df, group_col='district', price_col='price'):
    def filter_group(g):
        if len(g) < 10: return g
        low, high = g[price_col].quantile(0.05), g[price_col].quantile(0.95)
        return g[(g[price_col] >= low) & (g[price_col] <= high)]
    Df = df.groupby(group_col).apply(filter_group, include_groups=False).reset_index()
    return Df.drop(columns=[col for col in Df.columns if col in ['level_1', 'index']])


class KFoldTargetEncoder:
    def __init__(self, col, target, n_folds=5, smooth=10):
        self.col = col
        self.target = target
        self.n_folds = n_folds
        self.smooth = smooth
        self.global_mean = None
        self.map_dict = {}

    def fit_transform(self, df):
        df_encoded = df.copy()
        self.global_mean = df[self.target].mean()
        col_name = f"{self.col}_kfold_val"
        df_encoded[col_name] = np.nan
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(df_encoded):
            X_tr, X_val = df_encoded.iloc[train_idx], df_encoded.iloc[val_idx]
            mean = X_tr.groupby(self.col)[self.target].mean()
            count = X_tr.groupby(self.col)[self.target].count()
            smooth_mean = (count * mean + self.smooth * self.global_mean) / (count + self.smooth)
            df_encoded.loc[val_idx, col_name] = X_val[self.col].map(smooth_mean)
        df_encoded[col_name].fillna(self.global_mean, inplace=True)
        full_mean = df.groupby(self.col)[self.target].mean()
        full_count = df.groupby(self.col)[self.target].count()
        self.map_dict = (full_count * full_mean + self.smooth * self.global_mean) / (full_count + self.smooth)
        return df_encoded

    def transform(self, df):
        df_encoded = df.copy()
        col_name = f"{self.col}_kfold_val"
        df_encoded[col_name] = df_encoded[self.col].map(self.map_dict).fillna(self.global_mean)
        return df_encoded

def plot_xgboost_analysis(grid_search_object, X_train, y_train, X_test, y_test_real):
    """
    Vẽ 3 biểu đồ: Feature Importance, Residuals, và Loss Curve
    """
    # Lấy mô hình tốt nhất và tham số tốt nhất
    best_params = grid_search_object.best_params_
    best_model_trained = grid_search_object.best_estimator_
    
    # Thiết lập layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Lấy độ quan trọng và tên cột
    importances = best_model_trained.feature_importances_
    feature_names = X_train.columns
    
    # Tạo DataFrame và sort
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15) # Top 15
    
    sns.barplot(data=feat_df, y='Feature', x='Importance', ax=ax1, palette='viridis')
    ax1.set_title('Top 15 Feature Importances (XGBoost)')
    ax1.set_xlabel('Score')

    ax2 = fig.add_subplot(gs[0, 1])
    
    # Dự đoán (Log scale) -> Chuyển về Real scale
    y_pred_log = best_model_trained.predict(X_test)
    y_pred_real = np.expm1(y_pred_log)
    
    # Tính phần dư
    residuals = y_test_real - y_pred_real
    
    sns.scatterplot(x=y_pred_real, y=residuals, ax=ax2, alpha=0.6, color='coral')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('Residual Plot (Actual Scale)')
    ax2.set_xlabel('Giá Dự Đoán (Triệu VND)')
    ax2.set_ylabel('Sai Số: Thực tế - Dự đoán')
    
    ax3 = fig.add_subplot(gs[1, :])
    
    print(" Đang retrain mô hình với Best Params để lấy lịch sử Loss...")
    
    # Khởi tạo mô hình mới với tham số tốt nhất tìm được
    model_for_plotting = xgb.XGBRegressor(**best_params)
    
    # Chuẩn bị tập validation (cần log transform y_test để khớp với y_train)
    y_test_log = np.log1p(y_test_real)
    
    # Fit lại để lấy eval_set
    model_for_plotting.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test_log)],
        verbose=False
    )
    
    # Lấy kết quả
    results = model_for_plotting.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    # Vẽ
    ax3.plot(x_axis, results['validation_0']['rmse'], label='Train Loss (RMSE)', color='blue')
    ax3.plot(x_axis, results['validation_1']['rmse'], label='Test/Val Loss (RMSE)', color='orange')
    ax3.legend()
    ax3.set_ylabel('Log RMSE Loss')
    ax3.set_xlabel('Iterations (n_estimators)')
    ax3.set_title('XGBoost Learning Curve')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


    
def plot_catboost_evaluation(model, X_train, y_test, y_pred):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3)

    # --- PLOT 1: FEATURE IMPORTANCE ---
    # CatBoost có hàm lấy độ quan trọng tích hợp sẵn
    feature_importance = model.get_feature_importance()
    feature_names = X_train.columns
    
    # Tạo DataFrame để dễ vẽ
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False).head(20) # Lấy Top 20
    
    sns.barplot(x='Importance', y='Feature', data=fi_df, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Top 20 Feature Importances (CatBoost)', fontsize=14)
    axes[0, 0].set_xlabel('Score')

    # --- PLOT 2: ACTUAL vs PREDICTED (PERFECT FIT) ---
    sns.scatterplot(x=y_test, y=y_pred, ax=axes[0, 1], alpha=0.5, color='teal')
    # Vẽ đường chéo đỏ (kỳ vọng chuẩn)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 1].set_title(f'Actual vs Predicted (R2={r2_score(y_test, y_pred):.3f})', fontsize=14)
    axes[0, 1].set_xlabel('Giá Thực Tế')
    axes[0, 1].set_ylabel('Giá Dự Đoán')

    # --- PLOT 3: RESIDUAL PLOT (SCATTER) ---
    residuals = y_test - y_pred
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[1, 0], alpha=0.5, color='orange')
    axes[1, 0].axhline(0, color='red', linestyle='--', lw=2)
    axes[1, 0].set_title('Residual Plot (Kiểm tra phân phối lỗi)', fontsize=14)
    axes[1, 0].set_xlabel('Giá Dự Đoán')
    axes[1, 0].set_ylabel('Phần Dư (Thực tế - Dự đoán)')
    
    # Ghi chú cách đọc
    axes[1, 0].text(0.05, 0.95, "Tốt: Điểm phân bố ngẫu nhiên quanh đường 0", 
                    transform=axes[1, 0].transAxes, fontsize=10, color='green', va='top')

    # --- PLOT 4: RESIDUAL DISTRIBUTION (HISTOGRAM) ---
    sns.histplot(residuals, kde=True, ax=axes[1, 1], color='purple')
    axes[1, 1].axvline(0, color='red', linestyle='--')
    axes[1, 1].set_title('Phân Phối Của Phần Dư (Errors)', fontsize=14)
    axes[1, 1].set_xlabel('Error')
    
    plt.show()

def visualize_evaluation(model, X_train, y_train, X_test, y_test, feature_names):
    """
    Vẽ 3 biểu đồ chẩn đoán cho Linear Regression:
    1. Feature Coefficients (Độ quan trọng)
    2. Residual Plot (Kiểm tra lỗi)
    3. Learning Curve (Kiểm tra Overfit/Underfit)
    """
    
    # Setup layout: 3 biểu đồ
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Ưu tiên 1: Dùng feature_names được truyền vào
    if feature_names is not None:
        final_features = feature_names
    # Ưu tiên 2: Dùng tên cột của X_train nếu nó là DataFrame
    elif hasattr(X_train, 'columns'):
        final_features = X_train.columns
    # Ưu tiên 3: Tạo tên giả Feature 0, Feature 1...
    else:
        final_features = [f'Feature {i}' for i in range(X_train.shape[1])]
    
    if hasattr(model, 'coef_'):
        coefs = model.coef_
        # Nếu coefs là mảng 2D (1, n_features), làm phẳng nó
        if coefs.ndim > 1: coefs = coefs.ravel()
            
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs,
            'Abs_Coef': np.abs(coefs) # Dùng trị tuyệt đối để xếp hạng
        }).sort_values(by='Abs_Coef', ascending=False).head(15) # Lấy Top 15
        
        sns.barplot(x='Coefficient', y='Feature', data=coef_df, ax=axes[0], palette='viridis')
        axes[0].set_title("Top 15 Feature Coefficients (Importance)")
        axes[0].axvline(x=0, color='red', linestyle='--')
    else:
        axes[0].text(0.5, 0.5, "Model không có .coef_", ha='center')

    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train
    
    sns.scatterplot(x=y_pred_train, y=residuals, ax=axes[1], alpha=0.5, color='teal')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals (Ground Truth - Pred)")
    axes[1].set_title("Residual Plot")
    
    axes[1].text(0.05, 0.95, "Residuals", transform=axes[1].transAxes, fontsize=10, color='green', va='top')

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, 
        cv=5, 
        scoring='neg_root_mean_squared_error', # Dùng RMSE
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5) # 5 mốc kích thước training
    )
    
    # Chuyển đổi score từ âm sang dương (vì sklearn trả về neg_rmse)
    train_scores_mean = -np.mean(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)
    
    axes[2].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Error")
    axes[2].plot(train_sizes, val_scores_mean, 'o-', color="g", label="Validation Error")
    axes[2].set_xlabel("Training Samples")
    axes[2].set_ylabel("RMSE Error (Thấp là tốt)")
    axes[2].set_title("Learning Curve")
    axes[2].legend(loc="best")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()