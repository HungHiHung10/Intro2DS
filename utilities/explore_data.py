import pandas as pd
def analyze_numerical_statistics(df, numerical_cols):
    """
    Tính toán các chỉ số thống kê mô tả chi tiết cho các cột dạng số.
    """
    summary = []
    for name_col in numerical_cols:
        col = df[name_col]
        
        mean_val = round(col.mean(), 2)
        median_val = round(col.median(), 2)
        std_val = round(col.std(), 2)
        skew_val = round(col.skew(), 2)
        
        if skew_val > 1: 
            shape = "Lệch phải"
        elif skew_val < -1: 
            shape = "Lệch trái"
        else: 
            shape = "Đối xứng"


        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(col < lower_bound) | (col > upper_bound)]
        num_outliers = len(outliers)
        percent_outliers = (num_outliers / len(df)) * 100

        # Kiểm tra dữ liệu lỗi/thiếu
        num_missing = col.isnull().sum()
        pct_missing = (num_missing / len(df)) * 100.0 
        num_impossible = (col <= 0).sum()
        
        placeholders_found = []
        for p_val in [0, -1, 999]:
            if (col == p_val).any():
                placeholders_found.append(str(p_val))
        placeholder_note = ", ".join(placeholders_found) if placeholders_found else "None"

        summary.append({
            "Column": name_col,
            "Mean": mean_val,
            "Median": median_val,
            "Std": std_val,
            "Skew": skew_val,
            "Shape": shape,
            "Min": col.min(),
            "Max": col.max(),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Outliers Count": num_outliers,
            "Percentage Outliers": f"{percent_outliers:.2f}%",
            "Missing (%)": f"{pct_missing:.2f}%",
            "Impossible (<=0)": num_impossible,
            "Placeholders": placeholder_note,
        })
    
    return pd.DataFrame(summary)

