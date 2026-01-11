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

def load_data(file_path):
    """
    Đọc dữ liệu từ file CSV và chuyển đổi cột ngày tháng sang kiểu datetime.
    
    Tham số:
    file_path (str): Đường dẫn đến file csv.
    
    Trả về:
    DataFrame: Dữ liệu đã được đọc và xử lý sơ bộ.
    """
    try:
        df = pd.read_csv(file_path)
        # Chuyển đổi cột date sang datetime nếu cột tồn tại
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        print(f"Đã đọc dữ liệu thành công. Kích thước: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại {file_path}")
        return None

def analyze_categorical_counts(df, column_name, top_n=10):
    """
    Thống kê số lượng và tần suất xuất hiện của các giá trị trong một cột phân loại.
    
    Tham số:
    df (DataFrame): Dữ liệu đầu vào.
    column_name (str): Tên cột cần phân tích.
    top_n (int): Số lượng giá trị phổ biến nhất muốn lấy.
    
    Trả về:
    Series: Top N giá trị phổ biến và số lượng tương ứng.
    """
    if column_name not in df.columns:
        print(f"Cảnh báo: Cột '{column_name}' không tồn tại.")
        return None
    
    # Đếm số lượng mỗi giá trị
    counts = df[column_name].value_counts()
    return counts.head(top_n)

def analyze_utilities(df, utility_cols):
    """
    Phân tích tỷ lệ phổ biến của các tiện ích (các cột dạng boolean/nhị phân).
    
    Tham số:
    df (DataFrame): Dữ liệu đầu vào.
    utility_cols (list): Danh sách tên các cột tiện ích.
    
    Trả về:
    DataFrame: Bảng thống kê số lượng và tỷ lệ % của từng tiện ích.
    """
    stats = []
    # Lọc các cột thực sự có trong dataframe
    valid_cols = [col for col in utility_cols if col in df.columns]
    
    for col in valid_cols:
        # Tính tổng số lượng có tiện ích (giả sử giá trị 1 là có)
        count = df[col].sum()
        total = len(df)
        percentage = (count / total) * 100
        stats.append({
            'Utility': col,
            'Count': count,
            'Percentage': percentage
        })
            
    stats_df = pd.DataFrame(stats)
    # Sắp xếp giảm dần theo mức độ phổ biến
    return stats_df.sort_values(by='Percentage', ascending=False)

def analyze_time_trends(df, date_col='date'):
    """
    Phân tích phân bố dữ liệu theo thời gian (Năm và Tháng).
    
    Tham số:
    df (DataFrame): Dữ liệu đầu vào.
    date_col (str): Tên cột ngày tháng.
    
    Trả về:
    tuple: (Thống kê theo năm, Thống kê theo tháng)
    """
    if date_col not in df.columns:
        return None, None
        
    # Loại bỏ các dòng không có dữ liệu ngày tháng hợp lệ
    valid_data = df[df[date_col].notna()]
    
    # Thống kê số lượng tin đăng theo năm
    year_counts = valid_data[date_col].dt.year.value_counts().sort_index()
    
    # Thống kê số lượng tin đăng theo tháng (tổng hợp tất cả các năm)
    month_counts = valid_data[date_col].dt.month.value_counts().sort_index()
    
    return year_counts, month_counts

def analyze_missing_values(df):
    """
    Kiểm tra và thống kê dữ liệu bị thiếu trong toàn bộ dataframe.
    
    Trả về:
    DataFrame: Bảng thống kê số lượng và tỷ lệ thiếu của các cột có dữ liệu thiếu.
    """
    missing_count = df.isnull().sum()
    missing_ratio = (missing_count / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_count,
        'Missing_Percentage': missing_ratio
    })
    
    # Chỉ lấy các cột có dữ liệu thiếu (> 0)
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    
    return missing_df.sort_values(by='Missing_Percentage', ascending=False)