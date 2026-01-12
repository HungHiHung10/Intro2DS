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

def analyze_district_distribution(
    df,
    district_col="address",
    amenity_cols=None,
    top_n=10,
    verbose=True
):
    """
    Phân tích phân bố số lượng tin theo địa bàn và kiểm tra các tiện ích có trong dataset.

    Tham số
    -------
    df : pd.DataFrame
        Dataset đầu vào.
    district_col : str
        Tên cột đại diện cho địa bàn / khu vực.
    amenity_cols : list[str] hoặc None
        Danh sách các cột tiện ích cần kiểm tra.
    top_n : int
        Số lượng địa bàn top nhiều / ít tin cần lấy.
    verbose : bool
        Có in kết quả và hiển thị bảng hay không.

    Giá trị trả về
    --------------
    dict gồm:
        - amenity_cols_found : danh sách tiện ích thực sự tồn tại trong df
        - district_counts    : số lượng tin theo từng địa bàn
        - top_most           : top địa bàn có nhiều tin nhất
        - top_least          : top địa bàn có ít tin nhất
        - df_top             : dataframe con chỉ gồm top địa bàn nhiều tin
        - total_posts        : tổng số tin toàn dataset
        - top_posts          : số tin thuộc top địa bàn nhiều tin
        - top_ratio          : tỷ lệ (%) số tin thuộc top địa bàn
    """

    # Nếu không truyền danh sách tiện ích thì dùng danh sách rỗng
    if amenity_cols is None:
        amenity_cols = []

    # Chỉ giữ lại những cột tiện ích thực sự tồn tại trong dataset
    amenity_cols_found = [c for c in amenity_cols if c in df.columns]

    # Đếm số lượng tin theo địa bàn (loại bỏ giá trị NaN)
    district_counts = df[district_col].dropna().value_counts()

    # Top địa bàn có nhiều tin nhất
    top_most = district_counts.nlargest(top_n)

    # Top địa bàn có ít tin nhất
    top_least = district_counts.nsmallest(top_n)

    # Dataset con chỉ gồm các tin thuộc top địa bàn nhiều tin
    df_top = df[df[district_col].isin(top_most.index)].copy()

    # Tổng số tin toàn bộ dataset
    total_posts = len(df)

    # Tổng số tin thuộc top địa bàn nhiều tin
    top_posts = top_most.sum()

    # Tỷ lệ (%) số tin thuộc top địa bàn
    top_ratio = top_posts / total_posts * 100 if total_posts > 0 else 0

    # In kết quả nếu verbose = True
    if verbose:
        print("Các tiện ích tìm thấy trong dataset:", amenity_cols_found)

        print(f"\nTop {top_n} địa bàn có nhiều tin nhất:")
        display(top_most)

        print(f"\nTop {top_n} địa bàn có ít tin nhất:")
        display(top_least)

        print(f"\nTổng số tin toàn bộ dataset: {total_posts}")
        print(f"Số tin tại Top {top_n} địa bàn nhiều tin nhất: {top_posts}")
        print(f"Tỷ lệ chiếm: {top_ratio:.2f}%")

    return {
        "amenity_cols_found": amenity_cols_found,
        "district_counts": district_counts,
        "top_most": top_most,
        "top_least": top_least,
        "df_top": df_top,
        "total_posts": total_posts,
        "top_posts": top_posts,
        "top_ratio": top_ratio,
    }

def analyze_amenities_by_district(
    df_top,
    district_col,
    amenity_cols,
    top_districts,
    top_n=10,
    verbose=True
):
    """
    Phân tích tỷ lệ (%) phòng có từng tiện ích theo địa bàn (top N nhiều tin nhất),
    đồng thời tóm tắt địa bàn có tỷ lệ cao nhất / thấp nhất cho mỗi tiện ích.

    Tham số
    -------
    df_top : pd.DataFrame
        Dataset con chỉ gồm các địa bàn top nhiều tin.
    district_col : str
        Tên cột địa bàn.
    amenity_cols : list[str]
        Danh sách cột tiện ích (0/1 hoặc True/False).
    top_districts : pd.Index hoặc list
        Danh sách địa bàn top nhiều tin (thường là top_most.index).
    top_n : int
        Số lượng địa bàn top dùng cho hiển thị.
    verbose : bool
        Có in và display kết quả hay không.

    Giá trị trả về
    --------------
    dict gồm:
        - district_amenities_pct : bảng % tiện ích theo địa bàn
        - summary_amenities      : bảng tóm tắt max / min cho từng tiện ích
    """

    # =========================
    # 1. Bảng % tiện ích theo địa bàn
    # =========================

    district_amenities_pct = (
        df_top
        .groupby(district_col)[amenity_cols]
        .mean(numeric_only=True)     # trung bình → tỷ lệ
        .mul(100)                    # đổi sang %
        .reindex(top_districts)      # giữ đúng thứ tự top địa bàn
        .round(2)
    )

    # =========================
    # 2. Hàm tóm tắt cho 1 tiện ích
    # =========================
    def summarize_amenity(col_ser: pd.Series) -> pd.Series:
        """
        Tóm tắt 1 tiện ích:
        - Nếu toàn NaN → trả 'Không có dữ liệu'
        - Ngược lại → lấy địa bàn có tỷ lệ cao nhất / thấp nhất
        """
        s = col_ser.dropna()
        if s.empty:
            return pd.Series({
                "max_address": "Không có dữ liệu",
                "max_pct":     np.nan,
                "min_address": "Không có dữ liệu",
                "min_pct":     np.nan,
            })

        return pd.Series({
            "max_address": s.idxmax(),
            "max_pct":     s.max(),
            "min_address": s.idxmin(),
            "min_pct":     s.min(),
        })

    # =========================
    # 3. Bảng tóm tắt cho tất cả tiện ích
    # =========================
    summary_amenities = (
        district_amenities_pct
        .apply(summarize_amenity, axis=0)  # mỗi cột tiện ích
        .T                                  # tiện ích thành từng dòng
        .reset_index(names="amenity")
        .sort_values("max_pct", ascending=False)
    )

    # =========================
    # 4. Hiển thị
    # =========================
    if verbose:
        print(
            f"Bảng tổng hợp: Phần trăm phòng có từng tiện ích theo địa bàn "
            f"(top {top_n} nhiều tin nhất):"
        )
        display(district_amenities_pct)

        print("\nĐịa bàn có tỷ lệ tiện ích cao nhất / thấp nhất cho từng tiện ích:")
        display(summary_amenities.round(2))

    return {
        "district_amenities_pct": district_amenities_pct,
        "summary_amenities": summary_amenities,
    }

def summarize_price_by_top_districts(
    df_price: pd.DataFrame,
    district_col: str = "address",
    price_col: str = "price",
    top_n: int = 10,
    sort_by: str = "median",
    ascending: bool = False,
    round_digits: int = 0,
    display_styled: bool = True,
):
    """
    Tóm tắt thống kê giá theo top N địa bàn có nhiều tin nhất.

    Trả về:
      - price_by_district: DataFrame thống kê (count, mean, median, q25, q75, min, max)
      - target_districts: Index các địa bàn top N theo số tin
      - df_district: DataFrame con chỉ gồm các địa bàn đó
    """

    # Kiểm tra cột bắt buộc
    required = {district_col, price_col}
    missing = required - set(df_price.columns)
    assert not missing, f"Thiếu cột bắt buộc: {missing}"

    # Lấy top N địa bàn theo số lượng tin
    target_districts = df_price[district_col].value_counts().nlargest(top_n).index

    # Lọc dữ liệu thuộc top N địa bàn
    df_district = df_price.loc[df_price[district_col].isin(target_districts)].copy()

    # Thống kê giá theo địa bàn (không cần định nghĩa q25/q75 riêng)
    price_by_district = (
        df_district
        .groupby(district_col, dropna=True)[price_col]
        .agg(
            count="size",
            mean="mean",
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            min="min",
            max="max",
        )
        .sort_values(sort_by, ascending=ascending)
        .round(round_digits)
    )

    if display_styled:
        print("Giá phòng theo top địa bàn:", end="")
        display(
            price_by_district.style
                .format("{:,.0f}")
                .set_properties(**{
                    "text-align": "center",
                    "font-size": "13px"
                })
                .set_table_styles([
                    dict(selector="th", props=[
                        ("text-align", "center"),
                        ("font-weight", "bold"),
                    ])
                ])
        )

    return price_by_district, target_districts, df_district

def summarize_price_by_district_street(
    df_price: pd.DataFrame,
    district_col: str = "address",
    street_col: str = "street_name",
    price_col: str = "price",
    amenity_cols=None,
    has_area: bool = False,
    area_col: str = "area",
    min_ads_street: int = 15,
    min_ads_pair: int = 10,
    round_digits: int = 2,
    verbose: bool = True,
):
    """
    Thống kê giá theo cặp (địa bàn, street) sau khi:
      1) Lọc street có >= min_ads_street tin
      2) Groupby theo (district, street) để tính stats giá + (area nếu có) + tiện ích trung bình
      3) Lọc cặp (district, street) có >= min_ads_pair tin
    """

    if amenity_cols is None:
        amenity_cols = []

    # --- Kiểm tra cột bắt buộc ---
    required = {district_col, street_col, price_col}
    missing = required - set(df_price.columns)
    assert not missing, f"Thiếu cột bắt buộc: {missing}"

    # --- Lọc street đủ số tin ---
    street_counts = df_price[street_col].value_counts(dropna=True)
    valid_streets = street_counts[street_counts >= min_ads_street].index
    df_street = df_price.loc[df_price[street_col].isin(valid_streets)].copy()

    # --- Xác định các cột tiện ích thực sự tồn tại (không for) ---
    amenity_cols = pd.Index(amenity_cols)
    amenity_cols_found = df_street.columns.intersection(amenity_cols)

    # --- Build named-aggregation cho groupby.agg(**group_stats) ---
    group_stats = {
        "count":        (price_col, "size"),
        "mean_price":   (price_col, "mean"),
        "median_price": (price_col, "median"),
        "q25":          (price_col, lambda s: s.quantile(0.25)),
        "q75":          (price_col, lambda s: s.quantile(0.75)),
        "min_price":    (price_col, "min"),
        "max_price":    (price_col, "max"),
    }

    # Thêm area nếu có và cột tồn tại
    if has_area and (area_col in df_street.columns):
        group_stats.update({
            "mean_area":   (area_col, "mean"),
            "median_area": (area_col, "median"),
        })

    # Thêm tiện ích trung bình (không for):
    # tạo dict dạng {"amenity_x": ("x","mean"), ...}
    amenity_aggs = (
        pd.Series(amenity_cols_found, index=amenity_cols_found)
        .map(lambda c: (c, "mean"))
    )
    # đổi key thành amenity_<col>
    amenity_aggs.index = "amenity_" + amenity_aggs.index.astype(str)
    group_stats.update(amenity_aggs.to_dict())

    # --- Groupby & aggregate ---
    price_by_address_street = (
        df_street
        .groupby([district_col, street_col], dropna=True)
        .agg(**group_stats)
        .reset_index()
    )

    # --- Lọc cặp đủ dữ liệu ---
    price_by_address_street = price_by_address_street.loc[
        price_by_address_street["count"] >= min_ads_pair
    ].copy()

    # --- Sort & round ---
    price_by_address_street = (
        price_by_address_street
        .sort_values(by=["median_price", "mean_price"], ascending=[False, False])
        .round(round_digits)
    )

    if verbose:
        print("Số cặp (district, street) giữ lại:", len(price_by_address_street))
        display(price_by_address_street.head(20))

    return price_by_address_street, df_street, valid_streets, amenity_cols_found

def top_street_premium_vs_district(
    price_by_address_street: pd.DataFrame,
    district_col: str = "address",
    street_col: str = "street_name",
    median_col: str = "median_price",
    top_n: int = 20,
    include_area: bool = True,
    sample_amenities=None,
    round_digits: int = 2,
    display_result: bool = True,
):
    """
    Tính median giá của mỗi quận, sau đó tính chênh lệch (median_street - median_district),
    và lấy top N tuyến đường có chênh lệch cao nhất.
    """

    if sample_amenities is None:
        sample_amenities = [
            "amenity_air_conditioning",
            "amenity_balcony",
            "amenity_mezzanine",
            "amenity_elevator",
            "amenity_parking",
        ]

    df = price_by_address_street.copy()

    # 1) Median mỗi quận
    district_median = df.groupby(district_col)[median_col].median()

    # 2) Map vào từng dòng và tính delta
    df["district_median"] = df[district_col].map(district_median)
    df["delta_vs_district"] = df[median_col] - df["district_median"]

    # 3) Top N tuyến đường premium hơn quận
    top_delta = (
        df.sort_values("delta_vs_district", ascending=False)
          .head(top_n)
          .reset_index(drop=True)
    )

    # 4) Chọn cột hiển thị (không for)
    base_cols = pd.Index([
        district_col,
        street_col,
        "count",
        median_col,
        "district_median",
        "delta_vs_district",
    ])

    area_cols = pd.Index(["mean_area", "median_area"]) if include_area else pd.Index([])

    amenity_cols = pd.Index(sample_amenities)

    cols_to_show = (
        base_cols
        .append(area_cols)
        .append(amenity_cols)
    )

    # Giữ lại những cột thực sự tồn tại trong df
    cols_to_show = top_delta.columns.intersection(cols_to_show)

    out = top_delta.loc[:, cols_to_show].round(round_digits)

    if display_result:
        display(out)

    return out, top_delta

def analyze_amenity_price_effect(
    df_price: pd.DataFrame,
    amenity_cols,
    price_col: str = "price",
    round_digits: int = 2,
    display_result: bool = True,
):
    """
    Phân tích ảnh hưởng của từng tiện ích (0/1) đến median giá phòng.

    - So sánh median giá giữa nhóm có / không có tiện ích
    - Tính chênh lệch tuyệt đối và phần trăm
    """

    # Chỉ giữ các tiện ích tồn tại trong dữ liệu
    amenity_cols = df_price.columns.intersection(pd.Index(amenity_cols))
    if len(amenity_cols) == 0:
        print("Không có tiện ích hợp lệ.")
        return None

    # Chuẩn hóa tiện ích về 0/1
    amenity_df = (
        df_price[amenity_cols]
        .apply(lambda s: s.astype(int) if s.dtype == bool else s)
    )

    # Chỉ giữ các dòng có giá và tiện ích hợp lệ (0/1)
    valid_mask = amenity_df.isin([0, 1]).all(axis=1) & df_price[price_col].notna()
    df_valid = pd.concat(
        [df_price[[price_col]], amenity_df],
        axis=1
    ).loc[valid_mask]

    if df_valid.empty:
        print("Không đủ dữ liệu hợp lệ để phân tích.")
        return None

    # Chuyển sang dạng long để groupby 1 lần
    long_df = (
        df_valid
        .melt(
            id_vars=price_col,
            var_name="amenity",
            value_name="has_amenity"
        )
    )

    # Thống kê theo (amenity, có/không)
    stats = (
        long_df
        .groupby(["amenity", "has_amenity"])[price_col]
        .agg(
            count="size",
            median="median"
        )
        .reset_index()
    )

    # Pivot để tách nhóm 0 / 1
    pivot = stats.pivot(
        index="amenity",
        columns="has_amenity",
        values=["count", "median"]
    )

    # Chỉ giữ các tiện ích có đủ cả 2 nhóm
    pivot = pivot.dropna(subset=[("median", 0), ("median", 1)])

    if pivot.empty:
        print("Không có tiện ích nào có đủ 2 nhóm 0/1 để so sánh median.")
        return None

    # Tính chênh lệch
    result = pd.DataFrame({
        "amenity": pivot.index,
        "Số lượng (không có)": pivot[("count", 0)],
        "Số lượng (có)":      pivot[("count", 1)],
        "Median (không)":     pivot[("median", 0)],
        "Median (có)":        pivot[("median", 1)],
    })

    result["Chênh lệch"] = result["Median (có)"] - result["Median (không)"]
    result["Chênh lệch (%)"] = (
        result["Chênh lệch"] / result["Median (không)"] * 100
    )

    # Làm đẹp tên tiện ích
    result["Tiện ích"] = (
        result["amenity"]
        .str.replace("_", " ")
        .str.title()
    )

    # Sắp xếp theo mức ảnh hưởng
    result = (
        result[
            [
                "Tiện ích",
                "Số lượng (không có)",
                "Số lượng (có)",
                "Median (không)",
                "Median (có)",
                "Chênh lệch",
                "Chênh lệch (%)",
            ]
        ]
        .round(round_digits)
        .sort_values("Chênh lệch (%)", ascending=False)
        .reset_index(drop=True)
    )

    if display_result:
        print("Ảnh hưởng của từng tiện ích đến median giá phòng:")
        display(result)

    return result