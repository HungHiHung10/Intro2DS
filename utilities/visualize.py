import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("whitegrid")

def visualize_rental_price_analysis(df_line_chart, df_bar_chart):
    """
    Vẽ 2 biểu đồ so sánh giá thuê:
    - df_line_chart: Dữ liệu biến động theo thời gian (month_year, price, zone)
    - df_bar_chart: Dữ liệu gộp theo mùa vụ (season_type, zone, price)
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    

    sns.lineplot(
        data=df_line_chart,
        x='month_year',
        y='price',
        hue='zone',      
        style='zone',    
        markers={'Trung tâm': 'o', 'Ngoại thành': 'X'}, 
        dashes=False,   
        markersize=8,    
        ax=axes[0]       
    )


    axes[0].set_title('Biến động giá trung vị theo tháng tại TP.HCM (2024 - 2025)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Giá thuê (Triệu/tháng)')
    axes[0].set_xlabel('Thời gian')
    axes[0].tick_params(axis='x', rotation=45) 
    axes[0].grid(True, linestyle='--', alpha=0.7) 


    sns.barplot(
        data=df_bar_chart,
        x='zone',
        y='price',
        hue='season_type', 
        palette='viridis', 
        ax=axes[1]         
    )


    axes[1].set_title('Chênh lệch giá thuê: Mùa Cao Điểm vs Bình Thường', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Giá trung vị (Triệu/tháng)')
    axes[1].set_xlabel('Khu vực')
    axes[1].legend(title='Thời điểm trong năm', loc='lower right') 
    axes[1].grid(axis='y', linestyle='--', alpha=0.5) 

    plt.tight_layout()
    plt.show()

def plot_price_main_vs_alley(
    df,
    main_street_label='Mặt tiền (Main Street)',
    alley_label='Hẻm (Alley)',
    figsize=(14, 7)
):

    pivot_table = (
        df.groupby(['district_clean', 'street_type'])['price_per_m2']
          .median()
          .reset_index()
    )

    order_list = (
        pivot_table[pivot_table['street_type'] == main_street_label]
        .sort_values('price_per_m2', ascending=False)['district_clean']
    )


    plt.figure(figsize=figsize)
    sns.barplot(
        data=pivot_table,
        x='district_clean',
        y='price_per_m2',
        hue='street_type',
        order=order_list,
        palette={
            alley_label: '#3498db',
            main_street_label: '#e74c3c'
        }
    )

    plt.title('So sánh giá Mặt Tiền vs Hẻm', fontsize=15, fontweight='bold')
    plt.ylabel('Giá trung vị (Triệu VND/m2)')
    plt.xlabel('Khu vực')
    plt.xticks(rotation=45)
    plt.legend(title='Loại hình')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_monthly_seasonality_compare(
    df: pd.DataFrame,
    price_col: str = "price",
    district_col: str = "district",
    month_col: str = "month",
    figsize=(13, 4.5),
):
    """
    Vẽ 2 biểu đồ so sánh mùa vụ theo tháng:
    (1) Median theo tháng trên toàn bộ tin
    (2) Median theo tháng sau khi lấy median theo quận rồi trung bình theo tháng
    """

    # 1) Median theo tháng (toàn bộ tin)
    monthly = (
        df.groupby(month_col)[price_col]
          .median()
          .reset_index(name="median_price")
          .sort_values(month_col)
    )

    # 2) Median theo từng (quận, tháng)
    district_month = (
        df.groupby([district_col, month_col])[price_col]
          .median()
          .reset_index(name="district_month_median")
    )

    # 3) Trung bình median theo quận cho mỗi tháng
    district_adjusted = (
        district_month
        .groupby(month_col)["district_month_median"]
        .mean()
        .reset_index(name="avg_district_median")
        .sort_values(month_col)
    )

    # --- Vẽ ---
    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        sharey=True,
        constrained_layout=True
    )

    # Cấu hình trục (không dùng for)
    axes[0].set_xticks(range(1, 13))
    axes[0].grid(True, which="major", linestyle="--", alpha=0.6)
    axes[0].tick_params(axis="both", labelsize=10)

    axes[1].set_xticks(range(1, 13))
    axes[1].grid(True, which="major", linestyle="--", alpha=0.6)
    axes[1].tick_params(axis="both", labelsize=10)

    # Plot 1: median toàn bộ
    axes[0].plot(
        monthly[month_col],
        monthly["median_price"],
        marker="o",
        linewidth=2,
        markersize=5
    )
    axes[0].set_title("Median theo tháng (Lấy toàn bộ tin)", fontsize=12, pad=8)
    axes[0].set_xlabel("Tháng", fontsize=11)
    axes[0].set_ylabel("Giá thuê trung vị (triệu/tháng)", fontsize=11)

    # Plot 2: median theo quận rồi trung bình
    axes[1].plot(
        district_adjusted[month_col],
        district_adjusted["avg_district_median"],
        marker="o",
        linewidth=2,
        markersize=5
    )
    axes[1].set_title("Median theo tháng (Lấy trung bình theo từng quận)", fontsize=12, pad=8)
    axes[1].set_xlabel("Tháng", fontsize=11)

    fig.suptitle(
        "Ảnh hưởng mùa vụ theo tháng của giá thuê phòng trọ (TP.HCM)",
        fontsize=13,
        y=1.05
    )

    plt.show()

    return monthly, district_adjusted, district_month

def plot_monthly_boxplot_seasonality(
    df: pd.DataFrame,
    price_col: str = "price",
    district_col: str = "district",
    month_col: str = "month",
    figsize=(14, 4.5),
):
    """
    Vẽ 2 boxplot so sánh phân bố giá theo tháng:
    (1) Toàn bộ tin
    (2) Median theo từng quận rồi xét phân bố theo tháng
    """

    # =========================
    # 1) Dữ liệu boxplot: toàn bộ tin theo tháng
    # =========================
    all_month_data = (
        df[[month_col, price_col]]
        .dropna()
        .groupby(month_col)[price_col]
        .apply(list)
        .reindex(range(1, 13))
        .fillna(value=pd.Series([[]] * 12))
        .tolist()
    )

    # =========================
    # 2) Dữ liệu boxplot: median theo (quận, tháng)
    # =========================
    district_month = (
        df
        .groupby([district_col, month_col])[price_col]
        .median()
        .reset_index(name="district_month_median")
    )

    district_month_data = (
        district_month
        .groupby(month_col)["district_month_median"]
        .apply(list)
        .reindex(range(1, 13))
        .fillna(value=pd.Series([[]] * 12))
        .tolist()
    )

    # =========================
    # 3) Vẽ boxplot
    # =========================
    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        sharey=True,
        constrained_layout=True
    )

    # Boxplot 1: toàn bộ tin
    axes[0].boxplot(
        all_month_data,
        labels=list(range(1, 13)),
        showfliers=False
    )
    axes[0].set_title("Phân bố giá theo tháng (Toàn bộ tin)", fontsize=12)
    axes[0].set_xlabel("Tháng")
    axes[0].set_ylabel("Giá thuê (triệu/tháng)")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.6)

    # Boxplot 2: median từng quận
    axes[1].boxplot(
        district_month_data,
        labels=list(range(1, 13)),
        showfliers=False
    )
    axes[1].set_title("Phân bố giá theo tháng (Median từng quận)", fontsize=12)
    axes[1].set_xlabel("Tháng")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.6)

    fig.suptitle(
        "Phân bố giá thuê phòng trọ theo tháng (Boxplot)",
        fontsize=13,
        y=1.05
    )

    plt.show()

    return district_month

def visualize_price_by_district(df, price_stats, x_limit=25):
    """
    Vẽ 2 biểu đồ: Boxplot (phân bố) và Barplot (giá trung bình) theo quận.
    
    Parameters:
    - df: DataFrame chứa dữ liệu gốc (df_clean).
    - price_stats: DataFrame thống kê đã sort (chứa index là tên quận và cột 'mean').
    - x_limit: Giới hạn trục X cho biểu đồ Boxplot (mặc định 25).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Boxplot - Phân bố giá
    sns.boxplot(data=df, x='price', y='district', 
                order=price_stats.index, color='lightblue', 
                showfliers=True, fliersize=3, ax=axes[0])
    
    axes[0].set_title('Phân bố giá thuê phòng trọ theo quận', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Giá thuê (Triệu đồng)', fontsize=11)
    axes[0].set_ylabel('Quận', fontsize=11)
    axes[0].set_xlim(0, x_limit)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)

    # 2. Biểu đồ cột - So sánh giá trung bình
    # Kiểm tra xem price_stats là Series hay DataFrame có cột 'mean'
    mean_values = price_stats['mean'] if 'mean' in price_stats.columns else price_stats
    
    axes[1].barh(price_stats.index, mean_values, color='steelblue', edgecolor='black')
    axes[1].set_title('So sánh giá trung bình theo quận', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Giá trung bình (Triệu đồng)', fontsize=11)
    axes[1].set_ylabel('Quận', fontsize=11)
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)

    # Thêm text hiển thị giá trị
    for i, (idx, val) in enumerate(mean_values.items()):
        axes[1].text(val + 0.2, i, f'{val:.1f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_amenity_charts(amenity_comparison, top_diff, top_k=8, figsize=(15, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Chart 1: So sánh trực tiếp
    top_amenities = top_diff.index[:top_k]
    data_plot = amenity_comparison.loc[top_amenities, ['Binh dan', 'Cao cap']]
    x = np.arange(len(top_amenities))
    width = 0.35

    axes[0].barh(
        x - width / 2,
        data_plot['Binh dan'],
        width,
        label='Bình dân',
        color='#e74c3c'
    )
    axes[0].barh(
        x + width / 2,
        data_plot['Cao cap'],
        width,
        label='Cao cấp',
        color='#2ecc71'
    )
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(top_amenities)
    axes[0].set_title('Tỷ lệ có tiện nghi: Bình dân vs Cao cấp')
    axes[0].legend()

    # Chart 2: Độ chênh lệch
    axes[1].barh(top_amenities, top_diff.values[:top_k], color='#3498db')
    axes[1].set_title('Mức độ chênh lệch (%)')

    for i, v in enumerate(top_diff.values[:top_k]):
        axes[1].text(v + 1, i, f'{v:.1f}%', va='center')

    plt.tight_layout()
    plt.show()



def plot_numerical_distributions(df, numerical_cols):
    """
    Vẽ biểu đồ Histogram và Boxplot cho các cột số.
    """
    n_cols = len(numerical_cols)
    fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(7 * n_cols, 8))
    fig.suptitle('Phân phối các biến định lượng (Numerical Columns)', fontsize=16)

    colors = ['skyblue', 'orange', 'green', 'purple'] 


    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, col in enumerate(numerical_cols):
        data_viz = df[df[col] < df[col].quantile(0.995)][col] 
        color = colors[i % len(colors)]


        sns.histplot(data_viz, kde=True, ax=axes[0, i], color=color, bins=30)
        axes[0, i].lines[-1].set_color('red')
        axes[0, i].set_title(f'Distribution of {col}')
        axes[0, i].axvline(data_viz.mean(), color='red', linestyle='--', label='Mean')
        axes[0, i].legend()

        sns.boxplot(x=data_viz, ax=axes[1, i], color=color)
        axes[1, i].set_title(f'Box Plot of {col}')

    plt.tight_layout()
    plt.show()

def plot_bar_chart(data, title, xlabel, ylabel, orientation='h', color='skyblue'):
    """
    Vẽ biểu đồ cột (ngang hoặc dọc).
    """
    plt.figure(figsize=(10, 6))
    
    if orientation == 'h':
        sns.barplot(x=data.values, y=data.index, color=color)
    else:
        sns.barplot(x=data.index, y=data.values, color=color)
        
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_utility_percentages(data, title):
    """
    Vẽ biểu đồ tỷ lệ phần trăm các tiện ích.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Utility', y='Percentage', data=data)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Loại tiện ích', fontsize=12)
    plt.ylabel('Tỷ lệ sở hữu (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_time_series(data, title, xlabel, ylabel, chart_type='line'):
    """
    Vẽ biểu đồ xu hướng theo thời gian (đường hoặc cột).
    """
    plt.figure(figsize=(10, 5))
    
    if chart_type == 'line':
        data.plot(kind='line', marker='o', color='teal', linewidth=2)
    else:
        data.plot(kind='bar', color='orange', alpha=0.8)
        
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_missing_heatmap(df):
    """
    Vẽ biểu đồ nhiệt (heatmap) để trực quan hóa vị trí dữ liệu thiếu.
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Bản đồ phân bố dữ liệu thiếu (Vàng = Thiếu)', fontsize=14)
    plt.show()

def plot_correlation_heatmap(
    corr_matrix,
    title="Correlation heatmap (numerical variables)",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    figsize=(9, 7),
    annotate=True,
    annot_fmt="{:.2f}",
    annot_fontsize=8,
    threshold_color=0.5,
    cbar_label="Correlation"
):
    """
    Plot a correlation heatmap with optional annotations.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix (square DataFrame).
    title : str
        Title of the plot.
    cmap : str
        Matplotlib colormap.
    vmin, vmax : float
        Color scale limits.
    figsize : tuple
        Figure size.
    annotate : bool
        Whether to annotate correlation values.
    annot_fmt : str
        Format for annotation text.
    annot_fontsize : int
        Font size for annotation.
    threshold_color : float
        Threshold to switch text color for readability.
    cbar_label : str
        Label for colorbar.
    """

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.grid(False)
    # Ticks & labels
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.index)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=90)

    ax.set_title(title, fontsize=14)

    # Annotate values
    if annotate:
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                value = corr_matrix.values[i, j]
                ax.text(
                    j, i, annot_fmt.format(value),
                    ha="center",
                    va="center",
                    color="black" if abs(value) < threshold_color else "white",
                    fontsize=annot_fontsize
                )

    plt.tight_layout()
    plt.show()

def plot_amenities_by_district(
    district_amenities_pct,
    amenity_cols,
    cols_per_row=3,
    figsize_per_col=(8, 5.5),
    dpi=120,
    cmap_name="tab20",
    title_prefix="Tỉ lệ tiện ích theo địa bàn",
):
    """
    Vẽ biểu đồ cột thể hiện tỷ lệ (%) phòng có từng tiện ích theo địa bàn.

    Tham số
    -------
    district_amenities_pct : pd.DataFrame
        Bảng % tiện ích theo địa bàn (index = địa bàn, columns = tiện ích).
    amenity_cols : list[str]
        Danh sách tiện ích cần vẽ (sẽ tự lọc những cột tồn tại).
    cols_per_row : int
        Số subplot trên mỗi hàng.
    figsize_per_col : tuple(float, float)
        Kích thước (width, height) cho mỗi subplot.
    dpi : int
        Độ phân giải hình vẽ.
    cmap_name : str
        Tên colormap matplotlib dùng để tô màu.
    title_prefix : str
        Tiêu đề chung cho toàn bộ figure.
    """

    # Cấu hình DPI cho hiển thị và lưu ảnh
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi

    # Chỉ giữ các tiện ích thực sự tồn tại trong bảng
    valid_cols = [
        col for col in amenity_cols
        if col in district_amenities_pct.columns
    ]

    num_cols = len(valid_cols)
    if num_cols == 0:
        print("Không có tiện ích hợp lệ để vẽ.")
        return

    # Tính số hàng subplot
    rows = int(np.ceil(num_cols / cols_per_row))

    # Lấy colormap (dùng modulo để tránh vượt giới hạn màu)
    cmap = plt.cm.get_cmap(cmap_name)

    # Tạo figure tổng
    plt.figure(
        figsize=(
            cols_per_row * figsize_per_col[0],
            rows * figsize_per_col[1]
        )
    )

    plt.suptitle(
        f"{title_prefix} (Top {len(district_amenities_pct)} địa bàn)",
        fontsize=18,
        weight="bold"
    )

    # Trục x là các địa bàn
    x = np.arange(len(district_amenities_pct.index))

    # Vẽ từng tiện ích
    for i, col in enumerate(valid_cols, 1):
        ax = plt.subplot(rows, cols_per_row, i)

        color = cmap(i % cmap.N)

        ax.bar(
            x,
            district_amenities_pct[col],
            color=color,
            edgecolor="black",
            linewidth=0.5
        )

        ax.set_title(
            col.replace("_", " ").title(),
            fontsize=13
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            district_amenities_pct.index,
            rotation=35,
            ha="right",
            fontsize=10
        )

        ax.set_ylabel("% phòng")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Chừa không gian cho tiêu đề chung
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

def plot_median_price_by_district(
    price_by_district,
    price_col="median",
    top_n=10,
    figsize=(10, 5),
    rotation=35
):
    """
    Vẽ biểu đồ cột thể hiện median giá phòng theo địa bàn.

    Tham số
    -------
    price_by_district : pd.DataFrame
        Bảng thống kê giá theo địa bàn (index = địa bàn).
    price_col : str
        Tên cột giá cần vẽ (mặc định là 'median').
    top_n : int
        Số địa bàn dùng để hiển thị trong tiêu đề.
    figsize : tuple
        Kích thước figure.
    rotation : int
        Góc xoay nhãn trục X.
    """

    # Lấy series median (đã được sort từ trước)
    median_sorted = price_by_district[price_col]

    plt.figure(figsize=figsize)

    plt.bar(
        median_sorted.index,
        median_sorted.values,
        edgecolor="black",
        linewidth=0.7,
    )

    plt.xticks(rotation=rotation, ha="right")
    plt.ylabel("Median price")
    plt.title(
        f"Median giá phòng theo địa bàn (Top {top_n} địa bàn nhiều tin nhất)"
    )

    plt.tight_layout()
    plt.show()

def plot_top_streets_by_median_price(
    price_by_address_street,
    district_col="address",
    street_col="street_name",
    median_col="median_price",
    top_n=20,
    figsize=(13, 8),
    bar_color="#4C72B0",
):
    """
    Vẽ biểu đồ barplot Top tuyến đường có median giá phòng cao nhất

    Tham số
    -------
    price_by_address_street : pd.DataFrame
        Bảng thống kê theo (district, street) đã sort sẵn.
    district_col : str
        Tên cột địa bàn.
    street_col : str
        Tên cột tuyến đường.
    median_col : str
        Tên cột median giá.
    top_n : int
        Số tuyến đường top cần hiển thị.
    figsize : tuple
        Kích thước figure.
    bar_color : str
        Màu cột.
    """

    # Lấy top N
    top_pairs = price_by_address_street.head(top_n).copy()

    # Tạo nhãn gộp: "Đường – Quận"
    top_pairs["label"] = (
        top_pairs[street_col].astype(str)
        + " – "
        + top_pairs[district_col].astype(str)
    )

    plt.figure(figsize=figsize)

    ax = sns.barplot(
        data=top_pairs,
        x=median_col,
        y="label",
        color=bar_color,
        edgecolor="black"
    )

    ax.set_title(
        "Top tuyến đường có median giá phòng cao nhất",
        fontsize=18,
        weight="bold",
        pad=12
    )
    ax.set_xlabel("Median giá phòng (triệu)", fontsize=13)
    ax.set_ylabel("Tuyến đường – Địa bàn", fontsize=13)

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

    # Hiển thị giá trị trên mỗi bar (KHÔNG for)
    ax.bar_label(
        ax.containers[0],
        fmt="%.2f",
        padding=4,
        fontsize=10,
        weight="bold"
    )

    plt.tight_layout()
    plt.show()

def plot_street_delta_vs_district(
    top_delta,
    district_col="address",
    street_col="street_name",
    delta_col="delta_vs_district",
    top_n=20,
    figsize=(15, 8),
    bar_color="#4C72B0",
):
    """
    Vẽ biểu đồ chênh lệch median giá tuyến đường so với median giá quận.

    Tham số
    -------
    top_delta : pd.DataFrame
        DataFrame đã có cột delta_vs_district và đã được sort.
    district_col : str
        Tên cột quận/địa bàn.
    street_col : str
        Tên cột tuyến đường.
    delta_col : str
        Tên cột chênh lệch giá.
    top_n : int
        Số tuyến đường hiển thị (dùng cho tiêu đề).
    figsize : tuple
        Kích thước figure.
    bar_color : str
        Màu cột.
    """

    df = top_delta.copy()

    # Tạo label gộp: "street – district"
    df["label"] = (
        df[street_col].astype(str)
        + " – "
        + df[district_col].astype(str)
    )

    plt.figure(figsize=figsize)

    ax = sns.barplot(
        data=df,
        x=delta_col,
        y="label",
        color=bar_color,
        edgecolor="black"
    )

    ax.set_title(
        "Chênh lệch median giá tuyến đường so với median giá quận",
        fontsize=18,
        weight="bold",
        pad=12
    )

    ax.set_xlabel("Chênh lệch (triệu)", fontsize=13)
    ax.set_ylabel("Tuyến đường – Địa bàn", fontsize=13)

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

    # Ghi số lên thanh (KHÔNG for)
    ax.bar_label(
        ax.containers[0],
        fmt="%.2f",
        padding=4,
        fontsize=10,
        weight="bold"
    )

    plt.tight_layout()
    plt.show()

def plot_median_price_yes_no_amenity(
    df_show,
    amenity_col="Tiện ích",
    col_no="Median (không)",
    col_yes="Median (có)",
    title="So sánh median giá phòng giữa nhóm CÓ và KHÔNG CÓ tiện ích",
    ylabel="Median giá (triệu)",
    figsize=(15, 7),
    colors=("#4C72B0", "#A6C8E0"),
    label_threshold=0.3,
    label_offset=0.03,
):
    # Chuẩn bị dữ liệu
    df_plot = df_show[[amenity_col, col_no, col_yes]].copy()

    df_melt = df_plot.melt(
        id_vars=amenity_col,
        value_vars=[col_no, col_yes],
        var_name="Nhóm",
        value_name="Median Giá"
    )

    # Vẽ
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df_melt,
        x=amenity_col,
        y="Median Giá",
        hue="Nhóm",
        palette=list(colors),
        edgecolor="black"
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(amenity_col, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Ghi nhãn lên bar: chỉ hiện khi > threshold
    # Seaborn hue=2 nhóm => thường có 2 containers
    if len(ax.containers) >= 2:
        c0, c1 = ax.containers[0], ax.containers[1]

        v0 = np.asarray(getattr(c0, "datavalues", []), dtype=float)
        v1 = np.asarray(getattr(c1, "datavalues", []), dtype=float)

        labels0 = np.where(v0 > label_threshold, np.char.mod("%.2f", v0), "")
        labels1 = np.where(v1 > label_threshold, np.char.mod("%.2f", v1), "")

        ax.bar_label(c0, labels=labels0, padding=3, fontsize=9)
        ax.bar_label(c1, labels=labels1, padding=3, fontsize=9)

        # Nếu muốn đẩy label lên cao hơn chút (giống bạn h + 0.03)
        # bar_label dùng padding theo points; label_offset giữ để bạn tinh chỉnh nếu cần.
        # (Nếu bạn muốn offset đúng theo data-unit, nói mình chỉnh bản matplotlib thuần.)

    # Legend
    ax.legend(
        title="Nhóm",
        fontsize=10,
        title_fontsize=11,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        facecolor="white",
        edgecolor="black"
    )

    plt.tight_layout()
    plt.show()