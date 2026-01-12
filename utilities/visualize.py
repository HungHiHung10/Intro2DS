import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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