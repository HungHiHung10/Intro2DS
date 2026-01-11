import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

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
    sns.barplot(x='Utility', y='Percentage', data=data, palette='viridis')
    
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

    