import matplotlib.pyplot as plt
import seaborn as sns
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