import matplotlib.pyplot as plt
import seaborn as sns
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

