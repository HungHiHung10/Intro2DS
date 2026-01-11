import matplotlib.pyplot as plt
import seaborn as sns

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