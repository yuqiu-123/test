"""
搜索树不同深度指标对比表
用于对比搜索树在不同深度时提取的chunk数、节点总数、CRR等指标
"""

import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def create_comparison_table():
    """
    创建搜索树不同深度指标对比表
    数据可以手动填充
    """
    # 定义表格数据（示例数据，可根据实际情况修改）
    data = {
        '深度': [4, 5, 6, 7],
        '节点总数': [21, 41, 70, 82],  # 请填入实际数据
        '提取的Chunk数': [35, 48, 67, 70],  # 请填入实际数据
        '提取率': ["35%", "48%", "67%", "70%"],  # 请填入实际数据（0-1之间的小数）
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    return df


def display_table(df):
    """
    以美观的格式显示表格
    """
    print("\n" + "="*80)
    print("搜索树不同深度指标对比表")
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False, floatfmt='.4f'))
    print("="*80 + "\n")


def save_to_csv(df, filename='search_tree_comparison.csv'):
    """
    将表格保存为CSV文件
    """
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"表格已保存到: {filename}")


def save_to_excel(df, filename='search_tree_comparison.xlsx'):
    """
    将表格保存为Excel文件
    """
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"表格已保存到: {filename}")


def save_to_png(df, filename='search_tree_comparison.png', dpi=300):
    """
    将表格绘制成PNG图片文件
    """
    # 创建图形和轴，为标题留出空间
    fig, ax = plt.subplots(figsize=(14, max(7, len(df) * 1.0 + 2)))  # 增大图形尺寸以适应更大字体
    ax.axis('tight')
    ax.axis('off')
    
    # 添加标题
    fig.suptitle('搜索树不同深度指标对比表', fontsize=20, fontweight='bold', y=0.95)
    
    # 创建表格，为标题留出空间
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 0.9])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(20)  # 增大字体大小
    table.scale(1, 2.5)  # 相应调整行高
    
    # 设置表头样式
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置数据行样式（交替颜色）
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
            else:
                table[(i, j)].set_facecolor('white')
            table[(i, j)].set_edgecolor('#CCCCCC')
    
    # 设置边框
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(1)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"表格图片已保存到: {filename}")


if __name__ == '__main__':
    # 创建对比表
    comparison_df = create_comparison_table()
    
    # 显示表格
    display_table(comparison_df)
    
    # 保存为PNG图片
    save_to_png(comparison_df)
    
    # 保存为CSV（可选）
    # save_to_csv(comparison_df)
    
    # 保存为Excel（可选，需要安装openpyxl: pip install openpyxl）
    # save_to_excel(comparison_df)
    
    print("\n提示：请在代码中修改数据后重新运行此脚本。")
    print("数据字段说明：")
    print("  - 深度: 搜索树的深度层级")
    print("  - 提取的Chunk数: 在该深度下提取到的chunk总数")
    print("  - 节点总数: 搜索树中该深度及之前所有深度的节点总数")
    print("  - CRR: 累积召回率 (Cumulative Recall Rate)")
    print("  - 平均节点扩展数: 每个节点的平均子节点数（可选）")
    print("  - 搜索时间: 达到该深度所需的搜索时间（可选）")

