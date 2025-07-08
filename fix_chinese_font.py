"""
修复matplotlib中文字体显示问题
运行此脚本来配置中文字体
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import platform
import os

def setup_chinese_font():
    """设置中文字体"""
    
    # 检测操作系统
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # macOS常用中文字体
        chinese_fonts = [
            'Arial Unicode MS',
            'Heiti TC',
            'STHeiti',
            'SimHei',
            'Hiragino Sans GB'
        ]
    elif system == "Windows":  # Windows
        chinese_fonts = [
            'SimHei',
            'Microsoft YaHei',
            'KaiTi',
            'SimSun'
        ]
    else:  # Linux
        chinese_fonts = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'DejaVu Sans'
        ]
    
    # 尝试设置字体
    for font in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            # 测试字体
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, '中文测试 Chinese Test', 
                   fontsize=16, ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('字体测试 Font Test')
            plt.savefig('font_test.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 成功设置中文字体: {font}")
            print("字体测试图片已保存为 font_test.png")
            return True
            
        except Exception as e:
            print(f"❌ 字体 {font} 设置失败: {e}")
            continue
    
    print("⚠️ 未找到可用的中文字体，使用默认设置")
    plt.rcParams['axes.unicode_minus'] = False
    return False

def get_system_fonts():
    """获取系统可用字体列表"""
    from matplotlib.font_manager import get_font_names
    
    fonts = get_font_names()
    chinese_fonts = [font for font in fonts if any(
        keyword in font.lower() for keyword in 
        ['chinese', 'simhei', 'simsun', 'heiti', 'kaiti', 'microsoft', 'wenquanyi']
    )]
    
    print("系统中可用的中文字体:")
    for font in chinese_fonts[:10]:  # 只显示前10个
        print(f"  - {font}")
    
    if not chinese_fonts:
        print("  未找到明显的中文字体")
    
    return chinese_fonts

def create_font_config_file():
    """创建字体配置文件"""
    config_content = '''
# 在每个Python文件开头添加这些配置代码

import matplotlib.pyplot as plt
import matplotlib
import platform

# 设置中文字体
system = platform.system()
if system == "Darwin":  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
elif system == "Windows":  # Windows
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
'''
    
    with open('font_config.txt', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✅ 字体配置文件已创建: font_config.txt")

if __name__ == "__main__":
    print("=== 修复matplotlib中文字体显示问题 ===\n")
    
    print("1. 检查系统字体...")
    get_system_fonts()
    
    print("\n2. 设置中文字体...")
    success = setup_chinese_font()
    
    print("\n3. 创建配置文件...")
    create_font_config_file()
    
    print("\n=== 修复完成 ===")
    if success:
        print("✅ 中文字体配置成功！")
        print("所有图表中的中文现在应该能正常显示。")
    else:
        print("⚠️ 中文字体配置可能不完整。")
        print("建议手动安装中文字体或使用英文标题。")
    
    print("\n使用建议:")
    print("1. 如果中文仍无法显示，请安装系统中文字体")
    print("2. 可以将图表标题改为英文避免显示问题")
    print("3. 参考 font_config.txt 文件中的配置代码")