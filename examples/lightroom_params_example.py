"""
Lightroom参数映射使用示例

本脚本演示如何使用Lightroom参数映射功能
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from config import cfg
from isp.filters import *
from isp_blocks import ISPBlocks
from lightroom_params import (
    LightroomParamConverter,
    save_params_lightroom,
    load_params_lightroom,
    print_params_comparison,
    LIGHTROOM_RANGES
)


def example_1_convert_params():
    """示例1：转换参数并打印对比"""
    print("\n" + "="*80)
    print("示例1：参数转换和对比")
    print("="*80)
    
    # 初始化ISP块
    cfg.custom_isp = [
        ExposureFilter, 
        ImprovedWhiteBalanceFilter, 
        SaturationFilter, 
        ContrastFilter,
        HighlightFilter,
        ShadowFilter,
        SharpenFilter
    ]
    isp_blocks = ISPBlocks(cfg, is_blackbox=True)
    isp_blocks.init_filters(cfg.custom_isp)
    
    # 模拟模型输出的参数（内部格式，通常在[-1, 1]之间或特定范围）
    params_list = [
        torch.tensor([[0.06]], dtype=torch.float32),   # Exposure
        torch.tensor([[0.43, -0.01, 0.13]], dtype=torch.float32),  # WB
        torch.tensor([[0.08]], dtype=torch.float32),   # Saturation
        torch.tensor([[-0.06]], dtype=torch.float32),  # Contrast
        torch.tensor([[0.22]], dtype=torch.float32),   # Highlight
        torch.tensor([[0.004]], dtype=torch.float32),  # Shadow
        torch.tensor([[0.03]], dtype=torch.float32),   # Sharpen
    ]
    
    # 打印参数对比
    print_params_comparison(params_list, isp_blocks)
    

def example_2_save_and_load():
    """示例2：保存和加载Lightroom参数"""
    print("\n" + "="*80)
    print("示例2：保存和加载Lightroom参数")
    print("="*80)
    
    # 初始化
    cfg.custom_isp = [ExposureFilter, SaturationFilter, ContrastFilter]
    isp_blocks = ISPBlocks(cfg, is_blackbox=True)
    isp_blocks.init_filters(cfg.custom_isp)
    
    # 原始参数
    params_list = [
        torch.tensor([[0.5]], dtype=torch.float32),   # Exposure: 内部0.5
        torch.tensor([[0.3]], dtype=torch.float32),   # Saturation: 内部0.3
        torch.tensor([[-0.2]], dtype=torch.float32),  # Contrast: 内部-0.2
    ]
    
    print("\n原始参数:")
    for i, (filter_obj, param) in enumerate(zip(isp_blocks.filters, params_list)):
        print(f"  {filter_obj.__class__.__name__}: {param[0].tolist()}")
    
    # 保存为Lightroom格式
    save_path = '/tmp/test_params_lightroom.json'
    save_params_lightroom(params_list, isp_blocks, save_path, include_internal=True)
    print(f"\n✓ 参数已保存到: {save_path}")
    
    # 查看保存的内容
    import json
    with open(save_path, 'r') as f:
        saved_data = json.load(f)
    
    print("\nLightroom格式参数:")
    for filter_name, data in saved_data['lightroom_params'].items():
        print(f"\n  {data.get('display_name', filter_name)}:")
        for param_name, value in data['parameters'].items():
            unit = data.get('unit', '')
            print(f"    {param_name}: {value:.2f} {unit}")
    
    # 重新加载
    loaded_params = load_params_lightroom(save_path, isp_blocks)
    print("\n✓ 参数已重新加载")
    
    print("\n重新加载的参数:")
    for i, (filter_obj, param) in enumerate(zip(isp_blocks.filters, loaded_params)):
        print(f"  {filter_obj.__class__.__name__}: {param[0].tolist()}")
    
    # 验证一致性
    print("\n参数一致性检查:")
    for i, (orig, loaded) in enumerate(zip(params_list, loaded_params)):
        diff = torch.abs(orig - loaded).max().item()
        status = "✓ 一致" if diff < 1e-5 else f"✗ 差异: {diff}"
        print(f"  Filter {i}: {status}")


def example_3_manual_edit():
    """示例3：手动编辑Lightroom参数"""
    print("\n" + "="*80)
    print("示例3：手动编辑和应用Lightroom参数")
    print("="*80)
    
    # 初始化
    cfg.custom_isp = [ExposureFilter, SaturationFilter]
    isp_blocks = ISPBlocks(cfg, is_blackbox=True)
    isp_blocks.init_filters(cfg.custom_isp)
    
    # 创建一个Lightroom格式的参数字典
    lightroom_params = {
        "ExposureFilter": {
            "parameters": {
                "exposure": 1.5  # +1.5 EV
            }
        },
        "SaturationFilter": {
            "parameters": {
                "saturation": 25.0  # +25
            }
        }
    }
    
    print("\n手动设置的Lightroom参数:")
    print(f"  Exposure: +1.5 EV")
    print(f"  Saturation: +25")
    
    # 转换为内部参数
    converter = LightroomParamConverter(isp_blocks)
    params_list = converter.lightroom_to_internal(lightroom_params)
    
    print("\n转换后的内部参数:")
    for filter_obj, param in zip(isp_blocks.filters, params_list):
        print(f"  {filter_obj.__class__.__name__}: {param[0].tolist()}")
    
    # 这些内部参数现在可以直接用于图像处理
    print("\n✓ 参数已准备好，可以应用到图像处理")


def example_4_show_ranges():
    """示例4：显示所有支持的Lightroom参数范围"""
    print("\n" + "="*80)
    print("示例4：支持的Lightroom参数范围")
    print("="*80)
    
    for filter_name, config in LIGHTROOM_RANGES.items():
        print(f"\n【{config['display_name']}】({filter_name})")
        for param_name, param_range in config.items():
            if param_name not in ['display_name', 'unit']:
                print(f"  {param_name}: {param_range[0]} 到 {param_range[1]}")


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("Lightroom参数映射系统 - 使用示例")
    print("="*80)
    
    # 运行所有示例
    example_1_convert_params()
    example_2_save_and_load()
    example_3_manual_edit()
    example_4_show_ranges()
    
    print("\n" + "="*80)
    print("所有示例运行完成！")
    print("="*80)
    print("\n提示：")
    print("1. 查看保存的JSON文件以了解Lightroom格式")
    print("2. 可以手动编辑JSON中的Lightroom参数并重新加载")
    print("3. 参考 LIGHTROOM_PARAMS_GUIDE_CN.md 获取完整文档")
    print()


if __name__ == "__main__":
    main()
