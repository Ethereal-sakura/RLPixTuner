#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 RLPixTuner 模型参数分度值

验证模型输出的参数精度和分度值
"""

import numpy as np
import torch
from config import cfg
from isp.filters import (ExposureFilter, ImprovedWhiteBalanceFilter, 
                         SaturationFilter, ContrastFilter,
                         HighlightFilter, ShadowFilter, SharpenFilter)
from isp_blocks import ISPBlocks


def test_float32_precision():
    """测试 float32 的精度"""
    print("=" * 70)
    print("【1】Float32 精度测试")
    print("=" * 70)
    
    epsilon = np.finfo(np.float32).eps
    print(f"float32 机器精度 (ε): {epsilon:.15e}")
    print(f"动作空间范围: [-1, 1]")
    print(f"理论最小分度值: {epsilon * 2:.15e}")
    print()


def test_action_to_param_conversion():
    """测试动作到参数的转换"""
    print("=" * 70)
    print("【2】动作到参数转换测试")
    print("=" * 70)
    
    # 配置 ISP
    cfg.custom_isp = [ExposureFilter, ImprovedWhiteBalanceFilter, SaturationFilter]
    isp_blocks = ISPBlocks(cfg, is_blackbox=True)
    isp_blocks.init_filters(cfg.custom_isp)
    
    # 测试不同的动作值
    test_actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for filter_obj in isp_blocks.filters:
        print(f"\n{filter_obj.__class__.__name__}:")
        print(f"  参数范围: [{filter_obj.range_l}, {filter_obj.range_r}]")
        print(f"  范围宽度: {filter_obj.range_r - filter_obj.range_l}")
        
        range_width = filter_obj.range_r - filter_obj.range_l
        theoretical_resolution = 2.4e-7 * range_width
        print(f"  理论分度值: {theoretical_resolution:.10e}")
        
        print(f"\n  动作 → 参数转换:")
        for action in test_actions:
            # 转换公式: para = range_l + (range_r - range_l) * (action + 1) / 2
            param = filter_obj.range_l + range_width * (action + 1) / 2
            print(f"    action={action:5.1f} → param={param:8.5f}")
    print()


def test_parameter_precision_impact():
    """测试参数精度对图像的影响"""
    print("=" * 70)
    print("【3】参数精度对图像影响测试")
    print("=" * 70)
    
    # 创建测试图像
    test_image = torch.rand(1, 3, 256, 256, dtype=torch.float32)
    print(f"测试图像: shape={test_image.shape}, dtype={test_image.dtype}")
    print(f"像素范围: [{test_image.min():.3f}, {test_image.max():.3f}]")
    print()
    
    # 测试曝光参数的不同精度
    cfg.custom_isp = [ExposureFilter]
    isp = ISPBlocks(cfg, is_blackbox=True)
    isp.init_filters(cfg.custom_isp)
    
    base_exposure = 0.0
    test_precisions = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    print("曝光参数微小变化对图像的影响:")
    print(f"{'参数变化':>15} | {'平均像素差':>15} | {'最大像素差':>15} | {'可见性':>10}")
    print("-" * 70)
    
    # 基准图像
    base_param = torch.tensor([[base_exposure]], dtype=torch.float32)
    base_output = isp.run(test_image.clone(), [base_param])
    
    for precision in test_precisions:
        param = torch.tensor([[base_exposure + precision]], dtype=torch.float32)
        output = isp.run(test_image.clone(), [param])
        
        diff = output - base_output
        mean_diff = torch.mean(torch.abs(diff)).item()
        max_diff = torch.max(torch.abs(diff)).item()
        
        # 判断可见性（8-bit 图像，1/255 ≈ 0.004）
        visibility = "不可见" if mean_diff < 0.001 else ("微弱" if mean_diff < 0.004 else "可见")
        
        print(f"{precision:15.0e} | {mean_diff:15.6f} | {max_diff:15.6f} | {visibility:>10}")
    print()


def test_network_output_precision():
    """测试网络实际输出精度"""
    print("=" * 70)
    print("【4】神经网络输出精度分析")
    print("=" * 70)
    
    print("TD3 模型特性:")
    print("  • Actor 网络输出层: tanh 激活")
    print("  • 输出范围: [-1, 1]")
    print("  • 数据类型: float32")
    print("  • 批归一化: 可能影响精度")
    print()
    
    # 模拟网络输出
    simulated_actions = np.array([
        0.123456789012345,  # 高精度输入
        0.5,                 # 简单值
        -0.987654321098765, # 负值高精度
    ], dtype=np.float32)
    
    print("模拟网络输出:")
    for i, action in enumerate(simulated_actions):
        print(f"  Action[{i}]: {action:.15f}")
        # 转换为参数（以曝光为例，范围 [-2, 2]）
        param = -2.0 + 4.0 * (action + 1) / 2
        print(f"    → 曝光参数: {param:.15f}")
    print()


def test_quantization_effect():
    """测试参数量化的影响"""
    print("=" * 70)
    print("【5】参数量化影响测试")
    print("=" * 70)
    
    # 生成随机参数
    np.random.seed(42)
    original_params = np.random.uniform(-1, 1, size=10).astype(np.float32)
    
    quantization_levels = [1e-2, 1e-3, 1e-4, 1e-5]
    
    print(f"{'量化精度':>15} | {'平均误差':>15} | {'最大误差':>15} | {'相对误差':>15}")
    print("-" * 70)
    
    for precision in quantization_levels:
        quantized = np.round(original_params / precision) * precision
        error = np.abs(quantized - original_params)
        mean_error = np.mean(error)
        max_error = np.max(error)
        relative_error = mean_error / 2.0  # 相对于动作空间宽度 2.0
        
        print(f"{precision:15.0e} | {mean_error:15.8f} | {max_error:15.8f} | {relative_error*100:14.6f}%")
    print()


def test_practical_resolution():
    """测试实际应用中的推荐分度值"""
    print("=" * 70)
    print("【6】实际应用推荐")
    print("=" * 70)
    
    recommendations = {
        'Exposure': {
            'range': [-2.0, 2.0],
            'theoretical': 9.6e-7,
            'network': 1e-5,
            'display': 0.004,
            'perception': 0.02,
            'recommended': 0.01,
        },
        'White Balance': {
            'range': [0.606, 1.649],
            'theoretical': 2.5e-7,
            'network': 1e-5,
            'display': 0.004,
            'perception': 0.01,
            'recommended': 0.01,
        },
        'Saturation': {
            'range': [-1.0, 1.0],
            'theoretical': 4.8e-7,
            'network': 1e-5,
            'display': 0.004,
            'perception': 0.03,
            'recommended': 0.01,
        },
    }
    
    for filter_name, specs in recommendations.items():
        print(f"\n{filter_name}:")
        print(f"  参数范围: {specs['range']}")
        print(f"  理论分度值: {specs['theoretical']:.2e}")
        print(f"  网络精度限制: {specs['network']:.2e}")
        print(f"  显示精度限制: {specs['display']:.3f}")
        print(f"  感知精度限制: {specs['perception']:.3f}")
        print(f"  推荐分度值: {specs['recommended']:.3f} ⭐")
    
    print("\n" + "=" * 70)
    print("总结:")
    print("  • 理论分度值: ~2.4×10⁻⁷ (float32 精度)")
    print("  • 实际有效分度值: ~0.01 - 0.05")
    print("  • 推荐使用分度值: 0.01 ⭐")
    print("=" * 70)


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "RLPixTuner 参数分度值测试" + " " * 29 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        test_float32_precision()
        test_action_to_param_conversion()
        test_parameter_precision_impact()
        test_network_output_precision()
        test_quantization_effect()
        test_practical_resolution()
        
        print("\n✅ 所有测试完成！\n")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
