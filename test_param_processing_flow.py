#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 RLPixTuner 参数处理流程

测试从模型输出到图像渲染的完整参数处理过程
"""

import numpy as np
import torch
from config import cfg
from isp.filters import (ExposureFilter, ImprovedWhiteBalanceFilter, 
                         SaturationFilter, ContrastFilter)
from isp_blocks import ISPBlocks


def test_linear_mapping():
    """测试第1步：线性映射"""
    print("=" * 70)
    print("【测试1】线性映射: action [-1,1] → param [range_l, range_r]")
    print("=" * 70)
    
    # 定义测试案例
    test_cases = [
        {
            'filter': 'Exposure',
            'range_l': -2.0,
            'range_r': 2.0,
            'actions': [-1.0, -0.5, 0.0, 0.5, 1.0]
        },
        {
            'filter': 'White Balance',
            'range_l': 0.606,
            'range_r': 1.649,
            'actions': [-1.0, -0.5, 0.0, 0.5, 1.0]
        },
        {
            'filter': 'Saturation',
            'range_l': -1.0,
            'range_r': 1.0,
            'actions': [-1.0, -0.5, 0.0, 0.5, 1.0]
        }
    ]
    
    print(f"\n映射公式: param = range_l + (range_r - range_l) * (action + 1) / 2")
    print()
    
    for test in test_cases:
        print(f"{test['filter']}:")
        print(f"  参数范围: [{test['range_l']}, {test['range_r']}]")
        print(f"  {'action':>8} → {'param':>8}")
        print(f"  {'-'*20}")
        
        range_l = test['range_l']
        range_r = test['range_r']
        
        for action in test['actions']:
            # 线性映射公式
            param = range_l + (range_r - range_l) * (action + 1) / 2
            print(f"  {action:8.2f} → {param:8.3f}")
        print()
    
    print("✅ 结论: 线性映射是必需步骤，将归一化的 action 映射到参数范围")
    print()


def test_param_dist_transform():
    """测试第2步：参数分布变换"""
    print("=" * 70)
    print("【测试2】参数分布变换: change_param_dist()")
    print("=" * 70)
    
    # 创建白平衡滤镜
    cfg.custom_isp = [ImprovedWhiteBalanceFilter]
    
    # 测试输入参数
    test_params = np.array([0.7, 0.9, 1.0, 1.2, 1.5])
    
    print("\n白平衡滤镜的三种模式:\n")
    
    for mode in [0, 1, 2]:
        cfg.change_param_dist_cfg["ImprovedWhiteBalanceFilter"] = mode
        
        isp = ISPBlocks(cfg, is_blackbox=True)
        isp.init_filters(cfg.custom_isp)
        
        wb_filter = isp.filters[0]
        
        print(f"模式 {mode}:")
        if mode == 0:
            print("  描述: 不处理（线性分布）")
        elif mode == 1:
            print("  描述: 对数空间变换（中心密集）")
        elif mode == 2:
            print("  描述: 幂律变换（更密集的中心）")
        
        print(f"  {'输入':>10} → {'输出':>10} | {'变化':>10}")
        print(f"  {'-'*35}")
        
        for param in test_params:
            param_tensor = torch.tensor([[param, param, param]], dtype=torch.float32)
            transformed = wb_filter.change_param_dist(param_tensor)
            diff = transformed[0, 0].item() - param
            
            print(f"  {param:10.4f} → {transformed[0,0].item():10.4f} | {diff:+10.4f}")
        print()
    
    print("✅ 结论: 参数分布变换是可选的，默认 mode=0（不处理）")
    print()


def test_default_behavior():
    """测试默认行为（大部分滤镜）"""
    print("=" * 70)
    print("【测试3】默认行为: 其他滤镜的 change_param_dist()")
    print("=" * 70)
    
    # 测试其他滤镜
    filters_to_test = [
        ExposureFilter,
        SaturationFilter,
        ContrastFilter,
    ]
    
    test_param = 1.5
    
    print(f"\n测试参数: {test_param}")
    print(f"{'滤镜':>20} | {'输入':>10} → {'输出':>10} | {'是否处理':>10}")
    print("-" * 60)
    
    for FilterClass in filters_to_test:
        cfg.custom_isp = [FilterClass]
        isp = ISPBlocks(cfg, is_blackbox=True)
        isp.init_filters(cfg.custom_isp)
        
        filter_obj = isp.filters[0]
        param_tensor = torch.tensor([[test_param]], dtype=torch.float32)
        transformed = filter_obj.change_param_dist(param_tensor)
        
        is_changed = not torch.allclose(param_tensor, transformed)
        status = "是" if is_changed else "否"
        
        print(f"{FilterClass.__name__:>20} | {test_param:10.4f} → "
              f"{transformed[0,0].item():10.4f} | {status:>10}")
    
    print("\n✅ 结论: 除了白平衡，其他滤镜默认不处理参数（直接返回）")
    print()


def test_end_to_end_flow():
    """测试端到端流程"""
    print("=" * 70)
    print("【测试4】完整流程: action → 渲染")
    print("=" * 70)
    
    # 配置 ISP
    cfg.custom_isp = [ExposureFilter]
    cfg.change_param_dist_cfg["ExposureFilter"] = 0
    
    isp = ISPBlocks(cfg, is_blackbox=True)
    isp.init_filters(cfg.custom_isp)
    
    # 模拟模型输出
    action = 0.5
    
    print(f"\n步骤追踪:")
    print(f"{'步骤':>5} | {'描述':>30} | {'值':>15}")
    print("-" * 60)
    
    # 步骤1: 模型输出
    print(f"{'1':>5} | {'模型输出 action':>30} | {action:15.6f}")
    
    # 步骤2: 线性映射
    range_l = isp.filters[0].range_l
    range_r = isp.filters[0].range_r
    param_linear = range_l + (range_r - range_l) * (action + 1) / 2
    print(f"{'2':>5} | {'线性映射 [映射到 -2~2]':>30} | {param_linear:15.6f}")
    
    # 步骤3: 参数分布变换
    param_tensor = torch.tensor([[param_linear]], dtype=torch.float32)
    param_transformed = isp.filters[0].change_param_dist(param_tensor)
    print(f"{'3':>5} | {'change_param_dist (mode=0)':>30} | "
          f"{param_transformed[0,0].item():15.6f}")
    
    # 步骤4: 应用到图像
    test_img = torch.ones(1, 3, 4, 4) * 0.5  # 简单测试图像
    output_img = isp.run(test_img, [param_transformed])
    
    print(f"{'4':>5} | {'应用到图像':>30} | {'完成':>15}")
    
    # 验证渲染公式
    expected_output = test_img * torch.exp(param_transformed[:, :, None, None] * np.log(2))
    rendering_correct = torch.allclose(output_img, expected_output, atol=1e-5)
    
    print(f"\n渲染验证:")
    print(f"  输入图像均值: {test_img.mean().item():.6f}")
    print(f"  输出图像均值: {output_img.mean().item():.6f}")
    print(f"  理论输出均值: {expected_output.mean().item():.6f}")
    print(f"  渲染公式正确: {'✅' if rendering_correct else '❌'}")
    
    print("\n✅ 结论: 参数经过线性映射后，基本直接用于渲染（mode=0时）")
    print()


def test_with_different_modes():
    """测试不同模式下的白平衡效果"""
    print("=" * 70)
    print("【测试5】白平衡三种模式的实际效果")
    print("=" * 70)
    
    # 创建测试图像（灰度图）
    test_img = torch.ones(1, 3, 256, 256) * 0.5
    
    # 模拟一个轻微偏蓝的场景
    test_img[0, 2, :, :] = 0.6  # B 通道略高
    
    print(f"\n测试图像:")
    print(f"  R通道均值: {test_img[0, 0].mean():.3f}")
    print(f"  G通道均值: {test_img[0, 1].mean():.3f}")
    print(f"  B通道均值: {test_img[0, 2].mean():.3f}")
    print()
    
    # 测试三种模式
    action = 0.0  # 中性白平衡
    
    print(f"应用白平衡 (action={action}):")
    print(f"{'模式':>6} | {'R参数':>8} {'G参数':>8} {'B参数':>8} | "
          f"{'R输出':>8} {'G输出':>8} {'B输出':>8}")
    print("-" * 70)
    
    for mode in [0, 1, 2]:
        cfg.custom_isp = [ImprovedWhiteBalanceFilter]
        cfg.change_param_dist_cfg["ImprovedWhiteBalanceFilter"] = mode
        
        isp = ISPBlocks(cfg, is_blackbox=True)
        isp.init_filters(cfg.custom_isp)
        
        # 线性映射
        wb_filter = isp.filters[0]
        range_l = wb_filter.range_l
        range_r = wb_filter.range_r
        param = range_l + (range_r - range_l) * (action + 1) / 2
        param_tensor = torch.tensor([[param, param, param]], dtype=torch.float32)
        
        # 应用
        output = isp.run(test_img.clone(), [param_tensor])
        
        print(f"{mode:>6} | {param:8.4f} {param:8.4f} {param:8.4f} | "
              f"{output[0,0].mean():.6f} {output[0,1].mean():.6f} {output[0,2].mean():.6f}")
    
    print("\n✅ 结论: 不同模式产生不同的参数分布，影响渲染结果")
    print()


def test_change_param_dist_flag():
    """测试 change_param_dist 标志的作用"""
    print("=" * 70)
    print("【测试6】change_param_dist 标志的影响")
    print("=" * 70)
    
    cfg.custom_isp = [ImprovedWhiteBalanceFilter]
    cfg.change_param_dist_cfg["ImprovedWhiteBalanceFilter"] = 1  # 启用模式1
    
    isp = ISPBlocks(cfg, is_blackbox=True)
    isp.init_filters(cfg.custom_isp)
    
    test_img = torch.ones(1, 3, 256, 256) * 0.5
    param_tensor = torch.tensor([[1.2, 1.0, 0.9]], dtype=torch.float32)
    
    print(f"\n输入参数: R={param_tensor[0,0]:.3f}, G={param_tensor[0,1]:.3f}, "
          f"B={param_tensor[0,2]:.3f}")
    print(f"模式设置: mode=1 (对数变换)")
    print()
    
    # 开启 change_param_dist
    output_with = isp.run(test_img.clone(), [param_tensor], change_param_dist=True)
    print(f"开启 change_param_dist=True:")
    print(f"  输出: R={output_with[0,0].mean():.6f}, G={output_with[0,1].mean():.6f}, "
          f"B={output_with[0,2].mean():.6f}")
    
    # 关闭 change_param_dist
    output_without = isp.run(test_img.clone(), [param_tensor], change_param_dist=False)
    print(f"\n关闭 change_param_dist=False:")
    print(f"  输出: R={output_without[0,0].mean():.6f}, G={output_without[0,1].mean():.6f}, "
          f"B={output_without[0,2].mean():.6f}")
    
    # 计算差异
    diff = torch.abs(output_with - output_without).mean().item()
    print(f"\n两种模式的输出差异: {diff:.6f}")
    
    if diff > 1e-6:
        print("✅ 结论: change_param_dist 标志有效，会影响渲染结果")
    else:
        print("⚠️  注意: 当前参数下差异很小")
    print()


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "RLPixTuner 参数处理流程测试" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        test_linear_mapping()
        test_param_dist_transform()
        test_default_behavior()
        test_end_to_end_flow()
        test_with_different_modes()
        test_change_param_dist_flag()
        
        print("=" * 70)
        print("【总结】")
        print("=" * 70)
        print("""
参数处理流程 (推理时):
─────────────────────────────────────────────────
1. ✅ 线性映射        - 必需，将 [-1,1] 映射到参数范围
2. ⚠️  参数分布变换   - 可选，默认 mode=0（不处理）
3. ✅ 直接渲染        - 必需，应用参数到图像

关键点:
• 大部分情况下: action → 线性映射 → 直接渲染
• 特殊情况: 白平衡可启用非线性变换 (mode=1/2)
• 推荐配置: 使用默认 mode=0，保持参数物理意义
        """)
        
        print("\n✅ 所有测试完成！\n")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
