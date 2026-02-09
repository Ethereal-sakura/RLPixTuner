#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单张图片推理脚本
使用 RLPixTuner 对单张输入图片进行照片后期处理调优
"""

import os
import sys
import argparse
import shutil
import torch
import numpy as np
from PIL import Image
import cv2

from config import cfg
from isp.filters import *
from isp_blocks import ISPBlocks
from envs.isp_env import ISPEnv
from envs.custom_td3 import CustomTD3
from stable_baselines3.common.logger import configure


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def prepare_temp_dataset(input_image_path, target_image_path, temp_dir):
    """
    创建临时数据集目录结构
    """
    val_dir = os.path.join(temp_dir, 'val')
    os.makedirs(val_dir, exist_ok=True)
    
    # 复制输入和目标图像到临时目录，使用标准命名格式
    input_ext = os.path.splitext(input_image_path)[1]
    target_ext = os.path.splitext(target_image_path)[1]
    
    input_dest = os.path.join(val_dir, f'Image0001-Input{input_ext}')
    target_dest = os.path.join(val_dir, f'Image0001-Target{target_ext}')
    
    shutil.copy(input_image_path, input_dest)
    shutil.copy(target_image_path, target_dest)
    
    print(f"[Info] 临时数据集创建在: {temp_dir}")
    return temp_dir


def cleanup_temp_dataset(temp_dir):
    """
    清理临时数据集目录
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"[Info] 清理临时目录: {temp_dir}")


def load_model_and_infer(model_path, dataset_dir, output_dir, args):
    """
    加载模型并执行推理
    """
    # 配置 ISP 流程
    if args.isp == "wb":
        cfg.custom_isp = [ImprovedWhiteBalanceFilter]
    elif args.isp == "exp-wb-cont":
        cfg.custom_isp = [ExposureFilter, ImprovedWhiteBalanceFilter, ContrastFilter]
    elif args.isp == "full":
        # 完整的7个滤镜流程
        cfg.custom_isp = [ExposureFilter, ImprovedWhiteBalanceFilter, SaturationFilter, 
                          ContrastFilter, HighlightFilter, ShadowFilter, SharpenFilter]
    else:
        raise NotImplementedError(f"不支持的 ISP 流程: {args.isp}")
    
    print(f"[Info] 使用 ISP 流程: {[f.__name__ for f in cfg.custom_isp]}")
    
    # 初始化 ISP blocks
    isp_blocks = ISPBlocks(cfg, is_blackbox=True)
    isp_blocks.init_filters(cfg.custom_isp)
    
    # 创建评估环境
    eval_env = ISPEnv(
        cfg,
        args=args,
        is_train=False,
        data_path=os.path.join(dataset_dir, 'val'),
        image_size=args.env_img_sz,
        isp_blocks=isp_blocks,
        max_step=args.max_step,
        obs_stack_ori=False,
        obs_stack_step=True,
        obs_stack_stop=True,
        obs_history_action=False,
        obs_img_mean_rgb=False,
        obs_inp_laplacian=0,
        joint_obs=args.joint_obs,
        truncate_param=False,
        truncate_retouch_mean=False,
        isp_inp_original=True,
        loss_type=args.loss_type,
        reward_scale=args.reward_scale,
        eval_use_best_img=False,
        save_freq=1,
        only_eval=True,
    )
    
    # 加载模型
    print(f"[Info] 加载模型: {model_path}")
    model = CustomTD3.load(model_path, env=eval_env)
    
    # 执行推理
    print(f"[Info] 开始推理 (最大步数: {args.max_step})...")
    obs, info = eval_env.reset()
    
    for step in range(args.max_step):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        print(f"  步骤 {step + 1}/{args.max_step}: PSNR = {info['psnr']:.2f} dB")
        
        if terminated or truncated:
            print(f"[Info] 推理在第 {step + 1} 步终止")
            break
    
    # 获取最佳结果
    best_psnr = info.get('best_psnr', info['psnr'])
    best_ssim = info.get('best_ssim', 0)
    print(f"\n[结果] 最佳 PSNR: {best_psnr:.2f} dB")
    print(f"[结果] 最佳 SSIM: {best_ssim:.4f}")
    
    # 保存输出图像
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取最佳图像（如果有记录的话）
    if len(eval_env.psnr_steps) > 0:
        best_idx = eval_env.psnr_steps.index(max(eval_env.psnr_steps))
        best_image = eval_env.image_steps[best_idx]
    else:
        best_image = eval_env.images[0]
    
    # 转换为 numpy 并保存
    output_image = best_image.detach().cpu().permute(1, 2, 0).numpy()
    output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
    
    output_path = os.path.join(output_dir, 'output.png')
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"[Info] 输出图像保存至: {output_path}")
    
    # 保存输入和目标（用于对比）
    input_image = eval_env.original_images[0].detach().cpu().permute(1, 2, 0).numpy()
    input_image = np.clip(input_image * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, 'input.png'), 
                cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
    
    target_image = eval_env.targets[0].detach().cpu().permute(1, 2, 0).numpy()
    target_image = np.clip(target_image * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, 'target.png'), 
                cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR))
    
    print(f"[Info] 推理完成！所有结果保存在: {output_dir}")
    
    return output_path, best_psnr, best_ssim


def main():
    parser = argparse.ArgumentParser(description='RLPixTuner 单张图片推理')
    
    # 必需参数
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--target', type=str, required=True,
                        help='目标图像路径（参考效果）')
    parser.add_argument('--model_path', type=str, 
                        default='envs/checkpoints/best_model.zip',
                        help='预训练模型路径')
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                        help='输出目录')
    
    # ISP 配置
    parser.add_argument('--isp', type=str, default='full',
                        choices=['wb', 'exp-wb-cont', 'full'],
                        help='ISP 流程类型: wb(白平衡), exp-wb-cont(曝光+白平衡+对比度), full(完整7滤镜)')
    parser.add_argument('--max_step', type=int, default=10,
                        help='最大优化步数')
    
    # 模型配置（需要与训练时一致）
    parser.add_argument('--joint_obs', type=str2bool, default=False,
                        help='是否使用联合观测空间')
    parser.add_argument('--env_img_sz', type=int, default=64,
                        help='环境图像尺寸')
    parser.add_argument('--loss_type', type=str, default='psnr',
                        choices=['l1', 'l2', 'psnr'],
                        help='损失类型')
    parser.add_argument('--reward_scale', type=float, default=0.01,
                        help='奖励缩放因子')
    
    # 其他参数
    parser.add_argument('--save_path', type=str, default='single_image_inference',
                        help='保存路径名称')
    parser.add_argument('--data_name', type=str, default='custom')
    parser.add_argument('--net_arch', type=str, default='ultra')
    parser.add_argument('--agent', type=str, default='custom_isp_single_ddpg')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--keep_temp', action='store_true',
                        help='保留临时数据集目录')
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入图像不存在: {args.input}")
    if not os.path.exists(args.target):
        raise FileNotFoundError(f"目标图像不存在: {args.target}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    print("=" * 60)
    print("RLPixTuner 单张图片推理")
    print("=" * 60)
    print(f"输入图像: {args.input}")
    print(f"目标图像: {args.target}")
    print(f"模型路径: {args.model_path}")
    print(f"ISP 流程: {args.isp}")
    print(f"最大步数: {args.max_step}")
    print("=" * 60)
    
    # 创建临时数据集
    temp_dir = os.path.join(args.output_dir, '_temp_dataset')
    dataset_dir = prepare_temp_dataset(args.input, args.target, temp_dir)
    
    try:
        # 执行推理
        output_path, psnr, ssim = load_model_and_infer(
            args.model_path, 
            dataset_dir, 
            args.output_dir, 
            args
        )
        
        print("\n" + "=" * 60)
        print("推理完成！")
        print(f"输出图像: {output_path}")
        print(f"最终 PSNR: {psnr:.2f} dB")
        print(f"最终 SSIM: {ssim:.4f}")
        print("=" * 60)
        
    finally:
        # 清理临时数据集
        if not args.keep_temp:
            cleanup_temp_dataset(temp_dir)


if __name__ == "__main__":
    main()
