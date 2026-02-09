#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLPixTuner Python API
提供简单易用的推理接口
"""

import os
import shutil
import torch
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Dict, Optional, List

from config import cfg
from isp.filters import (ExposureFilter, ImprovedWhiteBalanceFilter, 
                         SaturationFilter, ContrastFilter, 
                         HighlightFilter, ShadowFilter, SharpenFilter)
from isp_blocks import ISPBlocks
from envs.isp_env import ISPEnv
from envs.custom_td3 import CustomTD3


class RLPixTunerInference:
    """
    RLPixTuner 推理封装类
    
    简化的推理接口，支持单张图片和批量推理
    
    Examples:
        >>> # 初始化
        >>> inferencer = RLPixTunerInference(
        ...     model_path='envs/checkpoints/best_model.zip',
        ...     isp_type='full'
        ... )
        >>> 
        >>> # 单张推理
        >>> output, metrics = inferencer.predict('input.jpg', 'target.jpg')
        >>> print(f"PSNR: {metrics['psnr']:.2f} dB")
    """
    
    def __init__(self, 
                 model_path: str,
                 isp_type: str = 'full',
                 max_step: int = 10,
                 device: str = 'cuda',
                 verbose: bool = True):
        """
        初始化 RLPixTuner 推理器
        
        Args:
            model_path: 预训练模型路径
            isp_type: ISP流程类型，可选:
                - 'wb': 仅白平衡 (1个滤镜)
                - 'exp-wb-cont': 曝光+白平衡+对比度 (3个滤镜)
                - 'full': 完整流程 (7个滤镜) [推荐]
            max_step: 最大优化迭代步数，推荐 5-10
            device: 计算设备 ('cuda' 或 'cpu')
            verbose: 是否打印详细信息
        
        Raises:
            FileNotFoundError: 模型文件不存在
            ValueError: 不支持的 ISP 类型
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model_path = model_path
        self.max_step = max_step
        self.device = device
        self.verbose = verbose
        
        # 配置 ISP 流程
        self._setup_isp(isp_type)
        
        # 延迟加载模型（首次推理时加载）
        self.model = None
        self.env = None
        self._temp_dir_counter = 0
    
    def _setup_isp(self, isp_type: str):
        """配置 ISP 滤镜流程"""
        isp_configs = {
            'wb': [ImprovedWhiteBalanceFilter],
            'exp-wb-cont': [ExposureFilter, ImprovedWhiteBalanceFilter, ContrastFilter],
            'full': [ExposureFilter, ImprovedWhiteBalanceFilter, SaturationFilter, 
                     ContrastFilter, HighlightFilter, ShadowFilter, SharpenFilter],
        }
        
        if isp_type not in isp_configs:
            raise ValueError(
                f"不支持的 ISP 类型: {isp_type}. "
                f"可选: {list(isp_configs.keys())}"
            )
        
        cfg.custom_isp = isp_configs[isp_type]
        self.isp_blocks = ISPBlocks(cfg, is_blackbox=True)
        self.isp_blocks.init_filters(cfg.custom_isp)
        
        if self.verbose:
            print(f"[RLPixTuner] ISP 流程: {[f.__name__ for f in cfg.custom_isp]}")
    
    def _create_temp_dataset(self, 
                            input_path: str, 
                            target_path: str) -> str:
        """创建临时数据集目录"""
        temp_dir = f'./temp_rlpixtuner_{self._temp_dir_counter}'
        self._temp_dir_counter += 1
        
        val_dir = os.path.join(temp_dir, 'val')
        os.makedirs(val_dir, exist_ok=True)
        
        input_ext = os.path.splitext(input_path)[1]
        target_ext = os.path.splitext(target_path)[1]
        
        shutil.copy(input_path, os.path.join(val_dir, f'temp-Input{input_ext}'))
        shutil.copy(target_path, os.path.join(val_dir, f'temp-Target{target_ext}'))
        
        return temp_dir
    
    def _cleanup_temp_dataset(self, temp_dir: str):
        """清理临时数据集目录"""
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def _load_model(self, temp_dir: str):
        """加载模型和创建环境（延迟加载）"""
        if self.model is not None:
            return
        
        # 创建 args 对象
        from easydict import EasyDict
        args = EasyDict({
            'save_path': 'temp_inference',
            'data_name': 'custom',
            'net_arch': 'ultra',
            'agent': 'custom_isp_single_ddpg',
            'batch_size': 1,
        })
        
        # 创建环境
        self.env = ISPEnv(
            cfg,
            args=args,
            is_train=False,
            data_path=os.path.join(temp_dir, 'val'),
            image_size=64,
            isp_blocks=self.isp_blocks,
            max_step=self.max_step,
            obs_stack_ori=False,
            obs_stack_step=True,
            obs_stack_stop=True,
            obs_history_action=False,
            obs_img_mean_rgb=False,
            obs_inp_laplacian=0,
            joint_obs=False,
            truncate_param=False,
            truncate_retouch_mean=False,
            isp_inp_original=True,
            loss_type='psnr',
            reward_scale=0.01,
            eval_use_best_img=False,
            save_freq=1,
            only_eval=True,
        )
        
        # 加载模型
        self.model = CustomTD3.load(self.model_path, env=self.env)
        
        if self.verbose:
            print(f"[RLPixTuner] 模型加载成功: {self.model_path}")
    
    def predict(self, 
                input_image_path: str,
                target_image_path: str,
                output_path: Optional[str] = None,
                return_numpy: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        对单张图片进行推理
        
        Args:
            input_image_path: 输入图像路径
            target_image_path: 目标参考图像路径
            output_path: 输出图像保存路径（可选）
            return_numpy: 是否返回 numpy 数组，否则返回 PIL Image
        
        Returns:
            output_image: 输出图像
                - 如果 return_numpy=True: numpy.ndarray (H, W, 3), RGB, uint8, 0-255
                - 如果 return_numpy=False: PIL.Image
            metrics: 评估指标字典
                - 'psnr': Peak Signal-to-Noise Ratio (dB)
                - 'ssim': Structural Similarity Index
                - 'steps': 实际优化步数
        
        Raises:
            FileNotFoundError: 输入图像不存在
        
        Examples:
            >>> inferencer = RLPixTunerInference('model.zip')
            >>> output, metrics = inferencer.predict('input.jpg', 'target.jpg')
            >>> print(f"PSNR: {metrics['psnr']:.2f} dB")
        """
        # 验证输入
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"输入图像不存在: {input_image_path}")
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"目标图像不存在: {target_image_path}")
        
        # 创建临时数据集
        temp_dir = self._create_temp_dataset(input_image_path, target_image_path)
        
        try:
            # 加载模型（首次调用时）
            self._load_model(temp_dir)
            
            # 执行推理
            obs, info = self.env.reset()
            
            for step in range(self.max_step):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                if self.verbose:
                    print(f"  步骤 {step + 1}/{self.max_step}: PSNR = {info['psnr']:.2f} dB")
                
                if terminated or truncated:
                    if self.verbose:
                        print(f"  推理在第 {step + 1} 步终止")
                    break
            
            # 获取最佳结果
            best_psnr = info.get('best_psnr', info['psnr'])
            best_ssim = info.get('best_ssim', 0)
            actual_steps = len(self.env.psnr_steps)
            
            if len(self.env.psnr_steps) > 0:
                best_idx = self.env.psnr_steps.index(max(self.env.psnr_steps))
                best_image = self.env.image_steps[best_idx]
            else:
                best_image = self.env.images[0]
            
            # 转换为 numpy
            output_array = best_image.detach().cpu().permute(1, 2, 0).numpy()
            output_array = np.clip(output_array * 255, 0, 255).astype(np.uint8)
            
            # 保存图像
            if output_path:
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                cv2.imwrite(output_path, cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR))
                if self.verbose:
                    print(f"[RLPixTuner] 输出已保存: {output_path}")
            
            # 准备返回值
            if return_numpy:
                output_image = output_array
            else:
                output_image = Image.fromarray(output_array)
            
            metrics = {
                'psnr': float(best_psnr),
                'ssim': float(best_ssim),
                'steps': actual_steps,
            }
            
            if self.verbose:
                print(f"[RLPixTuner] PSNR: {metrics['psnr']:.2f} dB, "
                      f"SSIM: {metrics['ssim']:.4f}, "
                      f"步数: {metrics['steps']}")
            
            return output_image, metrics
        
        finally:
            # 清理临时目录
            self._cleanup_temp_dataset(temp_dir)
    
    def predict_batch(self,
                     image_pairs: List[Tuple[str, str]],
                     output_dir: Optional[str] = None,
                     save_comparison: bool = True) -> List[Dict[str, any]]:
        """
        批量推理多张图片
        
        Args:
            image_pairs: 图像对列表 [(input_path, target_path), ...]
            output_dir: 输出目录（可选）
            save_comparison: 是否保存对比图（输入-输出-目标）
        
        Returns:
            results: 结果列表，每个元素包含:
                - 'input_path': 输入图像路径
                - 'output_image': 输出图像 (numpy array)
                - 'metrics': 评估指标
        
        Examples:
            >>> pairs = [
            ...     ('img1_input.jpg', 'img1_target.jpg'),
            ...     ('img2_input.jpg', 'img2_target.jpg'),
            ... ]
            >>> results = inferencer.predict_batch(pairs, output_dir='results')
            >>> for r in results:
            ...     print(f"{r['input_path']}: PSNR={r['metrics']['psnr']:.2f}")
        """
        results = []
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for idx, (input_path, target_path) in enumerate(image_pairs):
            if self.verbose:
                print(f"\n[{idx + 1}/{len(image_pairs)}] 处理: {input_path}")
            
            # 确定输出路径
            if output_dir:
                basename = os.path.splitext(os.path.basename(input_path))[0]
                output_path = os.path.join(output_dir, f'{basename}_output.png')
            else:
                output_path = None
            
            try:
                # 推理
                output_image, metrics = self.predict(
                    input_path, 
                    target_path, 
                    output_path=output_path
                )
                
                # 保存对比图
                if save_comparison and output_dir:
                    self._save_comparison(
                        input_path, 
                        target_path, 
                        output_image,
                        os.path.join(output_dir, f'{basename}_comparison.png')
                    )
                
                results.append({
                    'input_path': input_path,
                    'output_image': output_image,
                    'metrics': metrics,
                    'success': True,
                })
                
            except Exception as e:
                if self.verbose:
                    print(f"  [错误] {str(e)}")
                results.append({
                    'input_path': input_path,
                    'output_image': None,
                    'metrics': None,
                    'success': False,
                    'error': str(e),
                })
        
        # 打印汇总
        if self.verbose:
            print("\n" + "=" * 60)
            print("批量推理完成")
            print("=" * 60)
            successful = sum(1 for r in results if r['success'])
            print(f"成功: {successful}/{len(results)}")
            
            if successful > 0:
                avg_psnr = np.mean([r['metrics']['psnr'] for r in results if r['success']])
                avg_ssim = np.mean([r['metrics']['ssim'] for r in results if r['success']])
                print(f"平均 PSNR: {avg_psnr:.2f} dB")
                print(f"平均 SSIM: {avg_ssim:.4f}")
            print("=" * 60)
        
        return results
    
    def _save_comparison(self, 
                        input_path: str, 
                        target_path: str,
                        output_image: np.ndarray,
                        save_path: str):
        """保存输入-输出-目标对比图"""
        input_img = cv2.imread(input_path)
        target_img = cv2.imread(target_path)
        output_img = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        
        # 调整到相同高度
        h = output_img.shape[0]
        input_img = cv2.resize(input_img, (h, h))
        target_img = cv2.resize(target_img, (h, h))
        
        # 水平拼接
        comparison = np.hstack([input_img, output_img, target_img])
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Input', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Output', (h + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Target', (h * 2 + 10, 30), font, 1, (255, 255, 255), 2)
        
        cv2.imwrite(save_path, comparison)
    
    def get_isp_parameters(self) -> Dict[str, np.ndarray]:
        """
        获取最后一次推理的 ISP 参数
        
        Returns:
            params_dict: ISP 参数字典，键为滤镜名称，值为参数数组
        
        Examples:
            >>> output, metrics = inferencer.predict('input.jpg', 'target.jpg')
            >>> params = inferencer.get_isp_parameters()
            >>> print(params['ExposureFilter'])  # 曝光参数
        """
        if self.env is None:
            raise RuntimeError("请先调用 predict() 方法进行推理")
        
        params_dict = {}
        for idx, param in enumerate(self.env.params):
            filter_name = cfg.custom_isp[idx].__name__
            params_dict[filter_name] = param.cpu().numpy()
        
        return params_dict
    
    def __del__(self):
        """析构函数，清理资源"""
        # 清理可能残留的临时目录
        for i in range(self._temp_dir_counter):
            temp_dir = f'./temp_rlpixtuner_{i}'
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


# 便捷函数
def quick_inference(input_path: str,
                   target_path: str,
                   output_path: str,
                   model_path: str = 'envs/checkpoints/best_model.zip',
                   max_step: int = 10) -> Dict[str, float]:
    """
    快速推理函数（简化接口）
    
    Args:
        input_path: 输入图像路径
        target_path: 目标图像路径
        output_path: 输出图像路径
        model_path: 模型路径
        max_step: 最大步数
    
    Returns:
        metrics: 评估指标字典
    
    Examples:
        >>> from rlpixtuner_api import quick_inference
        >>> metrics = quick_inference('input.jpg', 'target.jpg', 'output.png')
        >>> print(f"PSNR: {metrics['psnr']:.2f} dB")
    """
    inferencer = RLPixTunerInference(
        model_path=model_path,
        isp_type='full',
        max_step=max_step,
        verbose=True
    )
    
    _, metrics = inferencer.predict(input_path, target_path, output_path)
    return metrics


if __name__ == '__main__':
    # 测试代码
    print("RLPixTuner API 测试")
    print("=" * 60)
    
    # 检查模型是否存在
    model_path = 'envs/checkpoints/best_model.zip'
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先下载或训练模型")
    else:
        print(f"[成功] 找到模型: {model_path}")
        print("\n使用示例:")
        print(">>> from rlpixtuner_api import RLPixTunerInference")
        print(">>> inferencer = RLPixTunerInference('envs/checkpoints/best_model.zip')")
        print(">>> output, metrics = inferencer.predict('input.jpg', 'target.jpg', 'output.png')")
        print(">>> print(f'PSNR: {metrics[\"psnr\"]:.2f} dB')")
