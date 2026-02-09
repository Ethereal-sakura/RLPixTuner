# RLPixTuner 推理部署指南

本指南详细说明如何使用 RLPixTuner 进行图像推理。

## 📋 目录

1. [环境准备](#环境准备)
2. [方法一：批量推理（推荐）](#方法一批量推理推荐)
3. [方法二：单张图片推理](#方法二单张图片推理)
4. [方法三：Python API 调用](#方法三python-api-调用)
5. [常见问题](#常见问题)

---

## 环境准备

### 1. 安装依赖

```bash
# 使用 conda（推荐）
conda create --name rlpixtuner python=3.9
conda activate rlpixtuner

# 安装核心依赖
pip install torch==2.0.0 torchvision==0.15.0
pip install stable-baselines3 gymnasium opencv-python pillow numpy scipy

# 或使用项目提供的 requirements.txt
conda create --name rlpixtuner --file requirements.txt
```

### 2. 验证模型文件

确认预训练模型存在：

```bash
ls -lh envs/checkpoints/
# 应该看到:
# best_model.zip          - Photo Finishing Tuning 模型
# best_model_style.zip    - Photo Stylization Tuning 模型
```

---

## 方法一：批量推理（推荐）

适合处理多张图片的场景。

### 数据准备

创建数据集目录结构：

```
my_dataset/
└── val/
    ├── img001-Input.jpg     # 输入图像
    ├── img001-Target.jpg    # 目标参考图像
    ├── img002-Input.png
    ├── img002-Target.png
    └── ...
```

**命名规则**：
- 输入：`*-Input.[jpg|png|tif]`
- 目标：`*-Target.[jpg|png|tif]`
- 前缀必须匹配

### 运行推理

```bash
CUDA_VISIBLE_DEVICES=0 python envs/run_sb3_eval.py \
    --model_path envs/checkpoints/best_model.zip \
    --dataset_dir /path/to/my_dataset \
    --save_path my_inference_results \
    --isp full \
    --max_step 10 \
    --joint_obs False \
    --env_img_sz 64 \
    --loss_type psnr \
    --reward_scale 0.01
```

### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--model_path` | 模型路径 | `envs/checkpoints/best_model.zip` |
| `--dataset_dir` | 数据集根目录 | 包含 `val/` 子目录 |
| `--isp` | ISP流程类型 | `dataset` (自动识别) 或 `full` |
| `--max_step` | 最大迭代步数 | 10 |
| `--joint_obs` | 观测空间类型 | False（推理时） |
| `--env_img_sz` | 图像处理尺寸 | 64 |

### 输出结果

结果保存在 `experiments/__sb3_saved_eval/<save_path>/images/`：

```
images/
├── 001_img001_in.png       # 输入图像
├── 001_img001_target.png   # 目标图像
├── 001_img001_out_32-45.png  # 输出（PSNR=32.45）
├── 001_img001_para.txt     # ISP 参数记录
└── 001_img001_traj.png     # 优化轨迹可视化
```

---

## 方法二：单张图片推理

使用提供的 `inference_single_image.py` 脚本。

### 基本用法

```bash
python inference_single_image.py \
    --input /path/to/input.jpg \
    --target /path/to/target.jpg \
    --model_path envs/checkpoints/best_model.zip \
    --output_dir ./my_output \
    --isp full \
    --max_step 10
```

### 完整示例

```bash
# 使用完整的 7 滤镜流程
python inference_single_image.py \
    --input examples/input_001.jpg \
    --target examples/target_001.jpg \
    --model_path envs/checkpoints/best_model.zip \
    --output_dir ./results/test_001 \
    --isp full \
    --max_step 10 \
    --joint_obs False \
    --loss_type psnr

# 使用白平衡流程
python inference_single_image.py \
    --input examples/input_002.jpg \
    --target examples/target_002.jpg \
    --isp wb \
    --max_step 5
```

### ISP 流程选项

- `wb`: 仅白平衡调整（1个滤镜）
- `exp-wb-cont`: 曝光+白平衡+对比度（3个滤镜）
- `full`: 完整流程（7个滤镜）
  - Exposure（曝光）
  - White Balance（白平衡）
  - Saturation（饱和度）
  - Contrast（对比度）
  - Highlight（高光）
  - Shadow（阴影）
  - Sharpen（锐化）

### 输出结果

```
my_output/
├── input.png      # 输入图像
├── target.png     # 目标图像
├── output.png     # 处理后的输出图像
└── _temp_dataset/ # 临时目录（可用 --keep_temp 保留）
```

---

## 方法三：Python API 调用

### 完整示例代码

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在 Python 脚本中调用 RLPixTuner
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2

# 导入项目模块
from config import cfg
from isp.filters import (ExposureFilter, ImprovedWhiteBalanceFilter, 
                         SaturationFilter, ContrastFilter, 
                         HighlightFilter, ShadowFilter, SharpenFilter)
from isp_blocks import ISPBlocks
from envs.isp_env import ISPEnv
from envs.custom_td3 import CustomTD3


class RLPixTunerInference:
    """RLPixTuner 推理封装类"""
    
    def __init__(self, model_path, isp_type='full', max_step=10, device='cuda'):
        """
        初始化推理器
        
        Args:
            model_path: 模型文件路径
            isp_type: ISP流程类型 ('wb', 'exp-wb-cont', 'full')
            max_step: 最大优化步数
            device: 计算设备
        """
        self.model_path = model_path
        self.max_step = max_step
        self.device = device
        
        # 配置 ISP 流程
        if isp_type == 'wb':
            cfg.custom_isp = [ImprovedWhiteBalanceFilter]
        elif isp_type == 'exp-wb-cont':
            cfg.custom_isp = [ExposureFilter, ImprovedWhiteBalanceFilter, ContrastFilter]
        elif isp_type == 'full':
            cfg.custom_isp = [ExposureFilter, ImprovedWhiteBalanceFilter, SaturationFilter, 
                              ContrastFilter, HighlightFilter, ShadowFilter, SharpenFilter]
        else:
            raise ValueError(f"不支持的 ISP 类型: {isp_type}")
        
        # 初始化 ISP blocks
        self.isp_blocks = ISPBlocks(cfg, is_blackbox=True)
        self.isp_blocks.init_filters(cfg.custom_isp)
        
        print(f"[Info] ISP 流程: {[f.__name__ for f in cfg.custom_isp]}")
        
        # 加载模型（延迟到需要时加载）
        self.model = None
        self.env = None
    
    def _create_temp_dataset(self, input_path, target_path, temp_dir='./temp_inference'):
        """创建临时数据集"""
        import shutil
        
        val_dir = os.path.join(temp_dir, 'val')
        os.makedirs(val_dir, exist_ok=True)
        
        input_ext = os.path.splitext(input_path)[1]
        target_ext = os.path.splitext(target_path)[1]
        
        shutil.copy(input_path, os.path.join(val_dir, f'temp-Input{input_ext}'))
        shutil.copy(target_path, os.path.join(val_dir, f'temp-Target{target_ext}'))
        
        return temp_dir
    
    def _cleanup_temp_dataset(self, temp_dir):
        """清理临时数据集"""
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def predict(self, input_image_path, target_image_path, output_path=None):
        """
        对单张图片进行推理
        
        Args:
            input_image_path: 输入图像路径
            target_image_path: 目标参考图像路径
            output_path: 输出图像保存路径（可选）
        
        Returns:
            output_image: 输出图像 (numpy array, RGB, 0-255)
            metrics: 评估指标字典 {'psnr': float, 'ssim': float}
        """
        # 创建临时数据集
        temp_dir = self._create_temp_dataset(input_image_path, target_image_path)
        
        try:
            # 创建环境和加载模型（仅第一次）
            if self.env is None:
                # 创建 args 对象（模拟命令行参数）
                from easydict import EasyDict
                args = EasyDict({
                    'save_path': 'temp_inference',
                    'data_name': 'custom',
                    'net_arch': 'ultra',
                    'agent': 'custom_isp_single_ddpg',
                    'batch_size': 1,
                })
                
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
                
                self.model = CustomTD3.load(self.model_path, env=self.env)
                print(f"[Info] 模型加载成功: {self.model_path}")
            
            # 执行推理
            obs, info = self.env.reset()
            
            for step in range(self.max_step):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                if terminated or truncated:
                    break
            
            # 获取结果
            best_psnr = info.get('best_psnr', info['psnr'])
            best_ssim = info.get('best_ssim', 0)
            
            if len(self.env.psnr_steps) > 0:
                best_idx = self.env.psnr_steps.index(max(self.env.psnr_steps))
                best_image = self.env.image_steps[best_idx]
            else:
                best_image = self.env.images[0]
            
            # 转换为 numpy
            output_image = best_image.detach().cpu().permute(1, 2, 0).numpy()
            output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
            
            # 保存图像
            if output_path:
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
                print(f"[Info] 输出已保存: {output_path}")
            
            metrics = {
                'psnr': float(best_psnr),
                'ssim': float(best_ssim),
            }
            
            return output_image, metrics
        
        finally:
            # 清理临时目录
            self._cleanup_temp_dataset(temp_dir)


# 使用示例
def main():
    """使用示例"""
    
    # 1. 初始化推理器
    inferencer = RLPixTunerInference(
        model_path='envs/checkpoints/best_model.zip',
        isp_type='full',
        max_step=10,
        device='cuda'
    )
    
    # 2. 单张图片推理
    output_image, metrics = inferencer.predict(
        input_image_path='examples/input.jpg',
        target_image_path='examples/target.jpg',
        output_path='results/output.png'
    )
    
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    
    # 3. 批量推理
    image_pairs = [
        ('examples/img1_input.jpg', 'examples/img1_target.jpg', 'results/img1_output.png'),
        ('examples/img2_input.jpg', 'examples/img2_target.jpg', 'results/img2_output.png'),
    ]
    
    for input_path, target_path, output_path in image_pairs:
        output, metrics = inferencer.predict(input_path, target_path, output_path)
        print(f"{os.path.basename(input_path)}: PSNR={metrics['psnr']:.2f} dB")


if __name__ == '__main__':
    main()
```

### 简单使用示例

```python
from rlpixtuner_inference import RLPixTunerInference

# 创建推理器
inferencer = RLPixTunerInference(
    model_path='envs/checkpoints/best_model.zip',
    isp_type='full',
    max_step=10
)

# 推理
output, metrics = inferencer.predict(
    'input.jpg',
    'target.jpg',
    'output.png'
)

print(f"PSNR: {metrics['psnr']:.2f} dB")
```

---

## 常见问题

### Q1: 模型输入需要什么格式的图像？

**A**: 支持 `.jpg`, `.png`, `.tif` 格式，RGB 彩色图像。图像会自动调整大小。

### Q2: 必须提供目标图像吗？

**A**: 是的。RLPixTuner 是**目标条件（Goal-Conditioned）**的强化学习模型，需要参考目标来指导优化方向。

### Q3: 如何选择 ISP 流程？

**A**: 
- 仅需白平衡调整 → `wb`
- 需要基础色调调整 → `exp-wb-cont`
- 需要完整后期处理 → `full`（推荐）

### Q4: max_step 应该设置多少？

**A**: 通常 10 步就能达到很好的效果。论文中提到 10 步的结果接近传统方法 500 步的效果。

### Q5: 推理速度如何？

**A**: 
- GPU（RTX 3090）：约 0.5-1 秒/步，10步约 5-10 秒
- CPU：约 2-5 秒/步

### Q6: 报错 "CUDA out of memory"

**A**: 
```bash
# 降低图像处理尺寸
--env_img_sz 32  # 默认是 64

# 或使用 CPU
CUDA_VISIBLE_DEVICES=-1 python inference_single_image.py ...
```

### Q7: 推理结果与论文不符

**A**: 确保：
1. 使用正确的预训练模型
2. `--joint_obs` 等参数与模型训练时一致
3. 目标图像质量合理（不是过度处理）

### Q8: 如何调整 ISP 参数范围？

**A**: 修改 `config.py` 中的配置：

```python
cfg.exposure_range = 2.0      # 曝光范围
cfg.wb_range = 1.1            # 白平衡范围
cfg.sharpen_range = (0.0, 10.0)  # 锐化范围
```

---

## 高级用法

### 自定义 ISP 流程

```python
from isp.filters import *

# 只使用特定滤镜
cfg.custom_isp = [
    ExposureFilter,      # 曝光调整
    SaturationFilter,    # 饱和度调整
    SharpenFilter,       # 锐化
]

isp_blocks = ISPBlocks(cfg, is_blackbox=True)
isp_blocks.init_filters(cfg.custom_isp)
```

### 提取 ISP 参数

```python
# 在推理后获取 ISP 参数
params = env.params  # 最终参数列表

for idx, param in enumerate(params):
    filter_name = cfg.custom_isp[idx].__name__
    print(f"{filter_name}: {param.cpu().numpy()}")
```

### 可视化优化轨迹

```python
import matplotlib.pyplot as plt

# 获取每一步的 PSNR
psnr_history = env.psnr_steps

plt.plot(range(1, len(psnr_history) + 1), psnr_history)
plt.xlabel('Step')
plt.ylabel('PSNR (dB)')
plt.title('Optimization Trajectory')
plt.savefig('trajectory.png')
```

---

## 性能优化建议

1. **批量处理**: 使用批量推理脚本而非逐张处理
2. **GPU 加速**: 确保 CUDA 可用
3. **图像尺寸**: 推理时使用较小的 `env_img_sz` (如 64)
4. **步数控制**: 根据需求调整 `max_step`，通常 5-10 步足够

---

## 技术支持

- 项目主页: https://openimaginglab.github.io/RLPixTuner/
- 论文: https://openreview.net/pdf?id=4kVHI2uXRE
- GitHub: https://github.com/OpenImagingLab/RLPixTuner

如有问题，请在 GitHub 提交 Issue。
