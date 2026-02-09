# RLPixTuner 推理部署文件说明

本目录包含完整的推理部署解决方案。

## 📁 文件清单

### 1. **inference_single_image.py** - 命令行推理工具
**用途**: 对单张图片进行推理的命令行脚本

**使用方法**:
```bash
python inference_single_image.py \
    --input examples/input.jpg \
    --target examples/target.jpg \
    --model_path envs/checkpoints/best_model.zip \
    --output_dir ./results \
    --isp full \
    --max_step 10
```

**主要参数**:
- `--input`: 输入图像路径（必需）
- `--target`: 目标参考图像路径（必需）
- `--model_path`: 预训练模型路径
- `--output_dir`: 输出目录
- `--isp`: ISP流程类型 (wb/exp-wb-cont/full)
- `--max_step`: 最大优化步数

---

### 2. **rlpixtuner_api.py** - Python API 封装
**用途**: 提供简洁的 Python API 接口，方便集成到其他项目

**核心类**:
- `RLPixTunerInference`: 主推理类
- `quick_inference()`: 快速推理函数

**使用示例**:
```python
from rlpixtuner_api import RLPixTunerInference

# 创建推理器
inferencer = RLPixTunerInference(
    model_path='envs/checkpoints/best_model.zip',
    isp_type='full',
    max_step=10
)

# 单张推理
output, metrics = inferencer.predict(
    'input.jpg',
    'target.jpg',
    'output.png'
)

print(f"PSNR: {metrics['psnr']:.2f} dB")

# 批量推理
pairs = [
    ('img1_input.jpg', 'img1_target.jpg'),
    ('img2_input.jpg', 'img2_target.jpg'),
]
results = inferencer.predict_batch(pairs, output_dir='results')
```

**主要功能**:
- ✅ 单张图片推理
- ✅ 批量推理
- ✅ 自动生成对比图
- ✅ 返回详细评估指标
- ✅ 提取 ISP 参数
- ✅ 支持 numpy 和 PIL Image 输出

---

### 3. **example_usage.py** - 使用示例
**用途**: 展示 API 的各种使用方式

**包含示例**:
1. 基本用法 - 单张图片推理
2. 批量推理
3. 快速推理（简化接口）
4. 不同 ISP 流程对比
5. 返回 PIL Image 对象
6. 自定义设置
7. 错误处理

**运行示例**:
```bash
python example_usage.py
```

在代码中取消注释相应的示例函数即可运行。

---

### 4. **INFERENCE_GUIDE.md** - 完整推理指南
**用途**: 详细的推理部署文档

**包含内容**:
- 环境准备步骤
- 三种推理方法详解（批量/单张/API）
- 完整的参数说明
- 输入输出格式说明
- 常见问题解答
- 高级用法示例
- 性能优化建议

---

## 🚀 快速开始

### 方法一：命令行工具（最简单）

```bash
# 1. 准备图像对
#    - input.jpg: 待处理的输入图像
#    - target.jpg: 期望效果的参考图像

# 2. 运行推理
python inference_single_image.py \
    --input examples/input.jpg \
    --target examples/target.jpg \
    --output_dir results

# 3. 查看结果
ls results/
# output.png - 处理后的图像
# input.png - 输入图像副本
# target.png - 目标图像副本
```

---

### 方法二：Python API（最灵活）

```python
# 一行代码快速推理
from rlpixtuner_api import quick_inference

metrics = quick_inference(
    'input.jpg',
    'target.jpg',
    'output.png'
)

print(f"PSNR: {metrics['psnr']:.2f} dB")
```

---

### 方法三：批量推理（最高效）

```bash
# 1. 组织数据集
mkdir -p my_dataset/val
# 放入图像对：*-Input.jpg 和 *-Target.jpg

# 2. 运行批量推理
python envs/run_sb3_eval.py \
    --model_path envs/checkpoints/best_model.zip \
    --dataset_dir my_dataset \
    --save_path batch_results \
    --max_step 10

# 3. 查看结果
ls experiments/__sb3_saved_eval/batch_results/images/
```

---

## 📊 对比三种方法

| 方法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **命令行工具** | 单张图片、快速测试 | 简单直接、无需编程 | 单张处理效率低 |
| **Python API** | 集成到项目、自定义需求 | 灵活、可编程控制 | 需要编写代码 |
| **批量推理** | 大量图片处理 | 效率最高、自动保存 | 需要特定目录结构 |

---

## 🔧 核心功能对比

|功能|命令行|API|批量|
|---|:---:|:-:|:-:|
|单张推理|✅|✅|✅|
|批量处理|❌|✅|✅|
|自定义ISP|✅|✅|✅|
|提取参数|❌|✅|✅|
|对比图|❌|✅|✅|
|轨迹可视化|❌|❌|✅|

---

## 💡 推荐使用方式

1. **初次使用**: 用命令行工具测试 1-2 张图片，验证效果
2. **少量图片**: 使用命令行工具或 Python API
3. **大量图片**: 使用批量推理脚本
4. **集成到项目**: 使用 Python API (`rlpixtuner_api.py`)
5. **研究分析**: 使用批量推理获取完整统计和可视化

---

## 📖 详细文档

- 完整指南: `INFERENCE_GUIDE.md`
- API 文档: `rlpixtuner_api.py` 中的 docstring
- 使用示例: `example_usage.py`

---

## ⚠️ 注意事项

1. **必须提供目标图像**: RLPixTuner 是目标条件模型，需要参考图像
2. **图像格式**: 支持 .jpg, .png, .tif
3. **GPU推荐**: 使用 GPU 可显著提升速度（约 10 倍）
4. **参数一致性**: 推理时的参数配置需与训练时一致

---

## 🐛 故障排除

### 问题1: ImportError
```bash
pip install torch torchvision stable-baselines3 gymnasium opencv-python pillow
```

### 问题2: CUDA out of memory
```bash
# 方案1: 使用 CPU
python inference_single_image.py ... --device cpu

# 方案2: 降低图像尺寸
python envs/run_sb3_eval.py ... --env_img_sz 32
```

### 问题3: 模型文件不存在
```bash
# 确认模型文件存在
ls -lh envs/checkpoints/best_model.zip

# 如果不存在，请从项目仓库下载或训练模型
```

---

## 📞 获取帮助

- 查看完整文档: `INFERENCE_GUIDE.md`
- 查看使用示例: `example_usage.py`
- 项目主页: https://openimaginglab.github.io/RLPixTuner/
- GitHub Issues: https://github.com/OpenImagingLab/RLPixTuner/issues

---

## 📝 更新日志

- 2024-02-09: 创建推理部署文件集合
  - 添加命令行工具 (`inference_single_image.py`)
  - 添加 Python API (`rlpixtuner_api.py`)
  - 添加使用示例 (`example_usage.py`)
  - 添加完整指南 (`INFERENCE_GUIDE.md`)
