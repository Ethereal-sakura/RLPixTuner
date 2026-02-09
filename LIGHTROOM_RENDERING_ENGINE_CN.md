# Lightroom渲染引擎范围修改说明

## 概述

已将渲染引擎的参数范围修改为与Lightroom一致。现在模型将直接学习和输出Lightroom标准范围的参数。

## 修改内容

### 1. 配置文件修改 (`config.py`)

**新增配置项**：
```python
cfg.use_lightroom_ranges = True  # 启用Lightroom范围

# 参数范围设置（与Lightroom一致）
cfg.exposure_range = 5.0      # -5.0 到 +5.0 EV
cfg.saturation_range = 100.0  # -100 到 +100
cfg.contrast_range = 100.0    # -100 到 +100
cfg.highlight_range = 100.0   # -100 到 +100
cfg.shadow_range = 100.0      # -100 到 +100
cfg.sharpen_range = 150.0     # 0 到 150
```

### 2. 渲染引擎修改 (`isp/filters.py`)

修改了所有相关Filter的参数范围和处理逻辑：

#### ExposureFilter
- **原范围**：-2.0 到 +2.0 EV
- **新范围**：-5.0 到 +5.0 EV（Lightroom标准）
- **处理方式**：不变，仍使用 `image * 2^EV`

#### SaturationFilter
- **原范围**：-1.0 到 +1.0（内部表示）
- **新范围**：-100 到 +100（Lightroom标准）
- **处理方式**：
  ```python
  # 转换公式：adjustment_factor = 1.0 + (param / 100.0)
  # -100 → 0.0 (完全去饱和，灰度)
  #    0 → 1.0 (原始)
  # +100 → 2.0 (饱和度加倍)
  ```

#### ContrastFilter
- **原范围**：-1.0 到 +1.0（内部表示）
- **新范围**：-100 到 +100（Lightroom标准）
- **处理方式**：
  ```python
  # 转换公式：blend_factor = param / 100.0
  # 归一化到 [-1, 1] 用于混合
  ```

#### HighlightFilter
- **原范围**：-1.0 到 +1.0（内部表示）
- **新范围**：-100 到 +100（Lightroom标准）
- **处理方式**：
  ```python
  # 转换公式：normalized_param = param / 100.0
  # -100 → 压暗高光
  #    0 → 不改变
  # +100 → 提亮高光
  ```

#### ShadowFilter
- **原范围**：-1.0 到 +1.0（内部表示）
- **新范围**：-100 到 +100（Lightroom标准）
- **处理方式**：
  ```python
  # 转换公式：normalized_param = param / 100.0
  # -100 → 压暗阴影
  #    0 → 不改变
  # +100 → 提亮阴影
  ```

#### SharpenFilter
- **原范围**：0.0 到 10.0（内部因子）
- **新范围**：0 到 150（Lightroom标准）
- **处理方式**：
  ```python
  # 转换公式：sharpness_factor = 1.0 + (param / 150.0) * 9.0
  #   0 → 1.0 (无锐化)
  #  75 → 5.5 (中等锐化)
  # 150 → 10.0 (最大锐化)
  ```

## 参数范围对比表

| Filter | 原范围 | Lightroom范围 | 说明 |
|--------|--------|--------------|------|
| Exposure | -2.0 ~ +2.0 | -5.0 ~ +5.0 | EV档位 |
| Saturation | -1.0 ~ +1.0 | -100 ~ +100 | 百分比 |
| Contrast | -1.0 ~ +1.0 | -100 ~ +100 | 百分比 |
| Highlights | -1.0 ~ +1.0 | -100 ~ +100 | 百分比 |
| Shadows | -1.0 ~ +1.0 | -100 ~ +100 | 百分比 |
| Sharpness | 0.0 ~ 10.0 | 0 ~ 150 | Lightroom标准 |

## 工作流程

### 现在的参数流程：

```
模型输出: [-1, 1]
    ↓
线性映射到 Lightroom 范围
    ↓
Exposure: [-5, +5] EV
Saturation: [-100, +100]
Contrast: [-100, +100]
    ↓
渲染引擎直接使用 Lightroom 范围参数
    ↓
输出图像
```

### 示例：

**模型输出** → **实际参数值** → **Lightroom等效**

- Action: 0.2 → Exposure: +1.0 EV → Lightroom曝光滑块在 +1.0
- Action: 0.3 → Saturation: +30 → Lightroom饱和度滑块在 +30
- Action: -0.15 → Contrast: -15 → Lightroom对比度滑块在 -15

## 重新训练的优势

### 1. 参数直观易懂
训练过程中保存的参数直接就是Lightroom格式，无需转换：

**之前（内部格式）**：
```json
{
  "ExposureFilter": [0.25],
  "SaturationFilter": [0.30]
}
```

**现在（Lightroom格式）**：
```json
{
  "ExposureFilter": [1.25],  // +1.25 EV
  "SaturationFilter": [30.0]  // +30
}
```

### 2. 模型学习更符合直觉
- 模型直接学习"曝光+2档"而不是"0.4"
- 学习"饱和度+20"而不是"0.2"
- 参数含义更加明确

### 3. 便于与Lightroom对比
- 可以直接对比模型输出和摄影师在Lightroom中的调整
- 更容易分析模型的行为

## 使用方法

### 1. 开始新的训练

```bash
# 配置已自动使用 Lightroom 范围
bash bash/train.sh
```

**训练时的参数将直接是Lightroom格式！**

### 2. 查看训练参数

训练过程中保存的参数现在是：

```json
{
  "step0": {
    "ExposureFilter": [1.23],      // +1.23 EV (Lightroom格式)
    "SaturationFilter": [25.6],    // +25.6 (Lightroom格式)
    "ContrastFilter": [-12.3],     // -12.3 (Lightroom格式)
    "HighlightFilter": [18.5],     // +18.5 (Lightroom格式)
    "ShadowFilter": [8.2],         // +8.2 (Lightroom格式)
    "SharpenFilter": [65.0]        // 65 (Lightroom格式)
  }
}
```

### 3. 手动设置参数进行测试

```python
from config import cfg
from isp_blocks import ISPBlocks
import torch

# 初始化 ISP 块（使用 Lightroom 范围）
cfg.use_lightroom_ranges = True
isp_blocks = ISPBlocks(cfg, is_blackbox=True)
isp_blocks.init_filters(cfg.custom_isp)

# 设置 Lightroom 格式的参数（无需转换！）
params_list = [
    torch.tensor([[2.0]], dtype=torch.float32),    # Exposure: +2.0 EV
    torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),  # WB
    torch.tensor([[35.0]], dtype=torch.float32),   # Saturation: +35
    torch.tensor([[-10.0]], dtype=torch.float32),  # Contrast: -10
    torch.tensor([[20.0]], dtype=torch.float32),   # Highlights: +20
    torch.tensor([[15.0]], dtype=torch.float32),   # Shadows: +15
    torch.tensor([[80.0]], dtype=torch.float32),   # Sharpness: 80
]

# 直接渲染（参数已经是 Lightroom 格式）
processed_img = isp_blocks.run(original_img, params_list)
```

## 向后兼容

如果需要使用旧的参数范围，可以设置：

```python
# 在 config.py 中修改
cfg.use_lightroom_ranges = False  # 使用旧范围
```

或创建新的配置文件。

## 测试验证

### 测试脚本示例：

```python
import torch
from config import cfg
from isp.filters import ExposureFilter, SaturationFilter

# 测试 Exposure
cfg.use_lightroom_ranges = True
cfg.exposure_range = 5.0

exp_filter = ExposureFilter(cfg)
print(f"Exposure 范围: [{exp_filter.range_l}, {exp_filter.range_r}]")
# 输出: Exposure 范围: [-5.0, 5.0]

# 测试参数处理
img = torch.rand(1, 3, 256, 256)
param = torch.tensor([[2.0]])  # +2 EV (Lightroom格式)

result = exp_filter.process(img, param)
expected = img * (2 ** 2.0)  # 2^2 = 4x brighter

print(f"参数测试通过: {torch.allclose(result, expected)}")
```

## 训练建议

### 1. 调整超参数

由于参数范围变大，可能需要调整：

```bash
# 增大探索噪声（因为范围更大）
--param_noise_std 5.0  # 原来是 0.02，现在建议 5.0

# 可能需要调整学习率
--agent_lr 5e-4
--value_lr 5e-5
```

### 2. 监控训练

观察训练日志中的参数值，应该看到：
- Exposure 在 -5 到 +5 之间
- Saturation/Contrast 在 -100 到 +100 之间
- Sharpness 在 0 到 150 之间

### 3. 初始化参数

建议设置合理的初始化：

```python
# 在 Filter 中设置 init_params
ExposureFilter.init_params = [0.0]      # 0 EV
SaturationFilter.init_params = [0.0]   # 不改变饱和度
ContrastFilter.init_params = [0.0]     # 不改变对比度
```

## 常见问题

### Q1: 旧模型还能用吗？

A: 旧模型使用旧参数范围训练，**不能**直接用于新范围。需要重新训练。

### Q2: 参数范围变大会影响训练吗？

A: 可能需要调整噪声标准差和学习率。建议：
- `param_noise_std`: 5.0（原来0.02）
- `target_policy_noise`: 10.0（原来0.04）

### Q3: 如何验证参数是否正确？

A: 查看保存的参数文件，参数值应该在Lightroom范围内：
```python
# 正确的参数示例
{
  "ExposureFilter": [1.5],      # -5 到 +5 之间 ✓
  "SaturationFilter": [30.0],   # -100 到 +100 之间 ✓
  "SharpenFilter": [75.0]       # 0 到 150 之间 ✓
}
```

### Q4: 渲染效果会改变吗？

A: 不会。虽然参数范围变了，但渲染算法是一样的。
例如：+2 EV的效果（无论旧范围还是新范围）都是图像亮度增加4倍。

### Q5: 需要重新准备数据集吗？

A: 不需要。数据集不变，只是模型学习的参数范围改变了。

## 总结

✅ **已完成**：
- 修改配置文件，定义Lightroom参数范围
- 修改所有Filter的参数范围
- 修改渲染逻辑，正确处理Lightroom范围参数
- 保持向后兼容性

✅ **现在可以**：
- 重新训练模型，学习Lightroom标准范围参数
- 直接理解模型输出的参数值
- 与Lightroom进行直接对比

✅ **下一步**：
1. 运行 `bash bash/train.sh` 开始训练
2. 监控参数值，确保在Lightroom范围内
3. 调整噪声标准差等超参数（如需要）

---

**注意**：这是对渲染引擎的底层修改，需要重新训练模型。旧模型不兼容新范围。
