# Lightroom参数映射使用指南

## 概述

本系统现在支持将模型输出的参数保存为Lightroom风格的格式，使参数更加直观易懂，便于：
1. **理解模型输出**：使用标准摄影术语（曝光、饱和度、对比度等）
2. **与Lightroom对比**：参数范围对齐到Lightroom的标准范围
3. **手动调整**：可以手动编辑Lightroom格式的参数并重新应用
4. **跨平台兼容**：导出的参数可用于其他支持类似范围的工具

## 参数范围映射

### 当前支持的Filter及其Lightroom映射

| Filter | 内部参数 | Lightroom参数 | Lightroom范围 |
|--------|---------|--------------|--------------|
| **ExposureFilter** | exposure | Exposure (EV) | -5.0 to +5.0 |
| **ImprovedWhiteBalanceFilter** | wb_r, wb_g, wb_b | WB R/G/B multipliers | 0.5 to 2.0 |
| **SaturationFilter** | saturation | Saturation | -100 to +100 |
| **ContrastFilter** | contrast | Contrast | -100 to +100 |
| **HighlightFilter** | highlight | Highlights | -100 to +100 |
| **ShadowFilter** | shadow | Shadows | -100 to +100 |
| **SharpenFilter** | sharpen | Sharpness | 0 to 150 |
| **ColorFilter** | color_curve | Color Curves (RGB×8) | -10 to +10 per step |
| **ToneFilter** | tone_curve | Tone Curve (8 steps) | -10 to +10 per step |

## 启用Lightroom参数保存

### 1. 训练时启用

在`bash/train.sh`中添加参数：

```bash
--use_lightroom_params True \
```

或直接使用（默认已启用）：

```bash
bash bash/train.sh
```

### 2. 评估时启用

```bash
# 编辑 bash/run_pft.sh
--use_lightroom_params True \

# 运行评估
bash bash/run_pft.sh
```

### 3. Python代码中使用

```python
from lightroom_params import (
    LightroomParamConverter, 
    save_params_lightroom, 
    load_params_lightroom,
    print_params_comparison
)

# 初始化环境时
env = ISPEnv(
    cfg, 
    isp_blocks=isp_blocks,
    use_lightroom_params=True,  # 启用Lightroom参数
    # ... 其他参数
)
```

## 输出格式

### Lightroom格式参数文件

保存的JSON文件包含两部分：

```json
{
  "lightroom_params": {
    "ExposureFilter": {
      "display_name": "Exposure",
      "unit": "EV",
      "parameters": {
        "exposure": 0.45
      }
    },
    "SaturationFilter": {
      "display_name": "Saturation",
      "unit": "",
      "parameters": {
        "saturation": 15.8
      }
    },
    "ContrastFilter": {
      "display_name": "Contrast",
      "unit": "",
      "parameters": {
        "contrast": -8.3
      }
    },
    ...
  },
  "internal_params": {
    "ExposureFilter": [0.059993743896484375],
    "SaturationFilter": [0.07925844192504883],
    "ContrastFilter": [-0.05569171905517578],
    ...
  }
}
```

### 参数解读

以上面的例子为例：

- **Exposure**: +0.45 EV → 增加约半档曝光
- **Saturation**: +15.8 → 增加饱和度约16%
- **Contrast**: -8.3 → 降低对比度约8%
- **Highlights**: +21.5 → 提亮高光区域
- **Shadows**: +0.4 → 轻微提亮阴影
- **Sharpness**: +2.7 → 轻微锐化

## 使用场景

### 场景1：查看模型学到的参数

```python
from lightroom_params import print_params_comparison

# 在评估后打印参数对比
print_params_comparison(params_list, isp_blocks)
```

输出示例：
```
================================================================================
参数对比：内部参数 vs Lightroom参数
================================================================================

【ExposureFilter】
  内部范围: [-2.000, 2.000]
  内部参数: [0.05999374]
  Lightroom参数:
    exposure: 0.450 EV

【SaturationFilter】
  内部范围: [-1.000, 1.000]
  内部参数: [0.07925844]
  Lightroom参数:
    saturation: 15.851 

【ContrastFilter】
  内部范围: [-1.000, 1.000]
  内部参数: [-0.05569172]
  Lightroom参数:
    contrast: -8.327 
```

### 场景2：手动调整参数

1. 导出Lightroom格式参数
2. 手动编辑JSON文件中的Lightroom参数
3. 重新加载并应用

```python
from lightroom_params import load_params_lightroom
import json

# 加载并修改参数
with open('params_lightroom.json', 'r') as f:
    data = json.load(f)

# 手动调整
data['lightroom_params']['SaturationFilter']['parameters']['saturation'] = 25.0
data['lightroom_params']['ExposureFilter']['parameters']['exposure'] = 1.0

# 保存修改
with open('params_lightroom_modified.json', 'w') as f:
    json.dump(data, f, indent=2)

# 加载为内部参数
params_list = load_params_lightroom('params_lightroom_modified.json', isp_blocks)

# 应用到图像
processed_img = isp_blocks.run(original_img, params_list)
```

### 场景3：批量处理并导出

```python
from lightroom_params import save_params_lightroom

# 处理多张图像
for img_path in image_paths:
    # ... 模型推理获取参数 ...
    
    # 保存Lightroom格式参数
    save_path = img_path.replace('.jpg', '_params_lightroom.json')
    save_params_lightroom(
        params_list, 
        isp_blocks, 
        save_path, 
        include_internal=True  # 同时保存内部参数以便对比
    )
```

## 高级用法

### 1. 自定义参数范围

如果您想使用不同的范围，可以修改`lightroom_params.py`中的`LIGHTROOM_RANGES`：

```python
LIGHTROOM_RANGES = {
    "ExposureFilter": {
        "exposure": (-10.0, 10.0),  # 扩大曝光范围
        "display_name": "Exposure",
        "unit": "EV"
    },
    # ... 其他filter
}
```

### 2. 添加新Filter的映射

```python
# 在 lightroom_params.py 中添加
LIGHTROOM_RANGES["YourCustomFilter"] = {
    "param_name": (min_value, max_value),
    "display_name": "Your Custom Filter",
    "unit": "custom_unit"
}
```

### 3. 只保存Lightroom参数（不保存内部参数）

```python
save_params_lightroom(
    params_list, 
    isp_blocks, 
    save_path, 
    include_internal=False  # 只保存Lightroom格式
)
```

## API 参考

### LightroomParamConverter 类

```python
converter = LightroomParamConverter(isp_blocks)

# 转换为Lightroom参数
lr_params = converter.internal_to_lightroom(params_list)

# 转换回内部参数
params_list = converter.lightroom_to_internal(lr_params)
```

### 保存函数

```python
save_params_lightroom(
    params_list,          # 内部参数列表
    isp_blocks,           # ISPBlocks实例
    save_path,            # 保存路径
    include_internal=True # 是否包含内部参数
)
```

### 加载函数

```python
params_list = load_params_lightroom(
    load_path,   # JSON文件路径
    isp_blocks   # ISPBlocks实例
)
```

### 打印对比函数

```python
print_params_comparison(
    params_list,  # 内部参数列表
    isp_blocks    # ISPBlocks实例
)
```

## 实际应用示例

### 示例1：分析模型在不同图像上的调整策略

```python
results = {}

for img_name, params_list in processed_images.items():
    converter = LightroomParamConverter(isp_blocks)
    lr_params = converter.internal_to_lightroom(params_list)
    
    results[img_name] = {
        'exposure': lr_params['ExposureFilter']['exposure'],
        'saturation': lr_params['SaturationFilter']['saturation'],
        'contrast': lr_params['ContrastFilter']['contrast']
    }

# 分析统计
import pandas as pd
df = pd.DataFrame(results).T
print(df.describe())
```

### 示例2：导出为Lightroom预设格式

虽然当前输出的是JSON，但可以很容易转换为Lightroom XMP格式：

```python
def export_to_xmp(lr_params, output_path):
    """将Lightroom参数导出为XMP预设（简化版）"""
    xmp_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
      <crs:Exposure2012>{lr_params['ExposureFilter']['exposure']}</crs:Exposure2012>
      <crs:Contrast2012>{lr_params['ContrastFilter']['contrast']}</crs:Contrast2012>
      <crs:Saturation>{lr_params['SaturationFilter']['saturation']}</crs:Saturation>
      <crs:Sharpness>{lr_params['SharpenFilter']['sharpen']}</crs:Sharpness>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""
    with open(output_path, 'w') as f:
        f.write(xmp_template)
```

## 常见问题

### Q1: 为什么某些参数看起来和Lightroom不完全一样？

A: 我们的ISP管道和Lightroom的实现可能有所不同，参数映射只是为了让数值更加直观。实际效果可能略有差异。

### Q2: 可以直接在Lightroom中使用这些参数吗？

A: JSON格式不能直接导入Lightroom，但可以作为参考手动调整。如需导入，需要转换为XMP格式（见示例2）。

### Q3: 内部参数和Lightroom参数哪个更准确？

A: 内部参数是模型实际使用的值，更准确。Lightroom参数只是一个更易读的表示方式。

### Q4: 如何关闭Lightroom参数保存？

A: 设置 `--use_lightroom_params False` 或在代码中设置 `use_lightroom_params=False`

### Q5: 能否添加自定义的参数范围？

A: 可以！修改`lightroom_params.py`中的`LIGHTROOM_RANGES`字典即可。

## 性能影响

- Lightroom参数转换是在CPU上进行的，对训练速度影响极小（<1%）
- 只在保存参数时进行转换，不影响训练过程
- JSON文件大小略大于原始参数文件（约2-3倍），但仍然很小（几KB）

## 更新日志

- **v1.0** (2026-02-09): 初始版本，支持7种常见Filter的Lightroom映射
- 支持双向转换（内部↔Lightroom）
- 支持参数对比打印
- 支持手动编辑和重新加载

---

**注意**：Lightroom参数映射是一个便利功能，旨在提高参数的可解释性。模型实际训练和推理仍然使用内部参数范围。
