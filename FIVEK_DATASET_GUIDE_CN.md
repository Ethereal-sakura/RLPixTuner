# MIT-Adobe FiveK 数据集准备指南

本指南详细说明如何下载、处理和准备MIT-Adobe FiveK数据集用于RLPixTuner训练。

## 数据集概述

MIT-Adobe FiveK数据集是图像润饰领域的著名资源，包含：
- **5,000张照片**：由多位摄影师使用单反相机拍摄
- **专家润饰结果**：提供了五位专家（Experts A-E）的润饰版本
- **原始格式**：提供RAW格式和处理后的图像

### 数据划分
根据论文，我们使用以下划分：
- **训练集**：4,500张图像
- **验证集**：500张图像

## 步骤1：下载数据集

### 官方数据集下载

MIT-Adobe FiveK数据集可以从以下位置下载：

**官方网站**：https://data.csail.mit.edu/graphics/fivek/

数据集包含多个部分：
1. **原始RAW文件**（DNG格式，约50GB）
2. **专家润饰参数**（Lightroom预设）
3. **预处理图像**（可选，约8GB）

### 推荐下载方式

由于完整数据集较大，推荐下载以下文件：

```bash
# 创建数据目录
mkdir -p ~/datasets/fivek
cd ~/datasets/fivek

# 下载方式1：使用wget下载DNG文件（推荐）
# 注意：需要从官方网站获取实际下载链接
wget <官方DNG文件下载链接>

# 下载方式2：如果已有预处理的TIF文件
wget <预处理TIF文件下载链接>

# 下载专家润饰参数
wget <Lightroom预设下载链接>
```

**注意**：由于数据集较大，下载可能需要数小时。建议使用稳定的网络连接。

### 替代方案：使用预处理数据

如果您在论文中使用了特定专家的润饰版本，可以下载已处理好的图像对：

```bash
# 示例：下载Expert C的润饰结果
# （具体链接请参考论文提供的数据）
```

## 步骤2：数据预处理

下载完成后，需要将RAW文件转换为训练所需的格式。

### 2.1 提取原始图像和目标图像

如果使用DNG格式，需要进行以下处理：

```python
import rawpy
import imageio
import os
from pathlib import Path

def process_dng_files(input_dir, output_dir, expert='C'):
    """
    处理DNG文件，提取输入图像和专家润饰后的目标图像
    
    参数:
        input_dir: DNG文件所在目录
        output_dir: 输出目录
        expert: 使用哪位专家的润饰 ('A', 'B', 'C', 'D', 'E')
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个DNG文件
    for dng_file in input_dir.glob('*.dng'):
        print(f"Processing {dng_file.name}...")
        
        # 读取RAW文件
        with rawpy.imread(str(dng_file)) as raw:
            # 提取输入图像（基础处理，无自动增强）
            input_img = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                output_bps=16,
                use_camera_wb=True
            )
            
            # 保存输入图像
            input_path = output_dir / f"{dng_file.stem}-Input.tif"
            imageio.imsave(str(input_path), input_img)
        
        # 注意：目标图像需要使用Lightroom预设或已提供的润饰版本
        # 这里需要根据实际情况调整

# 使用示例
process_dng_files(
    input_dir='~/datasets/fivek/raw',
    output_dir='~/datasets/fivek/processed'
)
```

### 2.2 应用专家润饰参数

如果您有Lightroom预设文件，可以使用以下工具：

```bash
# 使用Adobe DNG Converter或Lightroom批处理
# 或者使用开源工具如darktable

# 使用darktable命令行工具（Linux/Mac）
for file in *.dng; do
    darktable-cli $file expert_c_preset.xmp output/${file%.dng}.tif
done
```

## 步骤3：组织数据集结构

处理完成后，按以下结构组织数据集：

```
/path/to/your/dataset/
├── train/
│   ├── ExpertC0001-Input.tif
│   ├── ExpertC0001-Target.tif
│   ├── ExpertC0002-Input.tif
│   ├── ExpertC0002-Target.tif
│   └── ...（4,500对图像）
└── val/
    ├── ExpertC4501-Input.tif
    ├── ExpertC4501-Target.tif
    ├── ExpertC4502-Input.tif
    ├── ExpertC4502-Target.tif
    └── ...（500对图像）
```

### 命名规范

- **输入图像**：`ExpertC{编号}-Input.tif` 或 `ExpertC{编号}-0-Best-Input.tif`
- **目标图像**：`ExpertC{编号}-Target.tif` 或 `ExpertC{编号}-0-Best-Target.tif`

其中：
- `ExpertC`：表示使用Expert C的润饰标准
- `{编号}`：从0001到5000的图像编号
- 前4500张放入`train/`目录
- 后500张放入`val/`目录

### 自动化划分脚本

```python
import shutil
from pathlib import Path

def split_dataset(source_dir, dest_dir, train_count=4500):
    """
    将数据集划分为训练集和验证集
    
    参数:
        source_dir: 源数据目录
        dest_dir: 目标数据目录
        train_count: 训练集图像数量（默认4500）
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    # 创建目录
    train_dir = dest_dir / 'train'
    val_dir = dest_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有输入和目标图像对
    input_files = sorted(source_dir.glob('*-Input.tif'))
    
    for idx, input_file in enumerate(input_files, 1):
        # 找到对应的目标文件
        base_name = input_file.stem.replace('-Input', '')
        target_file = source_dir / f"{base_name}-Target.tif"
        
        if not target_file.exists():
            print(f"Warning: Target file not found for {input_file.name}")
            continue
        
        # 决定放入训练集还是验证集
        if idx <= train_count:
            dest = train_dir
        else:
            dest = val_dir
        
        # 复制文件
        shutil.copy2(input_file, dest / input_file.name)
        shutil.copy2(target_file, dest / target_file.name)
        
        if idx % 100 == 0:
            print(f"Processed {idx} image pairs...")
    
    print(f"Dataset split complete!")
    print(f"Training set: {train_count} pairs")
    print(f"Validation set: {len(input_files) - train_count} pairs")

# 使用示例
split_dataset(
    source_dir='~/datasets/fivek/processed',
    dest_dir='~/datasets/fivek/split'
)
```

## 步骤4：数据预处理（训练前准备）

在开始训练前，运行数据预处理脚本：

```bash
cd /workspace

# 预处理训练数据
python tools/process_data.py /path/to/your/dataset/train

# 预处理验证数据
python tools/process_data.py /path/to/your/dataset/val
```

这个脚本会：
- 验证图像对是否正确匹配
- 生成必要的元数据文件
- 进行数据格式检查

## 步骤5：配置训练脚本

更新训练脚本中的数据集路径：

```bash
# 编辑 bash/train.sh
vim bash/train.sh

# 修改以下参数：
--dataset_dir "/path/to/your/dataset"  # 替换为您的实际路径
```

## 步骤6：开始训练

一切准备就绪后，运行训练：

```bash
# 确保使用累积渲染模式（isp_inp_original=True）
bash bash/train.sh
```

## 数据集格式要求

### 图像格式
- **格式**：TIF、PNG或JPG（推荐TIF以保持质量）
- **位深度**：16-bit或8-bit
- **色彩空间**：sRGB
- **分辨率**：原始分辨率（训练时会自动resize到256x256或512x512）

### 文件大小参考
- 单张16-bit TIF图像：约10-20MB
- 完整训练集（4500对）：约90-180GB
- 完整验证集（500对）：约10-20GB

## 常见问题

### Q1: 下载速度太慢怎么办？
A: 可以使用以下方法：
1. 使用下载工具如aria2c进行多线程下载
2. 考虑使用学术网络或VPN
3. 联系数据集提供者获取替代下载方式

### Q2: 没有Lightroom如何处理专家润饰？
A: 可以：
1. 使用开源工具darktable（支持XMP预设）
2. 下载已经处理好的预处理版本
3. 使用RawTherapee等替代软件

### Q3: 可以使用其他专家的润饰吗？
A: 可以，只需在处理时选择对应的专家（A、B、C、D或E）。论文中通常使用Expert C。

### Q4: 训练集和验证集如何划分？
A: 
- 按顺序取前4500张作为训练集
- 后500张作为验证集
- 也可以随机划分，但要保证可重复性

### Q5: 数据增强需要在预处理时做吗？
A: 不需要，数据增强会在训练过程中自动进行。

## 磁盘空间要求

- **原始DNG文件**：约50GB
- **处理后的TIF文件**：约200GB
- **最终训练数据**：约200GB
- **建议预留空间**：至少300GB

## 性能优化建议

1. **使用SSD存储训练数据**：可显著提升数据加载速度
2. **预先resize图像**：如果训练时固定使用256x256，可以预先resize节省时间
3. **使用内存映射**：对于大数据集，考虑使用mmap加载

## 检查数据集是否准备正确

运行以下脚本验证数据集：

```python
import os
from pathlib import Path

def verify_dataset(dataset_dir):
    """验证数据集结构"""
    dataset_dir = Path(dataset_dir)
    
    for split in ['train', 'val']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"❌ Missing {split} directory")
            continue
        
        input_files = list(split_dir.glob('*-Input.tif'))
        target_files = list(split_dir.glob('*-Target.tif'))
        
        print(f"\n{split.upper()} Set:")
        print(f"✓ Input files: {len(input_files)}")
        print(f"✓ Target files: {len(target_files)}")
        
        # 检查配对
        unpaired = 0
        for input_file in input_files:
            base = input_file.stem.replace('-Input', '')
            target_file = split_dir / f"{base}-Target.tif"
            if not target_file.exists():
                print(f"❌ Missing target for {input_file.name}")
                unpaired += 1
        
        if unpaired == 0:
            print(f"✓ All files properly paired")
        else:
            print(f"❌ {unpaired} unpaired files")

# 使用
verify_dataset('/path/to/your/dataset')
```

## 参考资料

- [MIT-Adobe FiveK 官方网站](https://data.csail.mit.edu/graphics/fivek/)
- [论文：Learning Photographic Global Tonal Adjustment with a Database of Input / Output Image Pairs](http://people.csail.mit.edu/vladb/photoadjust/)
- [RLPixTuner 论文](https://openreview.net/pdf?id=4kVHI2uXRE)

## 联系与支持

如果在数据集准备过程中遇到问题，可以：
1. 查看项目的GitHub Issues
2. 参考论文的补充材料
3. 联系数据集作者

---

**注意**：本指南基于论文描述和常见实践编写。实际数据集格式可能因版本而异，请以官方文档为准。
