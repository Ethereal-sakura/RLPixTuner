# 快速开始指南

## 数据集准备（简化版）

### 1. 下载FiveK数据集

```bash
# 创建数据目录
mkdir -p ~/datasets/fivek/train
mkdir -p ~/datasets/fivek/val

# 从官方或论文提供的链接下载预处理数据
# https://data.csail.mit.edu/graphics/fivek/
```

### 2. 组织数据结构

确保您的数据集结构如下：

```
~/datasets/fivek/
├── train/
│   ├── ExpertC0001-Input.tif
│   ├── ExpertC0001-Target.tif
│   └── ...（共4,500对图像）
└── val/
    ├── ExpertC4501-Input.tif
    ├── ExpertC4501-Target.tif
    └── ...（共500对图像）
```

### 3. 预处理数据

```bash
cd /workspace

# 处理训练和验证数据
python tools/process_data.py ~/datasets/fivek/train
python tools/process_data.py ~/datasets/fivek/val
```

### 4. 修改训练脚本路径

编辑 `bash/train.sh`：

```bash
vim bash/train.sh

# 将 --dataset_dir 修改为您的实际路径：
--dataset_dir "~/datasets/fivek" \
```

### 5. 开始训练

```bash
# 使用累积渲染模式训练（默认配置）
bash bash/train.sh
```

## 关键参数说明

### 累积渲染模式
- `--isp_inp_original True`：**累积渲染**，每次step从原始图像开始处理（默认，推荐）
- `--isp_inp_original False`：增量模式，每次step在上一次结果基础上处理

### 其他重要参数
- `--max_step 10`：每个episode的最大步数
- `--agent_lr 5e-4`：agent学习率
- `--value_lr 5e-5`：value网络学习率
- `--loss_type psnr`：损失类型（psnr/l1/l2）
- `--isp dataset`：ISP管道配置（根据数据集自动选择）

## 评估模型

```bash
# 修改 bash/run_pft.sh 中的路径
vim bash/run_pft.sh

# 设置模型路径和数据集路径：
--model_path "/path/to/your/trained/model.zip" \
--dataset_dir "~/datasets/fivek" \

# 运行评估
bash bash/run_pft.sh
```

## 数据集下载链接

- **官方网站**：https://data.csail.mit.edu/graphics/fivek/
- **论文示例数据**（子集）：见README中的Google Drive链接

## 完整文档

详细的数据集准备步骤请参考：[FIVEK_DATASET_GUIDE_CN.md](./FIVEK_DATASET_GUIDE_CN.md)

## 系统要求

- **GPU**：NVIDIA GPU with CUDA（建议RTX 3090或更好）
- **内存**：至少32GB RAM
- **存储**：至少300GB可用空间
- **Python**：3.9+

## 常见问题

### Q: 训练需要多长时间？
A: 在单个RTX 3090上，完整训练通常需要1-2天（80万steps）。

### Q: 可以使用更小的数据集吗？
A: 可以，但性能会下降。建议至少使用1000对图像。

### Q: 如何监控训练进度？
A: 训练日志会保存在 `experiments/__sb3_saved_eval/` 目录，可以使用TensorBoard查看。

```bash
tensorboard --logdir experiments/__sb3_saved_eval/your_experiment/
```

## 疑难解答

### 错误：找不到数据集
```
检查 --dataset_dir 路径是否正确
确保 train/ 和 val/ 目录存在
确保图像文件命名正确
```

### 错误：CUDA out of memory
```
减小 --batch_size（默认2）
减小 --env_img_sz（默认256）
```

### 错误：训练很慢
```
检查是否使用了GPU
确保数据集在SSD上
考虑减少 --obs_inp_laplacian 参数
```

## 下一步

- 调整超参数以获得更好的性能
- 尝试不同的ISP管道配置
- 在自己的数据集上训练

---

更多详细信息请参考完整文档和论文。
