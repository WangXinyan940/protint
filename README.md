# ProtInt

**ProtInt** 是一个基于深度学习的抗原 - 抗体相互作用预测工具。它结合了蛋白质语言模型 (ESM-C) 和图神经网络 (ProteinMPNN + Graph Transformer) 来预测抗原与抗体之间的结合能力。

## 目录

- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [模型结构](#模型结构)
- [使用指南](#使用指南)
- [项目结构](#项目结构)

---

## 特性

- **多模态特征融合**: 结合序列嵌入 (ESM-C) 和结构嵌入 (ProteinMPNN)
- **IMGT 注释**: 支持抗体 CDR 区域自动注释 (FR1/FR2/FR3/FR4, CDR1/CDR2/CDR3)
- **Graph Transformer 架构**: 基于注意力机制的图神经网络
- **Target Attention**: 抗体节点对抗原表示的定向注意力机制
- **灵活的训练流程**: 基于 PyTorch Lightning，支持 GPU/TPU 加速
- **多种输出格式**: 支持 pickle (.pkl)、CSV、JSON 格式输出预测结果

---

## 安装

### 使用 pip

```bash
pip install -e .
```

### 依赖

- PyTorch >= 2.0
- PyTorch Lightning >= 2.0
- PyTorch Geometric
- ESM (Meta AI)
- RDKit
- ANARCI (用于 IMGT 注释)

---

## 快速开始

### 1. 生成嵌入

```bash
protint embed \
  -i data/my_dataset \
  -o output/embeddings \
  --esm checkpoints/esm_c.pt \
  --mpnn checkpoints/protein_mpnn.pt
```

输入目录结构:
```
data/my_dataset/
├── antigen/
│   ├── antigen1.pdb
│   └── antigen2.pdb
├── antibody/
│   ├── antibody1.pdb
│   └── antibody2.pdb
└── targets.csv
```

### 2. 训练模型

```bash
protint train \
  --data-dir output/embeddings \
  --epochs 100 \
  --batch-size 1 \
  --lr 1e-4 \
  --checkpoint-dir checkpoints
```

### 3. 预测

```bash
# 单个样本
protint predict \
  -c checkpoints/best-epoch=50-val_loss=0.123.ckpt \
  -i output/embeddings/sample.pkl \
  -o results.json

# 批量预测 (CSV 格式输出)
protint predict \
  -c checkpoints/best.ckpt \
  -i output/embeddings \
  -o results.csv
```

---

## 模型结构

ProtInt 的模型架构分为以下几个关键组件：

### 1. 输入特征

每个残基的特征由以下部分组成 (总计 1098 维):

| 特征来源 | 维度 | 说明 |
|---------|------|------|
| ESM-C (ESM3) | 960 | 蛋白质语言模型序列嵌入 |
| ProteinMPNN | 128 | 图神经网络结构嵌入 |
| IMGT Region | 7 | CDR/FR 区域 one-hot 编码 |
| Chain Type | 3 | 轻链/重链/非抗体 one-hot 编码 |

### 2. 图 Transformer 编码器

抗原和抗体分别通过相同的 Graph Transformer 编码器:

```
输入投影 → Graph Transformer Layer (×3) → LayerNorm → 编码节点嵌入
```

每层 Graph Transformer 使用:
- 多头注意力机制 (默认 8 头)
- 残差连接
- 边特征增强的注意力计算

### 3. 抗原池化

抗原分支采用简单的 sum pooling:

```
编码节点嵌入 → Sum Pooling → 线性投影 → 抗原向量 (128 维)
```

### 4. Target Attention

抗体分支使用 Target Attention 机制，使抗体节点关注抗原的整体表示:

```
抗体编码节点嵌入 ─┬→ Target Attention (Q)
                  │
抗原向量 ─────────┴→ Target Attention (K, V)
                       ↓
                   Sum Pooling → 线性投影 → 抗体向量 (128 维)
```

### 5. 预测头

```
[抗原向量 || 抗体向量] → MLP → 结合概率
```

MLP 结构：Linear(256→128) → ReLU → Dropout → Linear(128→1)

---

## 使用指南

### 嵌入生成 (`embed`)

**功能**: 从 PDB 文件生成蛋白质嵌入

```bash
protint embed -i <输入目录> -o <输出目录> --esm <ESM 模型路径> --mpnn <MPNN 模型路径>
```

**参数**:
| 参数 | 说明 |
|------|------|
| `-i, --input` | 包含抗原/抗体 PDB 文件和 targets.csv 的目录 |
| `-o, --output` | 输出嵌入 (.pkl) 和 train/val.csv 的目录 |
| `--esm` | ESM-C 预训练模型路径 |
| `--mpnn` | ProteinMPNN 预训练模型路径 |
| `--val-ratio` | 验证集比例 (默认 0.2) |
| `--seed` | 随机种子 (默认 42) |

**输出**:
- `antigen_name.pkl`: 每个抗原的嵌入文件
- `antibody_name.pkl`: 每个抗体的嵌入文件
- `train.csv`: 训练集配对信息
- `val.csv`: 验证集配对信息

### 训练 (`train`)

**功能**: 训练抗原 - 抗体结合预测模型

```bash
protint train --data-dir <数据目录> [选项]
```

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | - | 包含 pkl 文件和 train.csv 的目录 |
| `--val-data-dir` | - | 验证数据目录 (可选) |
| `--node-input-dim` | 1098 | 节点输入维度 |
| `--edge-input-dim` | 128 | 边输入维度 |
| `--hidden-dim` | 128 | 隐藏层维度 |
| `--num-heads` | 8 | 注意力头数 |
| `--num-layers` | 3 | Graph Transformer 层数 |
| `--dropout` | 0.1 | Dropout 比率 |
| `--batch-size` | 1 | 批次大小 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--weight-decay` | 1e-2 | 权重衰减 |
| `--accelerator` | auto | 加速器类型 (cpu/gpu/auto) |
| `--checkpoint-dir` | checkpoints | 检查点保存目录 |

### 预测 (`predict`)

**功能**: 使用训练好的模型进行预测

```bash
protint predict -c <检查点> -i <输入> [-o <输出>]
```

**参数**:
| 参数 | 说明 |
|------|------|
| `-c, --checkpoint` | 模型检查点路径 |
| `-i, --input` | 输入文件/目录 (支持 pkl 文件或包含 pairs.csv 的目录) |
| `-o, --output` | 输出文件路径 (格式由扩展名决定) |
| `--device` | 推理设备 (cpu/cuda) |

**输出格式** (由文件扩展名自动决定):
| 扩展名 | 格式 | 说明 |
|--------|------|------|
| `.pkl` | Pickle | 二进制格式，包含所有数据 |
| `.csv` | CSV | 人类可读，不含向量列 |
| `.json` | JSON | 人类可读，包含完整数据 |

**输出字段**:
- `classification_prob`: 结合概率 (0-1)
- `antigen_vec`: 抗原表示向量 (128 维)
- `antibody_vec`: 抗体表示向量 (128 维)
- `pair_id`: 配对标识符 (批量预测时)
- `antigen`/`antibody`: 抗原/抗体名称 (批量预测时)

---

## 项目结构

```
protint/
├── src/protint/
│   ├── cli.py              # 命令行入口
│   ├── embedding/          # 特征提取模块 (已迁移至 dataset/)
│   ├── dataset/
│   │   ├── parse.py        # PDB 文件解析
│   │   ├── gen_embed.py    # 嵌入生成 (ESM-C + ProteinMPNN)
│   │   ├── imgt_annotator.py  # IMGT 注释 (CDR/FR 区域)
│   │   └── dataloader.py   # PyTorch DataLoader
│   ├── model/
│   │   ├── model.py        # 主模型定义
│   │   ├── layers.py       # 自定义层 (Graph Transformer, Target Attention)
│   │   └── submodules/
│   │       ├── esm_c_embed.py      # ESM-C 模型加载
│   │       ├── protein_mpnn_embed.py  # ProteinMPNN 加载
│   │       └── protein_mpnn_utils.py  # ProteinMPNN 工具函数
│   └── workflow/
│       ├── train.py        # PyTorch Lightning 训练流程
│       └── predict.py      # 推理和预测
├── checkpoints/            # 预训练模型和检查点
├── docs/plans/             # 设计文档
└── pyproject.toml
```

### 数据流

```
PDB 文件 → 解析 → 序列/结构提取
           ↓
    ┌──────┴──────┐
    ↓             ↓
ESM-C 嵌入    ProteinMPNN 嵌入
(960 维)        (128 维)
    ↓             ↓
    └──────┬──────┘
           ↓
    + IMGT 注释 (7+3 维)
           ↓
    拼接节点特征 (1098 维)
           ↓
    Graph Transformer 编码
           ↓
    ┌──────┴──────┐
    ↓             ↓
抗原 Sum Pooling  抗体 Target Attention
    ↓             ↓
    └──────┬──────┘
           ↓
    拼接 → 预测头 → 结合概率
```

---

## 许可证

MIT License
