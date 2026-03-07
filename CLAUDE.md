# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供在此仓库中工作的指导。

## 项目目标

本项目旨在构造一个基于结构的神经网络模型，预测抗原 - 抗体相互作用，并管理它的训练、评估和推理。

**模型结构：**
1. **Feature 模块** - 使用 GNN 建模蛋白
   - 图结构和 edge features 来自 ProteinMPNN
   - node features = ESM-C 300M (ESM3) 序列嵌入 + ProteinMPNN 图嵌入（拼接）
2. **模型模块**
   - Graph Transformer 三层堆叠作为主干网络，分别得到抗原和抗体的嵌入表示
   - 抗原部分：sum pooling 成抗原特征 vector
   - 抗体部分：node features 作为序列，与抗原 vector 做 target attention，然后 sum pooling 成抗体特征 vector
   - 预测头：分类/回归任务
3. **训练模块** - 使用 PyTorch Lightning 进行训练和评估
4. **推理模块** - 使用训练好的模型进行预测

三部分都提供命令行工具，用户可以通过命令行接口来执行不同的任务。

## 构建与开发命令

```bash
# 安装/更新依赖
pixi install

# 生成嵌入
protint embed -i <pdb 文件夹> -o <输出文件夹> --esm <esm 模型路径> --mpnn <mpnn 模型路径>
```

## 架构概览

**包结构:** `src/protint/`

- **`cli.py`** - CLI 入口点，包含 `embed` 命令用于批量处理 PDB 文件
- **`embedding/`** - 特征提取模块
  - `seq_feat.py` - 通过 ESM-C (ESM3) 生成序列嵌入
  - `graph_feat.py` - 通过 ProteinMPNN 生成图嵌入
- **`model/`** - 神经网络组件
  - `submodules/` - 预训练模型加载和前向传播
    - `esm_c_embed.py` - ESM-C 模型加载和嵌入生成
    - `protein_mpnn_embed.py` - ProteinMPNN 模型加载和前向传播
    - `protein_mpnn_utils.py` - ProteinMPNN 工具（PDB 解析、特征化、模型架构）
  - `layers.py` - 自定义网络层（Graph Transformer、target attention 等）
  - `model.py` - 主干网络和预测头定义
- **`dataset/`** - 数据处理
  - `parse.py` - PDB 文件解析
  - `gen_embed.py` - 结合序列嵌入和图嵌入
- **`workflow/`** - PyTorch Lightning 训练/评估/推理流程

## 关键数据流

### 嵌入生成
1. 解析 PDB 文件，提取每条链的坐标和序列
2. ESM-C 生成序列嵌入（每个残基 960 维）
3. ProteinMPNN 生成图嵌入（节点和边特征）
4. 输出：`node_features`（拼接后）、`edge_features`、`edge_indices`

### 模型前向传播
1. Graph Transformer 处理抗原和抗体图
2. 抗原 sum pooling → 抗原 vector
3. 抗体 target attention → sum pooling → 抗体 vector
4. 预测头输出分类/回归结果

## 依赖

- **ESM3** (`esm` 包) - 序列嵌入模型
- **ProteinMPNN** (vendored 在 `model/submodules/`) - 图嵌入模型
- **RDKit** - 化学工具（via pixi）
- **PyTorch / PyTorch Lightning** - 深度学习框架

## 注意事项

**模型参数** feature构造所需的基模参数存储在根目录下的`checkpoints/`中。

**Plan 撰写** 每次做Brainstorm或做Plan时，都在`docs/plans/`目录下创建一个新的 Markdown 文件，然后基于文件与用户迭代。用户会阅读Plan并提供反馈，直到Plan完善为止。
