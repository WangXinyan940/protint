#!/bin/bash
# =============================================================================
# ProtInt 完整示例：Embed -> Train -> Infer
# =============================================================================
# 本脚本演示如何使用 example/model-input-data 中的数据完成完整的训练和推理流程
# =============================================================================

set -e  # 遇到错误立即退出

export KMP_DUPLICATE_LIB_OK=TRUE

# -----------------------------------------------------------------------------
# 配置变量
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 数据路径
DATA_DIR="$SCRIPT_DIR/model-input-data"
OUTPUT_DIR="$SCRIPT_DIR/output"

# 模型路径
ESM_MODEL="$PROJECT_ROOT/checkpoints/esmc_300m_2024_12_v0.pth"
MPNN_MODEL="$PROJECT_ROOT/checkpoints/v_48_020.pt"

# 训练配置
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
BATCH_SIZE=2
EPOCHS=10
LEARNING_RATE=0.0001
HIDDEN_DIM=64
NUM_HEADS=4
NUM_LAYERS=2

# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------
print_header() {
    echo ""
    echo "============================================================================="
    echo "  $1"
    echo "============================================================================="
    echo ""
}

print_step() {
    echo ">>> $1"
    echo ""
}

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------

print_header "ProtInt 完整示例流程"

echo "项目根目录：$PROJECT_ROOT"
echo "数据目录：$DATA_DIR"
echo "输出目录：$OUTPUT_DIR"
echo ""

# 检查必要文件是否存在
print_step "检查必要文件"
for file in "$ESM_MODEL" "$MPNN_MODEL" "$DATA_DIR/targets.csv"; do
    if [ ! -f "$file" ]; then
        echo "错误：文件不存在 - $file"
        exit 1
    fi
    echo "  [OK] $file"
done

# 创建输出目录
print_step "创建输出目录"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"
echo "  输出目录：$OUTPUT_DIR"

# -----------------------------------------------------------------------------
# 步骤 1: 生成嵌入 (Embed)
# -----------------------------------------------------------------------------
print_header "步骤 1: 生成蛋白质嵌入 (Embed)"

EMBED_OUTPUT_DIR="$OUTPUT_DIR/embeddings"

print_step "运行 embed 命令生成嵌入"
echo "命令："
echo "  protint embed \\"
echo "    --input $DATA_DIR \\"
echo "    --output $EMBED_OUTPUT_DIR \\"
echo "    --esm $ESM_MODEL \\"
echo "    --mpnn $MPNN_MODEL"
echo ""

protint embed \
    --input "$DATA_DIR" \
    --output "$EMBED_OUTPUT_DIR" \
    --esm "$ESM_MODEL" \
    --mpnn "$MPNN_MODEL"

echo ""
echo "嵌入生成完成！"
echo "  输出目录：$EMBED_OUTPUT_DIR"
echo "  生成的文件："
ls -la "$EMBED_OUTPUT_DIR"

# -----------------------------------------------------------------------------
# 步骤 2: 训练模型 (Train)
# -----------------------------------------------------------------------------
print_header "步骤 2: 训练预测模型 (Train)"

print_step "运行 train 命令训练模型"
echo "命令："
echo "  protint train \\"
echo "    --train-dir $EMBED_OUTPUT_DIR \\"
echo "    --batch-size $BATCH_SIZE \\"
echo "    --epochs $EPOCHS \\"
echo "    --lr $LEARNING_RATE \\"
echo "    --hidden-dim $HIDDEN_DIM \\"
echo "    --num-heads $NUM_HEADS \\"
echo "    --num-layers $NUM_LAYERS \\"
echo "    --checkpoint-dir $CHECKPOINT_DIR"
echo ""

protint train \
    --train-dir "$EMBED_OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-heads "$NUM_HEADS" \
    --num-layers "$NUM_LAYERS" \
    --checkpoint-dir "$CHECKPOINT_DIR"

echo ""
echo "训练完成！"
echo "  检查点目录：$CHECKPOINT_DIR"
echo "  生成的检查点："
ls -la "$CHECKPOINT_DIR"

# 获取最佳检查点路径
BEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/*.ckpt 2>/dev/null | head -n 1)
if [ -z "$BEST_CHECKPOINT" ]; then
    echo "错误：未找到检查点文件"
    exit 1
fi
echo ""
echo "使用检查点：$BEST_CHECKPOINT"

# -----------------------------------------------------------------------------
# 步骤 3: 模型推理 (Infer)
# -----------------------------------------------------------------------------
print_header "步骤 3: 模型推理 (Infer)"

INFER_OUTPUT_DIR="$OUTPUT_DIR/predictions"
mkdir -p "$INFER_OUTPUT_DIR"

print_step "运行 predict 命令进行推理"
echo "命令："
echo "  protint predict \\"
echo "    --checkpoint $BEST_CHECKPOINT \\"
echo "    --input $EMBED_OUTPUT_DIR \\"
echo "    --output $INFER_OUTPUT_DIR/results.pkl"
echo ""

protint predict \
    --checkpoint "$BEST_CHECKPOINT" \
    --input "$EMBED_OUTPUT_DIR" \
    --output "$INFER_OUTPUT_DIR/results.pkl"

echo ""
echo "推理完成！"
echo "  结果文件：$INFER_OUTPUT_DIR/results.pkl"

# -----------------------------------------------------------------------------
# 完成
# -----------------------------------------------------------------------------
print_header "流程完成"

echo "所有步骤已成功完成！"
echo ""
echo "输出文件汇总："
echo "  嵌入文件：$EMBED_OUTPUT_DIR/"
echo "  模型检查点：$CHECKPOINT_DIR/"
echo "  预测结果：$INFER_OUTPUT_DIR/results.pkl"
echo ""
echo "使用以下命令查看预测结果："
echo "  python -c \"import pickle; print(pickle.load(open('$INFER_OUTPUT_DIR/results.pkl', 'rb')))\""
