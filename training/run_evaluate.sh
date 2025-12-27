#!/bin/bash
# Isaac Sim评估脚本启动器
# 用法: bash run_evaluate.sh <checkpoint_path> [num_episodes] [steps_per_episode]

if [ -z "$1" ]; then
    echo "用法: bash run_evaluate.sh <checkpoint_path> [num_episodes] [steps_per_episode]"
    echo "示例: bash run_evaluate.sh ./checkpoints_bc_managed/best.pt 20 200"
    exit 1
fi

CHECKPOINT_PATH="$1"
NUM_EPISODES="${2:-20}"
STEPS_PER_EPISODE="${3:-200}"

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: checkpoint文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

# 检查Isaac Sim Python是否存在
ISAAC_SIM_PYTHON="${HOME}/isaacsim/python.sh"
if [ ! -f "$ISAAC_SIM_PYTHON" ]; then
    echo "错误: 找不到Isaac Sim Python: $ISAAC_SIM_PYTHON"
    echo "请确认Isaac Sim已正确安装"
    exit 1
fi

echo "=========================================="
echo "BC模型评估"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Episodes: $NUM_EPISODES"
echo "Steps per episode: $STEPS_PER_EPISODE"
echo "=========================================="
echo ""

# 使用Isaac Sim的Python运行评估脚本
"$ISAAC_SIM_PYTHON" /home/alphatok/ME5400/training/evaluate_bc.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --num_episodes "$NUM_EPISODES" \
    --steps_per_episode "$STEPS_PER_EPISODE"



