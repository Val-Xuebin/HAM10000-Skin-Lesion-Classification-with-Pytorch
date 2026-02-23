#!/bin/bash
# Kaggle API 配置脚本

echo "Kaggle API 配置助手"
echo "==================="
echo ""
echo "请选择配置方式："
echo "1. 使用环境变量（临时，仅当前会话有效）"
echo "2. 创建 kaggle.json 文件（永久配置）"
echo ""
read -p "请选择 (1 或 2): " choice

if [ "$choice" = "1" ]; then
    read -p "请输入你的 Kaggle 用户名: " username
    read -p "请输入你的 Kaggle API Key: " key
    export KAGGLE_USERNAME="$username"
    export KAGGLE_KEY="$key"
    echo ""
    echo "✓ 环境变量已设置（仅当前终端会话有效）"
    echo "现在可以运行: python main.py --mode train"
    
elif [ "$choice" = "2" ]; then
    read -p "请输入你的 Kaggle 用户名: " username
    read -p "请输入你的 Kaggle API Key: " key
    
    mkdir -p ~/.kaggle
    cat > ~/.kaggle/kaggle.json << EOF
{
  "username": "$username",
  "key": "$key"
}
EOF
    chmod 600 ~/.kaggle/kaggle.json
    echo ""
    echo "✓ kaggle.json 文件已创建在 ~/.kaggle/kaggle.json"
    echo "现在可以运行: python main.py --mode train"
else
    echo "无效的选择"
    exit 1
fi

