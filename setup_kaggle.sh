#!/bin/bash
# Kaggle API setup script

echo "Kaggle API Setup"
echo "================"
echo ""
echo "Choose configuration method:"
echo "1. Use environment variables (temporary, current session only)"
echo "2. Create kaggle.json file (persistent)"
echo ""
read -p "Choose (1 or 2): " choice

if [ "$choice" = "1" ]; then
    read -p "Enter your Kaggle username: " username
    read -p "Enter your Kaggle API key: " key
    export KAGGLE_USERNAME="$username"
    export KAGGLE_KEY="$key"
    echo ""
    echo "Environment variables set (current terminal session only)."
    echo "You can run: python main.py --mode train"
    
elif [ "$choice" = "2" ]; then
    read -p "Enter your Kaggle username: " username
    read -p "Enter your Kaggle API key: " key
    
    mkdir -p ~/.kaggle
    cat > ~/.kaggle/kaggle.json << EOF
{
  "username": "$username",
  "key": "$key"
}
EOF
    chmod 600 ~/.kaggle/kaggle.json
    echo ""
    echo "kaggle.json created at ~/.kaggle/kaggle.json"
    echo "You can run: python main.py --mode train"
else
    echo "Invalid choice."
    exit 1
fi
