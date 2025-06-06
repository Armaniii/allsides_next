#!/bin/bash

# Navigate to the backend directory
cd "$(dirname "$0")"

echo "Installing Open Deep Research and related dependencies..."

# Check environment variables
echo "Checking environment variables..."
echo "TAVILY_API_KEY present: ${TAVILY_API_KEY:+YES}"
echo "OPENAI_API_KEY present: ${OPENAI_API_KEY:+YES}"
echo "REDIS_HOST set to: ${REDIS_HOST:-localhost}"

# Install required packages
pip install -r requirements.txt

echo "Installation complete."
