# Using Ollama with AllSides Research

This guide explains how to set up Ollama to work with the AllSides Research feature, which allows you to run LLM models locally on your machine instead of using cloud API services.

## Why Use Ollama?

- **Cost-effective**: No usage fees or API costs
- **Privacy**: All data stays on your machine
- **Customization**: Use various open source models
- **Learning**: Great way to experiment with different models

## Prerequisites

1. A computer with sufficient resources:
   - **CPU-only**: Will work but be slow (minimum 16GB RAM recommended)
   - **GPU**: NVIDIA GPU with at least 8GB VRAM (12GB+ recommended for larger models)

## Installation

### 1. Install Ollama

Follow the instructions for your platform:
- **macOS/Linux/Windows**: [https://ollama.com/download](https://ollama.com/download)

### 2. Install the required Python packages

The AllSides Research feature requires specific versions of packages to work with Ollama:

```bash
pip install langchain-ollama==0.2.2 langchain_core>=0.1.20 pydantic>=1.10.8
```

### 3. Download the models

After installing Ollama, run these commands to download the supported models:

```bash
# Llama 3.2 (1.1GB)
ollama pull llama3.2

# Gemma 3 models
ollama pull gemma3:1b
ollama pull gemma3:4b
```

### 4. Configuration

#### Environment Variables

You can set the following environment variables:

```bash
# Optional: Set custom Ollama host (default is http://localhost:11434)
export OLLAMA_HOST=http://localhost:11434
```

## Using Ollama in the Research UI

1. Go to the "New Research" page
2. Enable "Advanced Configuration"
3. Select "Ollama (Local)" as the provider for "Planner Provider" and/or "Writer Provider"
4. Choose one of the available models (llama3.2, gemma3:1b, or gemma3:4b)

## Troubleshooting

### Common Issues

1. **"Failed to connect to Ollama server"**
   - Make sure Ollama is running (check the Ollama app or service)
   - Verify the correct OLLAMA_HOST environment variable if using a custom host

2. **"Model not found" errors**
   - Ensure you've downloaded the model using `ollama pull [model_name]`

3. **"NotImplementedError" or structured output errors**
   - This can happen if you're using an incompatible version of langchain packages
   - Ensure you've installed the required dependencies: `pip install langchain-ollama==0.2.2 langchain_core>=0.1.20 pydantic>=1.10.8`
   - Restart the server after installing dependencies

4. **Slow performance**
   - Local models are generally slower than cloud APIs, especially without a GPU
   - Models like gemma3:1b are faster but less capable than larger models

5. **Out of memory errors**
   - Try using a smaller model (gemma3:1b requires less memory)
   - Close other applications to free up system memory

### Logs

Check the server logs for more detailed information about Ollama-related issues:

- Look for lines with "Using custom Ollama implementation" to confirm the custom model is being used
- If you see "Failed to create custom Ollama chat model", make sure required packages are installed

## Technical Details

The AllSides Research feature uses the following methods to make Ollama work efficiently:

1. Uses a custom implementation for structured output instead of Ollama Functions
2. Modifies the Section model to use `Literal[True,False]` instead of `bool` to avoid encoding issues
3. Configures proper streaming for progress updates
4. Implements a JsonOutputParser approach for Ollama structured output

## Support

If you encounter issues with Ollama integration, please check the GitHub issues or contact support. 