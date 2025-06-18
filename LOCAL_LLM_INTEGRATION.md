# Local LLM Integration with Ollama and vLLM

This document provides comprehensive information about the local LLM integration implemented in the AllSides Next project using Ollama and vLLM services.

## Overview

The integration provides local Large Language Model capabilities for NLP tasks without relying solely on external APIs. This includes:

- **Ollama**: Easy-to-use local LLM server for general text generation
- **vLLM**: High-performance inference server with OpenAI-compatible API
- **CPU-only configuration**: Optimized for deployment without GPU requirements

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Django API    │    │     Ollama      │    │      vLLM       │
│   (Backend)     │◄──►│   (llama3.2)    │    │  (DialoGPT)     │
│                 │    │   Port: 11434   │    │   Port: 8001    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   LLMClient     │
│   (llm_client)  │
└─────────────────┘
```

## Services Configuration

### Ollama Service
- **Image**: `ollama/ollama:latest`
- **Port**: `11434:11434`
- **Model**: `llama3.2:1b` (CPU-optimized)
- **Resource Limits**: 2 CPUs, 4GB RAM

### vLLM Service  
- **Image**: `vllm/vllm-openai:latest`
- **Port**: `8001:8000`
- **Model**: `microsoft/DialoGPT-small`
- **Resource Limits**: 2 CPUs, 6GB RAM
- **API**: OpenAI-compatible endpoints

## Installation & Setup

### 1. Environment Variables

Add these to your `.env` file:

```bash
# Local LLM Configuration
OLLAMA_HOST=ollama:11434
OLLAMA_API_URL=http://ollama:11434/api
VLLM_API_URL=http://vllm:8000/v1
```

### 2. Docker Compose

The services are already configured in `docker-compose.yml`:

```yaml
ollama:
  image: ollama/ollama:latest
  container_name: allsides_ollama
  ports:
    - "11434:11434"
  # ... additional configuration

vllm:
  image: vllm/vllm-openai:latest  
  container_name: allsides_vllm
  ports:
    - "8001:8000"
  # ... additional configuration
```

### 3. Start Services

```bash
# Start all services including LLM
docker compose up -d

# Start only LLM services
docker compose up -d ollama vllm

# Check service status
docker compose ps
```

## API Endpoints

### LLM Status Endpoint

**GET** `/api/llm/status/`

Returns the current status of both LLM services.

```json
{
  "available": true,
  "services": {
    "ollama": {
      "available": true,
      "url": "http://ollama:11434/api",
      "default_model": "llama3.2:1b"
    },
    "vllm": {
      "available": true, 
      "url": "http://vllm:8000/v1",
      "model": "dialogpt-small"
    }
  },
  "last_check": 1640995200
}
```

### LLM Test Endpoint

**POST** `/api/llm/test/`

Test the LLM services with sample queries.

```json
{
  "query": "What are renewable energy benefits?",
  "argument": "Solar power reduces emissions",
  "stance": "Pro-Renewable Energy"
}
```

Response:
```json
{
  "test_query": "What are renewable energy benefits?",
  "query_enhancement": {
    "enhanced": true,
    "keywords": ["renewable", "energy", "benefits", "environment"],
    "sentiment": "neutral",
    "summary": "Query about renewable energy advantages"
  },
  "argument_analysis": {
    "analyzed": true,
    "stance": "Pro-Renewable Energy",
    "key_points": ["solar", "emissions", "reduction"],
    "sentiment": "positive"
  },
  "llm_status": { /* ... status object ... */ }
}
```

## Usage in Code

### Basic Text Generation

```python
from api.llm_client import llm_client

# Generate text with automatic service selection
result = llm_client.generate_text(
    prompt="Explain the benefits of renewable energy",
    max_tokens=200,
    prefer_service="ollama"  # or "vllm"
)
print(result)
```

### NLP Helper Functions

```python
from api.main_v3 import enhance_query_with_local_llm, analyze_argument_with_local_llm

# Enhance query analysis
enhancement = enhance_query_with_local_llm("Climate change debate")
print(enhancement['keywords'])  # ['climate', 'change', 'debate', 'environment']

# Analyze argument quality
analysis = analyze_argument_with_local_llm(
    "Renewable energy reduces carbon emissions significantly",
    "Pro-Environment"
)
print(analysis['sentiment'])  # 'positive'
```

### Service-Specific Usage

```python
# Ollama (better for longer text generation)
result = llm_client.generate_with_ollama(
    prompt="Write a summary of climate change impacts",
    model="llama3.2:1b",
    max_tokens=300
)

# vLLM (better for quick responses, chat format)
messages = [
    {"role": "user", "content": "What is renewable energy?"}
]
result = llm_client.chat_with_vllm(messages, max_tokens=150)
```

## Available NLP Functions

### Query Enhancement
- **extract_keywords()**: Extract key terms from text
- **classify_sentiment()**: Determine positive/negative/neutral sentiment  
- **summarize_text()**: Generate concise summaries

### Service Management
- **get_service_status()**: Check availability of both services
- **ensure_ollama_model()**: Download/verify Ollama models
- **_check_service_health()**: Periodic health monitoring

## Model Configuration

### Ollama Models

Default model: `llama3.2:1b` (CPU-optimized, ~1.3GB)

To use different models:

```bash
# Access Ollama container
docker compose exec ollama bash

# Pull a different model
ollama pull llama3.2:3b

# List available models
ollama list
```

### vLLM Models

Default model: `microsoft/DialoGPT-small` (CPU-compatible)

To change models, update `docker-compose.yml`:

```yaml
vllm:
  command: >
    --model microsoft/DialoGPT-medium
    --served-model-name dialogpt-medium
    # ... other flags
```

## Performance Considerations

### CPU-Only Deployment
- **Models**: Optimized for CPU inference
- **Memory**: 4-6GB allocated per service
- **Response Time**: 2-10 seconds depending on query complexity
- **Concurrency**: Limited to 1-2 parallel requests

### Optimization Tips
1. **Use smaller models** for better performance
2. **Limit max_tokens** to reduce generation time
3. **Implement caching** for repeated queries
4. **Monitor resource usage** with `docker stats`

## Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check container logs
docker compose logs ollama
docker compose logs vllm

# Restart services
docker compose restart ollama vllm
```

**Out of memory errors:**
```bash
# Increase memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G  # Increase from 4G/6G
```

**Model download issues:**
```bash
# Manually pull Ollama model
docker compose exec ollama ollama pull llama3.2:1b

# Check vLLM model cache
docker compose exec vllm ls -la /root/.cache/huggingface
```

### Health Checks

Both services include health checks:

```bash
# Check health status
docker compose ps

# Manual health check
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8001/v1/models  # vLLM
```

## Integration Examples

### Query Processing Enhancement

```python
def process_enhanced_query(query_text, diversity_score, num_stances):
    # Step 1: Enhance with local LLM
    enhancement = enhance_query_with_local_llm(query_text)
    
    # Step 2: Use keywords for better search
    if enhancement.get('keywords'):
        search_terms = enhancement['keywords']
        
    # Step 3: Process with main pipeline
    result = complete(query_text, diversity_score, num_stances)
    
    # Step 4: Analyze generated arguments
    for argument in result.get('arguments', []):
        analysis = analyze_argument_with_local_llm(
            argument['text'], 
            argument['stance']
        )
        argument['local_analysis'] = analysis
    
    return result
```

### Argument Quality Assessment

```python
def assess_argument_quality(arguments):
    assessments = []
    
    for arg in arguments:
        analysis = analyze_argument_with_local_llm(
            arg['supporting_arguments'][0],
            arg['stance']
        )
        
        quality_score = {
            'sentiment': analysis.get('sentiment'),
            'key_points': len(analysis.get('key_points', [])),
            'has_summary': bool(analysis.get('brief_summary'))
        }
        
        assessments.append(quality_score)
    
    return assessments
```

## Monitoring

### Service Logs
```bash
# Monitor real-time logs
docker compose logs -f ollama vllm

# Check specific service
docker compose logs ollama --tail=50
```

### Resource Usage
```bash
# Monitor container resources
docker stats allsides_ollama allsides_vllm

# Check memory usage
docker compose exec ollama free -h
```

### API Status
```bash
# Check via API endpoint
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  http://localhost:9000/api/llm/status/
```

## Future Enhancements

1. **GPU Support**: Add NVIDIA GPU configuration for faster inference
2. **Model Management**: Dynamic model loading/unloading
3. **Load Balancing**: Multiple instances for high availability
4. **Custom Models**: Fine-tuned models for domain-specific tasks
5. **Caching Layer**: Redis-based response caching for LLM outputs

## Security Considerations

- **Network Isolation**: Services run on internal Docker network
- **No External Access**: LLM services not exposed to public internet
- **Authentication**: All API endpoints require JWT authentication
- **Resource Limits**: CPU/memory limits prevent resource exhaustion

## Dependencies

The integration requires these Python packages (already included):

```txt
openai>=1.84.0          # For vLLM OpenAI-compatible API
requests>=2.32.3        # For HTTP requests to Ollama
langchain-ollama==0.2.2 # Optional: LangChain integration
```

---

For support or questions about the local LLM integration, check the service logs or test the endpoints using the provided API documentation.