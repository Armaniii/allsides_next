# Running Production-Like Tests for LLM Helpers

## Method 1: Test Inside Docker Container (Recommended)

This is exactly how it runs in production:

```bash
# 1. Make sure Docker services are running
docker compose up -d

# 2. Wait for services to be ready (check logs)
docker compose logs ollama
docker compose logs vllm

# 3. Execute test inside the backend container
docker compose exec backend python test_llm_helpers_production_like.py
```

## Method 2: Test from Host with Port Forwarding

If you want to test from your host machine:

```bash
# 1. Start services with exposed ports
docker compose up -d

# 2. Update your /etc/hosts file (temporary)
echo "127.0.0.1 ollama" | sudo tee -a /etc/hosts
echo "127.0.0.1 vllm" | sudo tee -a /etc/hosts

# 3. Ensure ports are exposed in docker-compose.yml:
# ollama:
#   ports:
#     - "11434:11434"
# vllm:
#   ports:
#     - "8000:8000"

# 4. Run the test
python3 test_llm_helpers_production_like.py
```

## Method 3: Use Docker Network from Host

```bash
# Run the test script in the same network as services
docker run --rm \
  --network allsides_next_project_default \
  -v $(pwd):/app \
  -w /app \
  python:3.9 \
  python test_llm_helpers_production_like.py
```

## Why Our Mock Test Was Sufficient

For testing the **logic** of the LLM helpers, our mock test was actually better because:

1. **Deterministic**: Same results every time
2. **Fast**: No waiting for LLM responses  
3. **No Dependencies**: Doesn't require Docker services
4. **Edge Case Testing**: Can test specific scenarios easily

## Production Monitoring

In production, you should monitor:

1. **LLM Service Health**:
   ```python
   # The llm_client already logs service availability
   # Check logs for: "❌ Ollama service unavailable"
   ```

2. **Response Times**:
   ```python
   # Add timing logs in production
   import time
   start = time.time()
   enhanced = await enhance_response_with_llm(...)
   logger.info(f"LLM enhancement took {time.time() - start:.2f}s")
   ```

3. **Fallback Behavior**:
   - If both LLM services fail, the helpers return empty/default values
   - The main response still works, just without enhancements