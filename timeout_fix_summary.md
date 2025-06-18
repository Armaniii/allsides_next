# API Timeout Fix Summary

## Issue
The frontend was experiencing "timeout of 60000ms exceeded" errors when making API calls to the backend, despite the backend being configured with a 5-minute timeout.

## Root Cause
The timeout mismatch occurred at multiple levels:
1. **Backend OpenAI client**: Configured with 300s (5 minutes) timeout ✓
2. **Nginx proxy**: Configured with 300s (5 minutes) timeout ✓
3. **Frontend axios instance**: Configured with 300s (5 minutes) default timeout ✓
4. **Frontend API call**: Was NOT explicitly passing the timeout, potentially using a default 60s timeout

## Solution Applied
Modified the `queries.create` function in `/allsides_next/frontend/src/lib/api.ts` to explicitly pass a 5-minute timeout:

```typescript
const response = await api.post('/queries/', data, {
  timeout: 300000 // 5 minutes timeout to match backend
});
```

## Configuration Summary
- **Backend (main_v3.py)**: OpenAI client timeout = 300s
- **Nginx**: proxy_connect_timeout, proxy_send_timeout, proxy_read_timeout = 300s
- **Frontend axios default**: timeout = 300000ms (5 minutes)
- **Frontend API call**: Now explicitly sets timeout = 300000ms (5 minutes)

## Testing
After applying this fix, the frontend should now wait up to 5 minutes for API responses, matching the backend's OpenAI processing timeout.