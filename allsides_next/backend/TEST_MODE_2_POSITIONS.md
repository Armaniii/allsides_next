# Test Mode: 2-Position Limitation

## Overview
The Junto Evidence Pipeline has been temporarily modified to process only the **first 2 positions** instead of all generated positions. This is for **faster testing** during development.

## Implementation Details

### Code Location
File: `/home/arman/allsides_next_project/allsides_next/backend/api/main_v3.py`
Lines: 1397-1406

### Changes Made
```python
# TEMPORARY: Limit to first 2 positions for faster testing
# TODO: Remove this limitation after testing is complete
original_position_count = len(generated_positions)
test_positions = generated_positions[:2]  # Only take first 2 positions
logger.warning(f"🧪 TESTING MODE: Limiting evidence search to first 2 positions out of {original_position_count} total positions")

# Notify user of testing mode
if progress_callback:
    progress_callback(f"🧪 TESTING MODE: Processing only 2 positions for faster testing")
```

### Visual Indicators
1. **Progress Message**: Users see "🧪 TESTING MODE: Processing only 2 positions for faster testing"
2. **Logs**: Backend logs show warning with 🧪 emoji indicating test mode
3. **Metadata**: `evidence_structure_version` includes "_TEST_LIMITED" suffix

### Expected Behavior
- Position generation: Still generates all positions (e.g., 6-10)
- Evidence search: Only searches evidence for first 2 positions
- Final output: Only 2 arguments returned instead of full set
- Processing time: ~5-10 seconds instead of 30-60 seconds

## Removing the Limitation

To restore full position processing, remove or comment out these sections:

1. **Position limiting logic** (lines 1397-1406)
2. **Replace `test_positions` with `generated_positions`** in:
   - `parallel_evidence_search()` call
   - Fallback loops for empty results
3. **Update pipeline metadata** to remove "_TEST_LIMITED" suffix

## Why This Helps Testing
1. **Faster iteration**: 5-10 second responses vs 30-60 seconds
2. **Reduced API costs**: Only 2 evidence searches instead of 6-10
3. **Easier debugging**: Less data to trace through
4. **Same code paths**: All async logic still executes normally

## Important Notes
- This is **TEMPORARY** for testing only
- Production deployment should remove this limitation
- All other async improvements remain active
- The limitation is clearly marked with TODO comments