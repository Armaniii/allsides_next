# AllSides Next - API Documentation

## Overview

This module implements the core API functionality for the AllSides Next application. It provides endpoints for query submission, argument retrieval, user authentication, and various user interactions with the generated content.

## API Endpoints

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/login/` | POST | Authenticate user and issue JWT tokens |
| `/api/auth/register/` | POST | Register a new user account |
| `/api/auth/refresh/` | POST | Refresh an expired access token |
| `/api/auth/logout/` | POST | Invalidate user tokens |

### Queries

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/queries/` | GET | List all queries for the authenticated user |
| `/api/queries/` | POST | Submit a new query to generate diverse perspectives |
| `/api/queries/{id}/` | GET | Retrieve a specific query and its response |
| `/api/queries/{id}/` | DELETE | Delete a specific query |
| `/api/queries/stream/` | GET | Stream arguments as they are generated (SSE) |

### Ratings

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ratings/argument/` | POST | Submit a political bias rating for an argument |
| `/api/ratings/thumbs/` | POST | Submit a thumbs up/down rating for an argument |
| `/api/ratings/stats/{query_id}/` | GET | Get rating statistics for a query |

### User Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats/user/` | GET | Get current user's statistics |
| `/api/leaderboard/` | GET | Get top users by AllStars points |

## Request/Response Examples

### Query Submission

**Request:**
```json
POST /api/queries/
{
  "query_text": "Should the government regulate social media?",
  "diversity_score": 0.7,
  "num_stances": 5
}
```

**Response:**
```json
{
  "id": 123,
  "query_text": "Should the government regulate social media?",
  "diversity_score": 0.7,
  "response": {
    "arguments": [
      {
        "stance": "Left",
        "core_argument": "Social media companies have become too powerful and need regulation to protect user data and prevent misinformation.",
        "supporting_arguments": [
          "Facebook's Cambridge Analytica scandal showed the dangers of unregulated data harvesting.",
          "Unregulated platforms have allowed foreign entities to interfere in elections.",
          "Without regulation, companies prioritize engagement metrics over user safety."
        ]
      },
      {
        "stance": "Right",
        "core_argument": "Government regulation of social media threatens free speech and would give too much power to bureaucrats.",
        "supporting_arguments": [
          "Regulation would inevitably favor certain political viewpoints over others.",
          "Private companies should set their own content moderation policies.",
          "Innovation thrives with less regulation, not more."
        ]
      }
      // Additional stances...
    ]
  },
  "created_at": "2023-06-15T12:34:56Z",
  "is_active": true
}
```

## Data Models

### User
Extended Django user model with additional fields:
- `bias_rating`: User's self-reported political bias
- `daily_query_count`: Number of queries made today
- `daily_query_limit`: Maximum queries allowed per day
- `allstars`: Points earned through platform engagement

### Query
Stores user queries and their responses:
- `user`: Foreign key to User model
- `query_text`: The text of the submitted query
- `diversity_score`: Score indicating how diverse the perspectives should be
- `response`: JSON field containing the generated arguments
- `created_at`: Timestamp of query submission

### ArgumentRating
Tracks user ratings for arguments on the political spectrum:
- `user`: User who submitted the rating
- `query`: Associated query
- `stance`: The political stance of the rated argument
- `core_argument`: The argument text being rated
- `rating`: Political bias rating (L, LL, C, LR, R)

### ThumbsRating
Captures thumbs up/down feedback for arguments:
- `user`: User who submitted the rating
- `query`: Associated query
- `core_argument`: The argument text being rated
- `stance`: The political stance of the rated argument
- `rating`: Either "UP" or "DOWN"

## Implementation Details

### Argument Generation

The query processing pipeline:
1. User submits a query through the API
2. System validates and preprocesses the query
3. Diversity score determines how widely varied the perspectives should be
4. Request is sent to AI service to generate diverse perspectives
5. Results are normalized and stored in the database
6. Response is returned to the user

### Rate Limiting

The API implements rate limiting to:
- Prevent abuse of the service
- Ensure fair resource allocation
- Track user engagement metrics
- Reset daily at UTC midnight

### Caching

The application implements caching to:
- Reduce duplicate processing of similar queries
- Speed up common requests
- Minimize AI service usage

## Error Handling

All API endpoints use consistent error handling:
- HTTP 400: Bad Request (invalid input)
- HTTP 401: Unauthorized (authentication required)
- HTTP 403: Forbidden (insufficient permissions)
- HTTP 404: Not Found (resource doesn't exist)
- HTTP 429: Too Many Requests (rate limit exceeded)
- HTTP 500: Internal Server Error (unexpected error)

Each error response includes:
- `detail`: Human-readable error message
- `code`: Machine-readable error code
- `errors`: Detailed validation errors (when applicable) 