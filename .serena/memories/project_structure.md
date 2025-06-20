# Project Directory Structure

```
allsides_next_project/
├── allsides_next/
│   ├── frontend/          # Next.js React application
│   │   ├── src/
│   │   │   ├── app/       # Next.js App Router pages
│   │   │   ├── components/# Reusable UI components
│   │   │   ├── contexts/  # React Context providers
│   │   │   ├── lib/       # Utilities and type definitions
│   │   │   ├── services/  # API client and services
│   │   │   └── utils/     # Helper utilities
│   │   ├── public/        # Static assets
│   │   └── package.json   # Frontend dependencies
│   │
│   └── backend/           # Django REST API
│       ├── api/           # Main API app
│       │   ├── models.py  # Database models
│       │   ├── views.py   # API endpoints
│       │   ├── main_v3.py # Core query processing logic
│       │   └── llm_helpers.py # Local LLM enhancement features
│       ├── core/          # Django project settings
│       ├── static/        # Static files
│       ├── staticfiles/   # Collected static files
│       ├── mediafiles/    # User uploaded files
│       └── manage.py      # Django management script
│
├── nginx/                 # Nginx configuration
├── certbot/              # SSL certificate management
├── backups/              # Database backups directory
├── docker-compose.yml    # Main Docker configuration
├── docker-compose.aurora.yml # Aurora migration config
├── .env                  # Environment variables (gitignored)
├── README.md             # Project documentation
├── CLAUDE.md            # Claude AI instructions
├── TODO.md              # Active task tracking
└── .gitignore           # Git ignore patterns
```

## Key Files
- **Backend Core Logic**: `allsides_next/backend/api/main_v3.py`
- **LLM Enhancements**: `allsides_next/backend/api/llm_helpers.py`
- **Frontend Entry**: `allsides_next/frontend/src/app/page.tsx`
- **API Client**: `allsides_next/frontend/src/services/api.ts`
- **Docker Services**: `docker-compose.yml`

## Service Ports
- Frontend: 3000 (internal), 9000 (external via Nginx)
- Backend: 8000 (internal only)
- PostgreSQL: 5432 (internal only)
- Redis: 6379
- pgAdmin: 5050
- Ollama: 11434
- vLLM: 8001