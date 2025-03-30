import os
import sys
import django
from datetime import datetime

# Add the backend directory to Python path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_path)

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from api.models import Query, User

def list_recent_queries(limit=10):
    """List the most recent queries."""
    queries = Query.objects.all().order_by('-created_at')[:limit]
    for q in queries:
        print(f"ID: {q.id}")
        print(f"Query: {q.query_text}")
        print(f"Created: {q.created_at}")
        print(f"Diversity Score: {q.diversity_score}")
        print("-" * 50)

def get_query_by_id(query_id):
    """Get a specific query and its results."""
    try:
        query = Query.objects.get(id=query_id)
        print(f"Query: {query.query_text}")
        print(f"Results:")
        for stance in query.results:
            print(f"\nStance: {list(stance.keys())[0]}")
            arguments = list(stance.values())[0]
            for core_arg, supporting_args in arguments.items():
                print(f"Core Argument: {core_arg}")
                print("Supporting Arguments:")
                for arg in supporting_args:
                    print(f"  • {arg}")
    except Query.DoesNotExist:
        print(f"Query with ID {query_id} not found")

def search_queries(search_term):
    """Search queries by text."""
    queries = Query.objects.filter(query_text__icontains=search_term)
    print(f"Found {queries.count()} queries containing '{search_term}':")
    for q in queries:
        print(f"ID: {q.id}")
        print(f"Query: {q.query_text}")
        print("-" * 50)

def get_user_stats(username):
    """Get statistics for a specific user."""
    try:
        user = User.objects.get(username=username)
        queries = Query.objects.filter(user=user)
        print(f"Stats for user: {username}")
        print(f"Total queries: {queries.count()}")
        print(f"Average diversity score: {queries.aggregate(avg_score=django.db.models.Avg('diversity_score'))['avg_score']:.2f}")
        print("\nMost recent queries:")
        for q in queries.order_by('-created_at')[:5]:
            print(f"- {q.query_text} ({q.created_at.strftime('%Y-%m-%d %H:%M')})")
    except User.DoesNotExist:
        print(f"User {username} not found")

def analyze_query_trends():
    """Analyze trends in queries."""
    total_queries = Query.objects.count()
    today = datetime.now()
    queries_today = Query.objects.filter(created_at__date=today.date()).count()
    avg_diversity = Query.objects.aggregate(avg_score=django.db.models.Avg('diversity_score'))['avg_score']
    
    print("Query Statistics:")
    print(f"Total queries: {total_queries}")
    print(f"Queries today: {queries_today}")
    print(f"Average diversity score: {avg_diversity:.2f}")

if __name__ == '__main__':
    # Example usage
    print("\n=== Recent Queries ===")
    list_recent_queries(5)
    
    print("\n=== Search Example ===")
    search_queries("abortion")
    
    print("\n=== Query Trends ===")
    analyze_query_trends() 

    print("\n=== User Stats ===")
    get_user_stats("admin")
