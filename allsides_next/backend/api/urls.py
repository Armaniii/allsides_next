from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from . import views

router = DefaultRouter()
router.register(r'queries', views.QueryViewSet)
router.register(r'users', views.UserViewSet)
router.register(r'ratings', views.ArgumentRatingViewSet, basename='rating')
router.register(r'thumbs-ratings', views.ThumbsRatingViewSet, basename='thumbs-rating')

urlpatterns = [
    # User specific endpoints
    path('users/leaderboard/', views.leaderboard, name='leaderboard'),
    path('users/me/', views.UserViewSet.as_view({'get': 'me'}), name='user-me'),
    path('users/stats/', views.UserViewSet.as_view({'get': 'stats'}), name='user-stats'),
    # JWT Token endpoints
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    # Registration endpoint
    path('register/', views.RegisterView.as_view(), name='register'),
    # Streaming endpoint - standalone view to avoid DRF content negotiation issues
    path('queries/stream/', views.stream_query_view, name='stream-query'),
    # Cache management endpoints
    path('cache/stats/', views.cache_statistics, name='cache_statistics'),
    path('cache/optimize/', views.optimize_cache_view, name='optimize_cache'),
    path('cache/warm/', views.warm_cache_view, name='warm_cache'),
    path('cache-stats/', views.get_cache_stats, name='cache-stats'),
    # Local LLM endpoints
    path('llm/status/', views.llm_status_view, name='llm_status'),
    path('llm/test/', views.test_llm_view, name='llm_test'),
    # Research API endpoints (disabled - open_deep_research dependency removed)
    # path('research/', include('api.research.urls')),
    # Include router URLs last to avoid conflicts
    path('', include(router.urls)),
]