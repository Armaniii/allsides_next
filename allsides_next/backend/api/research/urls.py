from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    ResearchReportViewSet,
    StartResearchView,
    SSEHealthCheckView
)

# Create the router and register viewsets
router = DefaultRouter()
router.register(r'reports', ResearchReportViewSet, basename='research-report')

# URL patterns
urlpatterns = [
    # Include router URLs (this will include the stream endpoint)
    path('', include(router.urls)),
    
    # Other endpoints
    path('start/', StartResearchView.as_view(), name='start-research'),
    path('health/sse/', SSEHealthCheckView.as_view(), name='sse-health-check'),
] 