"""
URL configuration for core project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from api.views import UserViewSet, QueryViewSet
from .admin import PortPreservingAdminSite
from django.http import HttpResponseRedirect
from urllib.parse import urlparse, urlunparse
import logging
from django.shortcuts import redirect

logger = logging.getLogger(__name__)

# Use the default admin site
admin_site = admin.site

# Custom view for handling admin redirects with port preservation
def admin_redirect_view(request, path=''):
    """
    Custom view handler for admin redirects to ensure port preservation.
    This handles cases where Django generates redirects that lose the port.
    """
    # Log the request details
    logger.debug(f"admin_redirect_view called for path: {path}")
    
    # Get the server port from various sources
    server_port = request.META.get('HTTP_X_FORWARDED_PORT', 
                  request.META.get('SERVER_PORT', 
                  getattr(settings, 'DEFAULT_HTTP_PORT', '9000')))
    
    # Get the scheme from request
    scheme = request.META.get('HTTP_X_FORWARDED_PROTO', 'http')
    
    # Get the host without port
    host = request.get_host().split(':')[0]
    
    # Construct the target URL with port
    target_url = f"{scheme}://{host}:{server_port}/admin/{path}"
    if not path.endswith('/') and path:
        target_url += '/'
    
    logger.debug(f"Redirecting to: {target_url}")
    
    # Return a redirect response
    return HttpResponseRedirect(target_url)

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'queries', QueryViewSet)

urlpatterns = [
    # Standard admin URL pattern
    path('admin/', admin.site.urls),
    
    # Redirect admin root without trailing slash
    path('admin', admin_redirect_view, {'path': ''}),
    
    # API URLs
    path('api/', include('api.urls')),  # Include the API urls directly
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]

# Add static and media URL patterns in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Custom 404 handler to preserve port in redirects
handler404 = 'core.views.custom_page_not_found'
