from django.contrib import admin
from django.contrib.auth import get_user_model
from django.utils import timezone
from .models import Query, ArgumentRating, ThumbsRating, CacheStatistics

User = get_user_model()

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('username', 'email', 'first_name', 'last_name', 'daily_query_count', 
                   'daily_query_limit', 'remaining_queries', 'last_query_reset', 'allstars')
    search_fields = ('username', 'email', 'first_name', 'last_name')
    list_filter = ('is_staff', 'is_superuser', 'is_active')
    readonly_fields = ('created_at', 'updated_at', 'last_login', 'date_joined')
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'email', 'bias_rating')}),
        ('Query Limits', {'fields': ('daily_query_count', 'daily_query_limit', 'last_query_reset')}),
        ('Rewards', {'fields': ('allstars',)}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined', 'created_at', 'updated_at')}),
    )
    
    def remaining_queries(self, obj):
        """Display remaining queries in the admin list view"""
        return obj.get_remaining_queries()
    remaining_queries.short_description = 'Remaining Queries'
    
    def save_model(self, request, obj, form, change):
        """Custom save method to handle query limit changes"""
        # Check if daily_query_limit was changed
        if change and 'daily_query_limit' in form.changed_data:
            old_obj = self.model.objects.get(pk=obj.pk)
            self.message_user(
                request, 
                f"Query limit changed from {old_obj.daily_query_limit} to {obj.daily_query_limit} for user {obj.username}"
            )
        
        # Check if daily_query_count was manually reset
        if change and 'daily_query_count' in form.changed_data:
            old_obj = self.model.objects.get(pk=obj.pk)
            if obj.daily_query_count == 0 and old_obj.daily_query_count > 0:
                obj.last_query_reset = timezone.now()
                self.message_user(
                    request, 
                    f"Query count reset from {old_obj.daily_query_count} to 0 for user {obj.username}"
                )
        
        super().save_model(request, obj, form, change)

@admin.register(Query)
class QueryAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'query_text_short', 'diversity_score', 'created_at', 'is_active')
    list_filter = ('is_active', 'created_at')
    search_fields = ('query_text', 'user__username', 'user__email')
    readonly_fields = ('created_at', 'updated_at')
    
    def query_text_short(self, obj):
        """Truncate query text for display"""
        return obj.query_text[:50] + ('...' if len(obj.query_text) > 50 else '')
    query_text_short.short_description = 'Query Text'

@admin.register(ArgumentRating)
class ArgumentRatingAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'query', 'stance', 'rating', 'created_at')
    list_filter = ('rating', 'created_at')
    search_fields = ('user__username', 'stance', 'core_argument')

@admin.register(ThumbsRating)
class ThumbsRatingAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'query', 'stance', 'rating', 'created_at')
    list_filter = ('rating', 'created_at')
    search_fields = ('user__username', 'stance', 'core_argument')

@admin.register(CacheStatistics)
class CacheStatisticsAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'cache_hits', 'cache_misses', 'hit_rate', 'memory_usage', 'total_entries')
    list_filter = ('timestamp',)
    readonly_fields = ('timestamp',)
