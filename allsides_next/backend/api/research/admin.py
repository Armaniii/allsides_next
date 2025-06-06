from django.contrib import admin
from .models import ResearchReport, ResearchSection

class ResearchSectionInline(admin.TabularInline):
    """Inline admin for ResearchSection model."""
    model = ResearchSection
    extra = 0
    readonly_fields = ['created_at', 'updated_at']
    fields = ['name', 'content', 'order', 'created_at', 'updated_at']

@admin.register(ResearchReport)
class ResearchReportAdmin(admin.ModelAdmin):
    """Admin for ResearchReport model."""
    list_display = ['id', 'user', 'topic', 'status', 'created_at', 'completed_at']
    list_filter = ['status', 'created_at', 'completed_at']
    search_fields = ['topic', 'user__username', 'user__email', 'thread_id']
    readonly_fields = ['created_at', 'updated_at', 'completed_at', 'thread_id']
    fieldsets = [
        (None, {'fields': ['user', 'query', 'topic', 'thread_id']}),
        ('Status', {'fields': ['status', 'created_at', 'updated_at', 'completed_at']}),
        ('Content', {'fields': ['content']}),
        ('Configuration', {'fields': ['plan', 'config']}),
    ]
    inlines = [ResearchSectionInline]
    save_on_top = True
    
    def get_readonly_fields(self, request, obj=None):
        """Make thread_id readonly only if it already exists."""
        if obj:  # editing an existing object
            return self.readonly_fields
        return [f for f in self.readonly_fields if f != 'thread_id']

@admin.register(ResearchSection)
class ResearchSectionAdmin(admin.ModelAdmin):
    """Admin for ResearchSection model."""
    list_display = ['id', 'report_link', 'name', 'order', 'created_at']
    list_filter = ['created_at', 'updated_at']
    search_fields = ['name', 'content', 'report__topic', 'report__user__username']
    readonly_fields = ['created_at', 'updated_at']
    
    def report_link(self, obj):
        """Link to the parent report."""
        if obj.report:
            report = obj.report
            return f"{report.topic[:30]} ({report.user.username})"
        return "-"
    report_link.short_description = "Report" 