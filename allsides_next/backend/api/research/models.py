from django.db import models
from django.utils import timezone
from django.contrib.auth import get_user_model
from ..models import Query

User = get_user_model()

class ResearchReport(models.Model):
    """Model to store deep research reports."""
    
    STATUS_CHOICES = [
        ('PLANNING', 'Planning'),
        ('RESEARCHING', 'Researching'),
        ('WRITING', 'Writing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='research_reports')
    query = models.ForeignKey(Query, on_delete=models.CASCADE, related_name='research_reports', null=True, blank=True)
    topic = models.CharField(max_length=500)
    thread_id = models.CharField(max_length=100, unique=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PLANNING')
    content = models.TextField(blank=True)
    plan = models.JSONField(default=dict)
    config = models.JSONField(default=dict)
    thread_state = models.JSONField(default=dict, blank=True)
    last_event = models.JSONField(default=dict, blank=True)
    last_event_time = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Research Reports'
        indexes = [
            models.Index(fields=['thread_id']),
            models.Index(fields=['status']),
            models.Index(fields=['user', 'status']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.topic[:50]}"
    
    def update_state(self, event_data: dict):
        """Update report state based on an event."""
        if not event_data:
            return
            
        event_type = event_data.get('type')
        
        if event_type == 'report_status':
            # Update status and save immediately
            new_status = event_data.get('status')
            if new_status and new_status != self.status:
                self.status = new_status
                # If transitioning to RESEARCHING, ensure we update immediately
                if new_status == 'RESEARCHING':
                    self.save(update_fields=['status'])
            
            # Update plan if provided
            if 'plan' in event_data:
                self.plan = event_data['plan']
            
            # Update content if provided
            if 'content' in event_data:
                self.content = event_data['content']
                
        # Store the event
        self.last_event = event_data
        self.last_event_time = timezone.now()
        
        # If completed, set completed_at
        if self.status == 'COMPLETED' and not self.completed_at:
            self.completed_at = timezone.now()
            
        self.save()

class ResearchSection(models.Model):
    """Model to store individual sections of a research report."""
    
    report = models.ForeignKey(ResearchReport, on_delete=models.CASCADE, related_name='sections')
    name = models.CharField(max_length=255)
    content = models.TextField(blank=True)
    order = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['order', 'created_at']
        verbose_name_plural = 'Research Sections'
    
    def __str__(self):
        return f"{self.report.topic[:30]} - {self.name}" 