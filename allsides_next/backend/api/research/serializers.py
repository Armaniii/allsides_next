from rest_framework import serializers
from .models import ResearchReport, ResearchSection

class ResearchSectionSerializer(serializers.ModelSerializer):
    """Serializer for ResearchSection model."""
    
    class Meta:
        model = ResearchSection
        fields = ['id', 'name', 'content', 'order', 'created_at', 'updated_at']


class ResearchReportSerializer(serializers.ModelSerializer):
    """Serializer for ResearchReport model."""
    
    sections = ResearchSectionSerializer(many=True, read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    username = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = ResearchReport
        fields = [
            'id', 'user', 'username', 'topic', 'thread_id', 'status', 'status_display',
            'content', 'plan', 'created_at', 'updated_at', 'completed_at',
            'sections'
        ]
        read_only_fields = [
            'thread_id', 'created_at', 'updated_at', 'completed_at'
        ]


class ResearchReportCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a new research report."""
    
    class Meta:
        model = ResearchReport
        fields = ['topic', 'config']
        
    def validate_config(self, value):
        """Validate the config field."""
        # Ensure config is a dictionary
        if not isinstance(value, dict):
            raise serializers.ValidationError("Config must be a JSON object")
        
        # Optional: Add validation for specific config fields if needed
        
        return value


class ResearchReportUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating a research report."""
    
    class Meta:
        model = ResearchReport
        fields = ['status', 'content', 'plan']


class ResearchFeedbackSerializer(serializers.Serializer):
    """Serializer for submitting feedback on a research plan."""
    
    feedback = serializers.CharField(required=True)
    
    def validate_feedback(self, value):
        """Validate feedback input."""
        if not value.strip():
            raise serializers.ValidationError("Feedback cannot be empty")
        return value 