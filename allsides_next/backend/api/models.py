from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.postgres.fields import JSONField
from django.utils import timezone
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class User(AbstractUser):
    """Custom user model extending Django's AbstractUser"""
    BIAS_CHOICES = [
        ('L', 'Left'),
        ('LL', 'Lean Left'),
        ('C', 'Center'),
        ('LR', 'Lean Right'),
        ('R', 'Right'),
    ]
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    daily_query_count = models.IntegerField(default=0)
    last_query_reset = models.DateTimeField(null=True, blank=True)
    bias_rating = models.CharField(max_length=2, choices=BIAS_CHOICES, null=True, blank=True)
    email = models.EmailField(unique=True)  # Make email required and unique
    allstars = models.IntegerField(default=0)  # Track user's earned stars
    daily_query_limit = models.IntegerField(default=20)
    
    def get_remaining_queries(self):
        now = timezone.now()
        
        # Calculate next midnight UTC
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        # If last_query_reset is None or if we've passed midnight since last reset
        if not self.last_query_reset or now >= next_midnight:
            try:
                self.daily_query_count = 0
                self.last_query_reset = now
                self.save()
                logger.info(f"Reset query count for user {self.username} (id: {self.id})")
            except Exception as e:
                logger.error(f"Failed to reset query count for user {self.username} (id: {self.id}): {str(e)}", exc_info=True)
                # Even if save fails, return the correct count for this request
            return self.daily_query_limit
            
        return max(self.daily_query_limit - self.daily_query_count, 0)
    
    def get_reset_time(self):
        """Get the next query reset time (midnight UTC)"""
        if not self.last_query_reset:
            return None
            
        # Calculate next midnight UTC from the last reset
        return (self.last_query_reset + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    
    def increment_query_count(self):
        """
        Increment the user's daily query count if they have remaining queries.
        Always updates last_query_reset to ensure consistent tracking.
        Returns True if successful, False if limit reached.
        """
        remaining = self.get_remaining_queries()
        if remaining > 0:
            try:
                self.daily_query_count += 1
                # Always update last_query_reset for consistent tracking
                self.last_query_reset = timezone.now()
                self.save()
                logger.info(f"Incremented query count for user {self.username} (id: {self.id}) to {self.daily_query_count}")
                return True
            except Exception as e:
                logger.error(f"Failed to increment query count for user {self.username} (id: {self.id}): {str(e)}", exc_info=True)
                return False
        return False

    def increment_allstars(self):
        """Increment user's allstars count by 1"""
        try:
            self.allstars = models.F('allstars') + 1
            self.save()
            # Refresh from db to get the actual value
            self.refresh_from_db()
            logger.info(f"Incremented allstars for user {self.username} (id: {self.id}) to {self.allstars}")
            return self.allstars
        except Exception as e:
            logger.error(f"Failed to increment allstars for user {self.username} (id: {self.id}): {str(e)}", exc_info=True)
            return self.allstars  # Return current value if increment fails

    def __str__(self):
        return self.username

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

    class Meta:
        db_table = 'auth_user'

class Query(models.Model):
    """Model to store user queries and their results"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='queries')
    query_text = models.CharField(max_length=500)
    diversity_score = models.FloatField()
    response = models.JSONField(default=dict)  # Store the GPT response
    system_prompt = models.TextField(null=True, blank=True)  # Store the system prompt used
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)  # For soft delete

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Queries'

    def __str__(self):
        return f"{self.user.username} - {self.query_text[:50]}"

class ArgumentRating(models.Model):
    """Model to store user ratings for arguments"""
    RATING_CHOICES = [
        ('L', 'Left'),
        ('LL', 'Lean Left'),
        ('C', 'Center'),
        ('LR', 'Lean Right'),
        ('R', 'Right'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='argument_ratings')
    query = models.ForeignKey(Query, on_delete=models.CASCADE, related_name='argument_ratings')
    stance = models.CharField(max_length=255)
    core_argument = models.TextField()
    rating = models.CharField(max_length=2, choices=RATING_CHOICES)
    argument_source = models.CharField(max_length=500, default='gpt-4')  # New field
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['user', 'core_argument'],
                name='unique_user_argument_rating'
            )
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.core_argument[:30]} - {self.rating}"

class ThumbsRating(models.Model):
    """Model to store thumbs up/down ratings for arguments"""
    RATING_CHOICES = [
        ('UP', 'Thumbs Up'),
        ('DOWN', 'Thumbs Down'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='thumbs_ratings')
    query = models.ForeignKey(Query, on_delete=models.CASCADE, related_name='thumbs_ratings')
    core_argument = models.TextField()
    stance = models.CharField(max_length=255)  # Add stance field
    rating = models.CharField(max_length=4, choices=RATING_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=('user', 'core_argument'),
                name='unique_user_argument_thumbs_rating'
            )
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.core_argument[:30]} - {self.rating}"

class CacheStatistics(models.Model):
    """Model to track cache usage statistics"""
    timestamp = models.DateTimeField(default=timezone.now)
    cache_hits = models.IntegerField(default=0)
    cache_misses = models.IntegerField(default=0)
    memory_usage = models.FloatField(default=0.0)  # in MB
    total_entries = models.IntegerField(default=0)
    hit_rate = models.FloatField(default=0.0)  # percentage

    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'Cache Statistics'

    def __str__(self):
        return f"Cache Stats {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - Hit Rate: {self.hit_rate:.2f}%"
