from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from .models import Query, ArgumentRating, ThumbsRating
import json
import logging
import re

User = get_user_model()
logger = logging.getLogger(__name__)

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'bias_rating', 'daily_query_count', 'last_query_reset', 'allstars']
        read_only_fields = ['id', 'daily_query_count', 'last_query_reset', 'allstars']

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = ['username', 'password', 'password2', 'email', 'first_name', 'last_name', 'bias_rating']

    def validate_email(self, value):
        """
        Check that the email is from allsides.com domain and properly formatted.
        """
        try:
            # Normalize email and handle None/empty values
            if not value:
                raise serializers.ValidationError("Email address is required.")
            
            value = value.lower().strip()
            
            # Basic email format validation
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, value):
                raise serializers.ValidationError("Please enter a valid email address.")
            
            if not value.endswith('@allsides.com'):
                raise serializers.ValidationError(
                    "Registration is currently limited to @allsides.com email addresses only."
                )
            
            # Check if email already exists
            if User.objects.filter(email=value).exists():
                raise serializers.ValidationError(
                    "This email address is already registered."
                )
                
            return value
        except serializers.ValidationError:
            raise
        except Exception as e:
            logger.error(f"Email validation error: {str(e)}", exc_info=True)
            raise serializers.ValidationError("Invalid email format.")

    def validate_bias_rating(self, value):
        """
        Validate and normalize the bias rating.
        """
        try:
            if value is None:
                return value
                
            # Normalize to uppercase and handle potential non-string inputs
            value = str(value).upper().strip()
            
            valid_ratings = dict(User.BIAS_CHOICES)
            if value not in valid_ratings:
                raise serializers.ValidationError(
                    f"Invalid bias rating. Must be one of: {', '.join(valid_ratings.keys())}"
                )
            return value
        except serializers.ValidationError:
            raise
        except Exception as e:
            logger.error(f"Bias rating validation error: {str(e)}", exc_info=True)
            raise serializers.ValidationError("Invalid bias rating format.")

    def validate(self, attrs):
        """
        Check that the passwords match and perform other validations.
        """
        try:
            # Normalize text fields and handle potential None values
            attrs['first_name'] = attrs.get('first_name', '').strip()
            attrs['last_name'] = attrs.get('last_name', '').strip()
            attrs['username'] = attrs.get('username', '').strip()

            # Check required fields
            required_fields = ['username', 'email', 'first_name', 'last_name', 'password', 'password2']
            missing_fields = [field for field in required_fields if not attrs.get(field, '').strip()]
            if missing_fields:
                raise serializers.ValidationError({
                    "missing_fields": f"The following fields are required: {', '.join(missing_fields)}"
                })
            
            # Validate username format
            username = attrs['username']
            if not username.isalnum():
                raise serializers.ValidationError({
                    "username": "Username can only contain letters and numbers (no spaces or special characters)"
                })
            
            # Check username length
            if len(username) < 3 or len(username) > 30:
                raise serializers.ValidationError({
                    "username": "Username must be between 3 and 30 characters long"
                })
                
            # Check if username exists
            if User.objects.filter(username=username).exists():
                raise serializers.ValidationError({
                    "username": "This username is already taken"
                })
            
            # Check password match
            if attrs['password'] != attrs['password2']:
                raise serializers.ValidationError({
                    "password": "The two password fields didn't match."
                })
            
            try:
                # Additional password validations
                validate_password(attrs['password'])
            except Exception as e:
                raise serializers.ValidationError({
                    "password": list(e.messages)
                })
            
            # Validate name fields
            if not attrs['first_name'].replace(' ', '').isalpha():
                raise serializers.ValidationError({
                    "first_name": "First name should only contain letters"
                })
                
            if not attrs['last_name'].replace(' ', '').isalpha():
                raise serializers.ValidationError({
                    "last_name": "Last name should only contain letters"
                })
            
            return attrs
            
        except serializers.ValidationError:
            raise
        except Exception as e:
            logger.error(f"Registration validation error: {str(e)}", exc_info=True)
            raise serializers.ValidationError({
                "non_field_errors": ["An error occurred during registration. Please try again."]
            })

    def create(self, validated_data):
        try:
            # Remove password2 as we don't need it anymore
            validated_data.pop('password2', None)
            
            # Create the user
            user = User.objects.create_user(**validated_data)
            
            return user
        except Exception as e:
            logger.error(f"User creation error: {str(e)}", exc_info=True)
            raise serializers.ValidationError({
                "non_field_errors": ["Failed to create user. Please try again."]
            })

class QuerySerializer(serializers.ModelSerializer):
    class Meta:
        model = Query
        fields = '__all__'
        read_only_fields = ['user', 'created_at', 'updated_at']

    def create(self, validated_data):
        # Ensure the user is set to the current user
        validated_data['user'] = self.context['request'].user
        return super().create(validated_data)

class ArgumentRatingSerializer(serializers.ModelSerializer):
    user = serializers.PrimaryKeyRelatedField(read_only=True)
    query = serializers.PrimaryKeyRelatedField(queryset=Query.objects.all())
    
    class Meta:
        model = ArgumentRating
        fields = ['id', 'user', 'query', 'stance', 'core_argument', 'rating', 'created_at', 'argument_source']
        read_only_fields = ['id', 'user', 'created_at', 'argument_source']

    def validate_rating(self, value):
        """
        Validate the rating value.
        """
        if not value:
            raise serializers.ValidationError("Rating cannot be empty")
        
        valid_ratings = dict(ArgumentRating.RATING_CHOICES)
        if value not in valid_ratings:
            raise serializers.ValidationError(
                f"Invalid rating. Must be one of: {', '.join(valid_ratings.keys())}"
            )
        return value

    def validate_query(self, value):
        """
        Validate that the query belongs to the current user.
        """
        user = self.context['request'].user
        if value.user != user:
            raise serializers.ValidationError("Cannot rate arguments from another user's query")
        return value

    def normalize_text(self, text):
        if not text:
            return ""
        # Remove URLs and normalize whitespace
        text_without_urls = re.sub(r'https?://[^\s]+', '', text)
        normalized = ' '.join(text_without_urls.split()).strip()
        print(f"Normalized text from '{text}' to '{normalized}'")
        return normalized

    def validate(self, attrs):
        print(f"Validating rating data: {attrs}")
        
        # Validate rating choices
        if attrs.get('rating') not in dict(ArgumentRating.RATING_CHOICES):
            raise serializers.ValidationError({"rating": "Invalid rating choice"})

        # Get the current user from the context
        user = self.context['request'].user

        # Check for existing rating with same user and core_argument
        existing_rating = ArgumentRating.objects.filter(
            user=user,
            core_argument=attrs.get('core_argument')
        ).first()

        if existing_rating:
            raise serializers.ValidationError({
                "core_argument": "You have already rated this argument"
            })

        return attrs

    def create(self, validated_data):
        print(f"Creating rating with data: {validated_data}")
        try:
            # Set the user from the context
            validated_data['user'] = self.context['request'].user
            rating = ArgumentRating.objects.create(**validated_data)
            print(f"Successfully created rating: {rating}")
            return rating
        except Exception as e:
            print(f"Error creating rating: {str(e)}")
            raise

    def update(self, instance, validated_data):
        # Normalize text for updates as well
        validated_data['core_argument'] = self.normalize_text(validated_data.get('core_argument', instance.core_argument))
        return super().update(instance, validated_data)

class ThumbsRatingSerializer(serializers.ModelSerializer):
    user = serializers.PrimaryKeyRelatedField(read_only=True)
    query = serializers.PrimaryKeyRelatedField(queryset=Query.objects.all())
    
    class Meta:
        model = ThumbsRating
        fields = ['id', 'user', 'query', 'core_argument', 'stance', 'rating', 'created_at']
        read_only_fields = ['id', 'user', 'created_at']

    def validate_rating(self, value):
        """
        Validate the rating value.
        """
        if not value:
            raise serializers.ValidationError("Rating cannot be empty")
        
        valid_ratings = dict(ThumbsRating.RATING_CHOICES)
        if value not in valid_ratings:
            raise serializers.ValidationError(
                f"Invalid rating. Must be one of: {', '.join(valid_ratings.keys())}"
            )
        return value

    def validate_query(self, value):
        """
        Validate that the query belongs to the current user.
        """
        user = self.context['request'].user
        if value.user != user:
            raise serializers.ValidationError("Cannot rate arguments from another user's query")
        return value

    def validate(self, attrs):
        # Get the current user from the context
        user = self.context['request'].user

        # Check for existing rating with same user and core_argument
        existing_rating = ThumbsRating.objects.filter(
            user=user,
            core_argument=attrs.get('core_argument')
        ).first()

        if existing_rating:
            raise serializers.ValidationError({
                "core_argument": "You have already rated this argument"
            })

        return attrs

    def create(self, validated_data):
        # Set the user from the context
        validated_data['user'] = self.context['request'].user
        rating = ThumbsRating.objects.create(**validated_data)
        return rating 