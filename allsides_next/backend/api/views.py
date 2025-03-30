from django.shortcuts import render
from rest_framework import viewsets, status, generics
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny, IsAdminUser
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.contrib.auth import get_user_model
# from django.http import StreamingHttpResponse
import json
from .models import Query, ArgumentRating, CacheStatistics, ThumbsRating
from .serializers import UserSerializer, QuerySerializer, RegisterSerializer, ArgumentRatingSerializer, ThumbsRatingSerializer
from . import main_v3 as gpt_service
import logging
# from .renderers import EventStreamRenderer
from django.db.models import F
from django.utils import timezone
from datetime import datetime, timedelta
from .main_v3 import (
    complete, 
    get_cache_stats, 
    get_cache_health, 
    # clear_cache,    
    optimize_cache,
    warm_cache,
    normalize_query,
    generate_cache_key,
    get_cached_response,
    set_cached_response
)
import asyncio
from functools import wraps
from django.utils.decorators import sync_and_async_middleware
# from asgiref.sync import async_to_sync
# import nest_asyncio
# nest_asyncio.apply()  # Allow nested event loops

User = get_user_model()
logger = logging.getLogger(__name__)

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (AllowAny,)
    serializer_class = RegisterSerializer

    def get_cors_headers(self):
        """Get standard CORS headers for responses"""
        return {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
            'Access-Control-Allow-Credentials': 'true',
        }

    def options(self, request, *args, **kwargs):
        """Handle preflight CORS requests"""
        headers = self.get_cors_headers()
        return Response(
            status=status.HTTP_200_OK,
            headers=headers
        )

    def create(self, request, *args, **kwargs):
        try:
            # Log the incoming request data for debugging (excluding sensitive fields)
            safe_data = {k: v for k, v in request.data.items() if k not in ['password', 'password2']}
            logger.info(f"Registration attempt from {request.META.get('REMOTE_ADDR')} with data: {json.dumps(safe_data)}")
            logger.info(f"Request headers: {dict(request.headers)}")
            
            # Check content type
            content_type = request.content_type or ''
            if not content_type.startswith('application/json'):
                logger.warning(f"Invalid content type: {content_type}")
                return Response(
                    {
                        "error": "Invalid content type",
                        "details": "Request must be application/json",
                        "help": "Please set the Content-Type header to application/json"
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                    headers=self.get_cors_headers()
                )

            # Validate request data format
            if not isinstance(request.data, dict):
                logger.warning(f"Invalid request data format: {type(request.data)}")
                return Response(
                    {
                        "error": "Invalid request format",
                        "details": "Request body must be a JSON object",
                        "help": "Please send the registration data as a JSON object"
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                    headers=self.get_cors_headers()
                )

            serializer = self.get_serializer(data=request.data)
            if not serializer.is_valid():
                # Log validation errors
                logger.warning(f"Registration validation errors: {serializer.errors}")
                return Response(
                    {
                        "error": "Invalid registration data",
                        "details": serializer.errors,
                        "help": "Please check all required fields and their formats"
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                    headers=self.get_cors_headers()
                )

            # Create the user
            try:
                self.perform_create(serializer)
                logger.info(f"Successfully created user: {serializer.data.get('username')}")
            except Exception as e:
                logger.error(f"Failed to create user: {str(e)}", exc_info=True)
                return Response(
                    {
                        "error": "User creation failed",
                        "details": "An error occurred while creating the user",
                        "help": "Please try again or contact support if the issue persists"
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    headers=self.get_cors_headers()
                )

            # Return success response
            headers = self.get_success_headers(serializer.data)
            headers.update(self.get_cors_headers())
            
            return Response(
                {
                    "message": "User registered successfully",
                    "user": serializer.data
                },
                status=status.HTTP_201_CREATED,
                headers=headers
            )

        except Exception as e:
            logger.error(f"Unexpected registration error: {str(e)}", exc_info=True)
            return Response(
                {
                    "error": "Registration failed",
                    "details": "An unexpected error occurred",
                    "help": "Please try again or contact support if the issue persists"
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                headers=self.get_cors_headers()
            )

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get_permissions(self):
        if self.action == 'create':
            return [AllowAny()]
        elif self.action in ['reset_user_stats', 'update_query_limit']:
            return [IsAdminUser()]
        return super().get_permissions()

    def get_queryset(self):
        # For security, only allow access to own user data
        return User.objects.filter(id=self.request.user.id)

    @action(detail=False, methods=['GET'])
    def me(self, request):
        """Get the current user's data."""
        try:
            serializer = self.get_serializer(request.user)
            return Response(serializer.data)
        except Exception as e:
            return Response(
                {"error": "Failed to fetch user data"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['GET'])
    def stats(self, request):
        """Get user stats including remaining queries and reset time."""
        try:
            user = request.user
            
            # Get total queries
            total_queries = Query.objects.filter(user=user).count()
            
            # Get remaining queries and reset time using model methods
            remaining_queries = user.get_remaining_queries()
            reset_time = user.get_reset_time()
            reset_time_iso = reset_time.isoformat() if reset_time else None

            return Response({
                'remaining_queries': remaining_queries,
                'reset_time': reset_time_iso,
                'daily_query_count': user.daily_query_count,
                'daily_query_limit': user.daily_query_limit,
                'total_queries': total_queries,
                'allstars': user.allstars
            })
        except Exception as e:
            logger.error(f"Error fetching user stats: {str(e)}", exc_info=True)
            return Response(
                {"error": "Failed to fetch user stats"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['POST'], url_path='reset-stats')
    def reset_user_stats(self, request, pk=None):
        """Admin endpoint to reset a user's stats cache."""
        try:
            user = self.get_object()
            # Force refresh user stats from database
            user.refresh_from_db()
            
            # Get fresh stats
            remaining_queries = user.get_remaining_queries()
            reset_time = user.get_reset_time()
            reset_time_iso = reset_time.isoformat() if reset_time else None
            total_queries = Query.objects.filter(user=user).count()
            
            return Response({
                'message': f'Stats cache reset for user {user.username}',
                'user_id': user.id,
                'username': user.username,
                'remaining_queries': remaining_queries,
                'daily_query_limit': user.daily_query_limit,
                'daily_query_count': user.daily_query_count,
                'reset_time': reset_time_iso,
                'total_queries': total_queries,
                'allstars': user.allstars
            })
        except Exception as e:
            logger.error(f"Error resetting user stats: {str(e)}", exc_info=True)
            return Response(
                {"error": f"Failed to reset stats: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
    @action(detail=True, methods=['POST'], url_path='update-query-limit')
    def update_query_limit(self, request, pk=None):
        """Admin endpoint to update a user's query limit."""
        try:
            user = self.get_object()
            
            # Get new limit from request
            new_limit = request.data.get('daily_query_limit')
            if new_limit is None:
                return Response(
                    {"error": "daily_query_limit is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
                
            try:
                new_limit = int(new_limit)
                if new_limit < 0:
                    raise ValueError("Limit must be a positive integer")
            except (ValueError, TypeError):
                return Response(
                    {"error": "daily_query_limit must be a positive integer"},
                    status=status.HTTP_400_BAD_REQUEST
                )
                
            # Update user limit
            user.daily_query_limit = new_limit
            user.save()
            
            # Get fresh stats
            remaining_queries = user.get_remaining_queries()
            reset_time = user.get_reset_time()
            reset_time_iso = reset_time.isoformat() if reset_time else None
            
            return Response({
                'message': f'Query limit updated for user {user.username}',
                'user_id': user.id,
                'username': user.username,
                'daily_query_limit': user.daily_query_limit,
                'remaining_queries': remaining_queries,
                'daily_query_count': user.daily_query_count,
                'reset_time': reset_time_iso
            })
        except Exception as e:
            logger.error(f"Error updating query limit: {str(e)}", exc_info=True)
            return Response(
                {"error": f"Failed to update query limit: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class QueryViewSet(viewsets.ModelViewSet):
    queryset = Query.objects.all()
    serializer_class = QuerySerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get_queryset(self):
        return Query.objects.filter(user=self.request.user, is_active=True)

    def create(self, request, *args, **kwargs):
        """Create a new query with caching support."""
        try:
            logger.info(f"Creating query with data: {json.dumps(request.data, indent=2)}")
            
            # Check query limit
            user = request.user
            logger.info(f"User {user.username} (id: {user.id}) has used {user.daily_query_count}/{user.daily_query_limit} queries")
            
            # Get remaining queries to ensure we have the latest count from DB
            remaining_queries = user.get_remaining_queries()
            logger.info(f"User {user.username} has {remaining_queries} remaining queries")
            
            if remaining_queries <= 0:
                logger.warning(f"Query limit reached for user {user.username} (id: {user.id})")
                return Response(
                    {"error": "Daily query limit reached"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Extract and validate query parameters - standardized
            original_query_text = request.data.get('query_text', '')
            diversity_score = float(request.data.get('diversity_score', 0.5))
            num_stances = int(request.data.get('num_stances', 3))
            
            if not original_query_text:
                logger.warning("Empty query text received")
                return Response(
                    {"error": "Query text is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            logger.info(f"Normalized query text: '{original_query_text}' (original: '{original_query_text}')")
            
            # Get system configuration first to ensure consistent cache key generation
            config = gpt_service.get_config()
            system_message = config['system_prompt']
            if isinstance(system_message, list):
                system_message = system_message[0]
            
            # Get base system content
            if isinstance(system_message, dict):
                system_content = system_message.get('content', gpt_service.DEFAULT_SYSTEM_PROMPT)
            else:
                system_content = str(system_message)
            
            # Add stance count to system prompt
            system_content = system_content + f"\n\nPlease attempt to provide at most or equal to {num_stances} different perspectives on this topic."
            
            # First, check the global cache
            cache_key = generate_cache_key(original_query_text, diversity_score, num_stances, system_content)
            logger.info(f"Checking cache with key: {cache_key}")
            cached_response = get_cached_response(cache_key)
            
            if cached_response:
                logger.info(f"✅ Cache HIT for key: {cache_key}")
                # Create a new query record for this user with the cached response
                query = Query.objects.create(
                    user=user,
                    query_text=original_query_text,
                    diversity_score=diversity_score,
                    response=cached_response,
                    system_prompt=system_content,
                    is_active=True
                )
                # Update user's query count
                before_count = user.daily_query_count
                increment_success = user.increment_query_count()
                
                if not increment_success:
                    logger.error(f"Failed to increment query count for user {user.username} (id: {user.id})")
                    return Response(
                        {"error": "Failed to update query count"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                
                logger.info(f"Query count for user {user.username} incremented from {before_count} to {user.daily_query_count}")
                
                logger.info(f"Query created successfully with ID: {query.id}")
                serializer = self.get_serializer(query)
                return Response(
                    serializer.data,
                    status=status.HTTP_200_OK,
                    headers={
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                    }
                )
            
            logger.info(f"❌ Cache MISS for key: {cache_key}")
            
            # If not in cache, check for existing queries across all users
            existing_query = Query.objects.filter(
                query_text__iexact=original_query_text,
                diversity_score=diversity_score,
                system_prompt=system_content,
                is_active=True
            ).first()
            
            if existing_query:
                logger.info(f"Found existing query with ID: {existing_query.id}, caching response")
                # Cache the existing response for future use by any user
                cache_success = set_cached_response(cache_key, existing_query.response)
                if cache_success:
                    logger.info(f"💾 Successfully cached response with key: {cache_key}")
                else:
                    logger.warning(f"⚠️ Failed to cache response with key: {cache_key}")
                
                # Create a new query record for this user with the existing response
                query = Query.objects.create(
                    user=user,
                    query_text=original_query_text,
                    diversity_score=diversity_score,
                    response=existing_query.response,
                    system_prompt=system_content,
                    is_active=True
                )
                # Update user's query count
                before_count = user.daily_query_count
                increment_success = user.increment_query_count()
                
                if not increment_success:
                    logger.error(f"Failed to increment query count for user {user.username} (id: {user.id})")
                    return Response(
                        {"error": "Failed to update query count"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                
                logger.info(f"Query count for user {user.username} incremented from {before_count} to {user.daily_query_count}")
                
                logger.info(f"Query created successfully with ID: {query.id}")
                serializer = self.get_serializer(query)
                return Response(
                    serializer.data,
                    status=status.HTTP_200_OK,
                    headers={
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                    }
                )
            
            # No cache hit and no existing query, generate new response
            logger.info(f"No existing query found, calling GPT service")
            gpt_response = complete(original_query_text, diversity_score, num_stances)
            logger.info(f"Received GPT response: {json.dumps(gpt_response, indent=2)}")
            
            # Cache the new response for future use by any user
            cache_success = set_cached_response(cache_key, gpt_response)
            if cache_success:
                logger.info(f"💾 Successfully cached new response with key: {cache_key}")
            else:
                logger.warning(f"⚠️ Failed to cache new response with key: {cache_key}")
            
            # Create new query record
            query = Query.objects.create(
                user=user,
                query_text=original_query_text,
                diversity_score=diversity_score,
                response=gpt_response,
                system_prompt=system_content,
                is_active=True
            )
            
            # Update user's query count
            before_count = user.daily_query_count
            increment_success = user.increment_query_count()
            
            if not increment_success:
                logger.error(f"Failed to increment query count for user {user.username} (id: {user.id})")
                return Response(
                    {"error": "Failed to update query count"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
            logger.info(f"Query count for user {user.username} incremented from {before_count} to {user.daily_query_count}")
            
            logger.info(f"Query created successfully with ID: {query.id}")
            serializer = self.get_serializer(query)
            
            return Response(
                serializer.data,
                status=status.HTTP_201_CREATED,
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                }
            )
        except Exception as e:
            logger.error(f"Error creating query: {str(e)}", exc_info=True)
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

#     @action(
#         detail=False,
#         methods=['POST'],
#         permission_classes=[AllowAny],
#         renderer_classes=[EventStreamRenderer]
#     )
#     def stream(self, request):
#         """Streaming endpoint for queries."""
#         try:
#             # Get query parameters from POST data
#             query_text = request.data.get('query_text')
#             diversity_score = float(request.data.get('diversity_score', 0.5))

#             if not query_text:
#                 return Response(
#                     {'error': 'query_text is required'},
#                     status=status.HTTP_400_BAD_REQUEST
#                 )

#             logger.info(f"Starting stream for query: '{query_text}' with diversity {diversity_score}")

#             def stream_response():
#                 final_response = None
#                 try:
#                     for chunk in gpt_service.stream_complete(query_text, diversity_score):
#                         if chunk.get("error"):
#                             yield f"data: {json.dumps(chunk)}\n\n"
#                             return
#                         if chunk.get("isComplete"):
#                             final_response = chunk
#                         yield f"data: {json.dumps(chunk)}\n\n"

#                     # Create query record after streaming is complete
#                     if final_response and request.user.is_authenticated:
#                         logger.info("Stream complete, creating query record")
#                         Query.objects.create(
#                             user=request.user,
#                             query_text=query_text,
#                             diversity_score=diversity_score,
#                             response={"arguments": [final_response]},
#                             is_active=True
#                         )

#                 except Exception as e:
#                     logger.error(f"Error in stream_response: {str(e)}", exc_info=True)
#                     error_chunk = {
#                         "error": str(e),
#                         "isComplete": True
#                     }
#                     yield f"data: {json.dumps(error_chunk)}\n\n"

#             response = StreamingHttpResponse(
#                 streaming_content=stream_response(),
#                 content_type='text/event-stream'
#             )

#             # Set required headers
#             response['Cache-Control'] = 'no-cache'
#             response['X-Accel-Buffering'] = 'no'
#             response['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
#             response['Access-Control-Allow-Credentials'] = 'true'
#             response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
#             response['Access-Control-Allow-Headers'] = 'Accept, Content-Type, Authorization'

#             return response

#         except ValueError as e:
#             return Response(
#                 {'error': 'Invalid diversity_score value'},
#                 status=status.HTTP_400_BAD_REQUEST
#             )
#         except Exception as e:
#             logger.error(f"Error in stream endpoint: {str(e)}", exc_info=True)
#             return Response(
#                 {'error': str(e)},
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )

#     def destroy(self, request, *args, **kwargs):
#         query = self.get_object()
#         query.is_active = False
#         query.save()
#         return Response(status=status.HTTP_204_NO_CONTENT)

#     @action(detail=False, methods=['delete'])
#     def clear(self, request):
#         self.get_queryset().update(is_active=False)
#         return Response(status=status.HTTP_204_NO_CONTENT)

#     @action(detail=False, methods=['GET'])
#     def latest(self, request):
#         """Get the user's latest query."""
#         try:
#             latest_query = self.get_queryset().latest('created_at')
#             serializer = self.get_serializer(latest_query)
#             return Response(serializer.data)
#         except Query.DoesNotExist:
#             return Response(
#                 {'error': 'No queries found'},
#                 status=status.HTTP_404_NOT_FOUND
#             )

class ArgumentRatingViewSet(viewsets.ModelViewSet):
    serializer_class = ArgumentRatingSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get_queryset(self):
        return ArgumentRating.objects.filter(user=self.request.user)

    def list(self, request, *args, **kwargs):
        logger.info(f"→ Fetching ratings for user: {request.user.username}")
        
        # Get query parameter
        query_id = request.query_params.get('query')
        logger.info(f"→ Query ID filter: {query_id}")
        
        # Filter queryset
        queryset = self.get_queryset()
        if query_id:
            queryset = queryset.filter(query_id=query_id)
        
        # Log the found ratings
        ratings_count = queryset.count()
        logger.info(f"→ Found {ratings_count} ratings")
        
        # Serialize and return
        serializer = self.get_serializer(queryset, many=True)
        logger.info(f"→ Serialized data: {serializer.data}")
        
        return Response(
            serializer.data,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            }
        )
    
    def create(self, request, *args, **kwargs):
        """Create a new argument rating."""
        try:
            # Extract data from request
            query_id = request.data.get('query')
            stance = request.data.get('stance')
            core_argument = request.data.get('core_argument')
            rating = request.data.get('rating')
            
            # Validate query exists and belongs to user
            try:
                query = Query.objects.get(id=query_id, user=request.user)
            except Query.DoesNotExist:
                return Response(
                    {"error": "Query not found or unauthorized"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Get the model source from the query response
            model_source = None
            if query.response:
                model_source = query.response.get('model_source', 'gpt-4')
                if not model_source:
                    # Check if model source is in the stance data
                    for argument in query.response.get('arguments', []):
                        if (argument.get('stance') == stance and 
                            argument.get('core_argument') == core_argument):
                            model_source = argument.get('model_source', 'gpt-4')
                            break
            
            # Create or update the rating
            rating_obj, created = ArgumentRating.objects.update_or_create(
                user=request.user,
                query=query,
                stance=stance,
                core_argument=core_argument,
                defaults={
                    'rating': rating,
                    'argument_source': model_source or 'gpt-4'
                }
            )
            
            # Increment user's allstars if this is a new rating
            if created:
                new_allstars = request.user.increment_allstars()
            
            serializer = self.get_serializer(rating_obj)
            response_data = serializer.data
            if created:
                response_data['new_allstars'] = new_allstars
            
            return Response(response_data, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"✗ Error creating rating: {str(e)}", exc_info=True)
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                }
            )

class ThumbsRatingViewSet(viewsets.ModelViewSet):
    serializer_class = ThumbsRatingSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Return thumbs ratings for the current user."""
        return ThumbsRating.objects.filter(user=self.request.user)

    def create(self, request, *args, **kwargs):
        """Create a new thumbs rating."""
        try:
            # Extract data from request
            query_id = request.data.get('query')
            core_argument = request.data.get('core_argument')
            rating = request.data.get('rating')
            stance = request.data.get('stance')
            
            # Validate query exists and belongs to user
            try:
                query = Query.objects.get(id=query_id, user=request.user)
            except Query.DoesNotExist:
                return Response(
                    {"error": "Query not found or unauthorized"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Create or update the rating
            rating_obj, created = ThumbsRating.objects.update_or_create(
                user=request.user,
                query=query,
                core_argument=core_argument,
                defaults={
                    'rating': rating,
                    'stance': stance
                }
            )
            
            # Increment user's allstars if this is a new rating
            new_allstars = None
            if created:
                new_allstars = request.user.increment_allstars()
                # Refresh user instance to get the actual value
                request.user.refresh_from_db()
                
            serializer = self.get_serializer(rating_obj)
            response_data = serializer.data
            if created:
                # Ensure allstars is returned as a number
                response_data['allstars'] = int(request.user.allstars)
            
            return Response(response_data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def leaderboard(request):
    """
    Get the top 10 users sorted by AllStars count.
    """
    try:
        User = get_user_model()
        top_users = User.objects.all().order_by('-allstars')[:10]
        
        leaderboard_data = [
            {
                'first_name': user.first_name,
                'allstars': user.allstars
            }
            for user in top_users
        ]
        
        return Response(
            leaderboard_data,
            status=status.HTTP_200_OK,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            }
        )
    except Exception as e:
        logger.error(f"Error fetching leaderboard: {str(e)}")
        return Response(
            {"error": "Failed to fetch leaderboard data"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            }
        )

def async_view(func):
    """Decorator to handle async views"""
    @wraps(func)
    def wrapped(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapped

@api_view(['POST'])
def get_arguments(request):
    """Get arguments for a topic with caching"""
    try:
        # Extract and validate parameters - standardized with QueryViewSet
        topic = request.data.get('query_text', '')  # Changed to match QueryViewSet
        diversity = float(request.data.get('diversity_score', 0.5))  # Changed to match QueryViewSet
        num_stances = int(request.data.get('num_stances', 3))

        if not topic:
            return Response({'error': 'Topic is required'}, status=status.HTTP_400_BAD_REQUEST)
            
        # Normalize topic first
        normalized_topic = normalize_query(topic)
        logger.info(f"Normalized topic: '{normalized_topic}' (original: '{topic}')")

        # Get system configuration first to ensure consistent cache key generation
        config = gpt_service.get_config()
        system_message = config['system_prompt']
        if isinstance(system_message, list):
            system_message = system_message[0]
        
        # Get base system content
        if isinstance(system_message, dict):
            system_content = system_message.get('content', gpt_service.DEFAULT_SYSTEM_PROMPT)
        else:
            system_content = str(system_message)
        
        # Add stance count to system prompt
        system_content = system_content + f"\n\nPlease attempt to provide at most or equal to {num_stances} different perspectives on this topic."
        
        # Generate cache key and check cache first
        cache_key = generate_cache_key(normalized_topic, diversity, num_stances, system_content)
        logger.info(f"Checking cache with key: {cache_key}")
        cached_response = get_cached_response(cache_key)
        
        if cached_response:
            logger.info(f"✅ Cache HIT for topic: '{normalized_topic}'")
            return Response(cached_response)

        logger.info(f"❌ Cache MISS for key: {cache_key}")
        
        # If not in cache, check for existing queries across all users
        existing_query = Query.objects.filter(
            query_text__iexact=normalized_topic,
            diversity_score=diversity,
            system_prompt=system_content,
            is_active=True
        ).first()
        
        if existing_query:
            logger.info(f"Found existing query with ID: {existing_query.id}, caching response")
            # Cache the existing response for future use
            cache_success = set_cached_response(cache_key, existing_query.response)
            if cache_success:
                logger.info(f"💾 Successfully cached response with key: {cache_key}")
            else:
                logger.warning(f"⚠️ Failed to cache response with key: {cache_key}")
            return Response(existing_query.response)
        
        # No cache hit and no existing query, generate new response
        logger.info(f"No existing query found, calling GPT service")
        result = complete(normalized_topic, diversity, num_stances)
        
        # Cache the new response
        cache_success = set_cached_response(cache_key, result)
        if cache_success:
            logger.info(f"💾 Successfully cached new response with key: {cache_key}")
        else:
            logger.warning(f"⚠️ Failed to cache new response with key: {cache_key}")
        
        return Response(result)

    except Exception as e:
        logger.error(f"Error in get_arguments: {str(e)}", exc_info=True)
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAdminUser])
def cache_statistics(request):
    """Get cache statistics and health metrics"""
    try:
        stats = get_cache_stats()
        health = get_cache_health()
        
        # Store statistics in database
        CacheStatistics.objects.create(
            cache_hits=health.get('hit_rate_percent', 0) * stats.get('total_cached_entries', 0) / 100,
            cache_misses=(100 - health.get('hit_rate_percent', 0)) * stats.get('total_cached_entries', 0) / 100,
            memory_usage=stats.get('cache_size_bytes', 0) / (1024 * 1024),  # Convert to MB
            total_entries=stats.get('total_cached_entries', 0),
            hit_rate=health.get('hit_rate_percent', 0)
        )

        return Response({
            'statistics': stats,
            'health': health
        })

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# @api_view(['POST'])
# @permission_classes([IsAdminUser])
# @async_view
# async def clear_cache_view(request):
#     """Clear cache entries"""
#     try:
#         cache_key = request.data.get('cache_key')  # Optional
#         result = await clear_cache(cache_key)
#         return Response({'success': result})

#     except Exception as e:
#         return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAdminUser])
def optimize_cache_view(request):
    """Manually trigger cache optimization"""
    try:
        result = optimize_cache()
        return Response(result)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAdminUser])
def warm_cache_view(request):
    """Pre-warm cache with a list of topics"""
    try:
        topics = request.data.get('topics', [])
        diversity = float(request.data.get('diversity', 0.7))
        num_stances = int(request.data.get('num_stances', 3))

        if not topics:
            return Response({'error': 'Topics list is required'}, status=status.HTTP_400_BAD_REQUEST)

        result = warm_cache(topics, diversity, num_stances)
        return Response({'success': True, 'results': result})

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
