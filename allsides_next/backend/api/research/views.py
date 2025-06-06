import json
import logging
import asyncio
import time
import os
import re
import traceback
from typing import Dict, Any, Optional, List, AsyncGenerator
import uuid
from asgiref.sync import sync_to_async

from django.http import StreamingHttpResponse, JsonResponse
from django.utils import timezone
from django.shortcuts import get_object_or_404, render
from rest_framework import viewsets, status, renderers
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.authentication import JWTAuthentication

from .models import ResearchReport, ResearchSection
from .serializers import (
    ResearchReportSerializer,
    ResearchReportCreateSerializer, 
    ResearchReportUpdateSerializer,
    ResearchFeedbackSerializer,
    ResearchSectionSerializer
)
from .graph_manager import ResearchManager
from .utils import (
    generate_cache_key,
    get_cached_report,
    set_cached_report,
    save_thread_state,
    get_thread_state,
    format_markdown_for_frontend,
    extract_sections_from_plan,
    extract_interrupt_message,
    debug_redis_thread_state,
    debug_langgraph_event,
    ensure_thread_has_required_props,
    verify_openai_api_key,
    setup_sse_response,
    generate_sse_event,
    generate_sse_keepalive,
    handle_sse_stream
)
from ..models import Query

logger = logging.getLogger(__name__)

# Helper function to collect all events from an async generator
async def collect_events(async_gen):
    """
    Collects all events from an async generator.
    
    Args:
        async_gen: An async generator
        
    Returns:
        List of events from the generator
    """
    events = []
    last_event_time = time.time()
    
    try:
        async for event in async_gen:
            try:
                # If the event is already a string (already formatted SSE event)
                if isinstance(event, str):
                    # If it's a keepalive comment, use it as is
                    if event == ":\n\n":
                        events.append(event)
                    # If it already has the data: prefix, use it as is
                    elif event.startswith('data:'):
                        # Ensure it ends with double newlines
                        if not event.endswith('\n\n'):
                            event = event.rstrip() + '\n\n'
                        events.append(event)
                    else:
                        # Add data: prefix and ensure double newlines
                        events.append(f"data: {event.rstrip()}\n\n")
                # If the event is a dict, format it as an SSE event
                elif isinstance(event, dict):
                    events.append(f"data: {json.dumps(event)}\n\n")
                # If the event is a keepalive comment, use it as is
                elif event == ":\n\n":
                    events.append(event)
                else:
                    # For any other type, try to convert to string
                    events.append(f"data: {json.dumps({'type': 'event', 'data': str(event)})}\n\n")
                
                # Add keepalive events if more than 15 seconds have passed since last event
                current_time = time.time()
                if current_time - last_event_time > 15:
                    events.append(":\n\n")  # SSE keepalive comment
                last_event_time = current_time
                
            except Exception as e:
                logger.error(f"Error formatting event: {str(e)}", exc_info=True)
                # Add a safe error event
                events.append(f"data: {json.dumps({'type': 'error', 'message': f'Error formatting event: {str(e)}'})}\n\n")
    except Exception as e:
        logger.error(f"Error collecting events: {str(e)}", exc_info=True)
        events.append(f"data: {json.dumps({'type': 'error', 'message': f'Error collecting events: {str(e)}'})}\n\n")
    
    # Always add a final keepalive
    events.append(":\n\n")
    
    return events

# Add a custom renderer for EventSource
class EventSourceRenderer(renderers.BaseRenderer):
    """Renderer for text/event-stream content."""
    media_type = 'text/event-stream'
    format = 'txt'
    charset = 'utf-8'
    
    def render(self, data, accepted_media_type=None, renderer_context=None):
        """Render the event stream data."""
        # For debugging: trace what type of data we're getting
        if isinstance(data, (dict, list)):
            logger.debug(f"EventSourceRenderer: Converting {type(data).__name__} to SSE format")
        elif isinstance(data, str):
            logger.debug(f"EventSourceRenderer: Got string data, length: {len(data)}")
        else:
            logger.debug(f"EventSourceRenderer: Got data of type: {type(data).__name__}")
        
        # Handle streaming content already rendered
        if isinstance(data, str):
            # Check if the string is already a properly formatted SSE event
            is_keepalive = data == ":\n\n"
            has_data_prefix = data.startswith('data:')
            has_double_newline = data.endswith('\n\n')
            
            if is_keepalive or (has_data_prefix and has_double_newline):
                logger.debug("EventSourceRenderer: Passing through already formatted SSE event")
                return data.encode(self.charset)
            
            # Not formatted yet, so add data: prefix and double newlines
            logger.debug("EventSourceRenderer: Formatting string data as SSE event")
            return f"data: {data.rstrip()}\n\n".encode(self.charset)
        
        # For other data types, convert to JSON and format as SSE event
        try:
            logger.debug("EventSourceRenderer: Converting data to JSON and formatting as SSE event")
            json_str = json.dumps(data)
            return f"data: {json_str}\n\n".encode(self.charset)
        except (TypeError, ValueError) as e:
            # If JSON serialization fails, convert to string
            logger.error(f"EventSourceRenderer: Error serializing data: {str(e)}")
            return f"data: {str(data)}\n\n".encode(self.charset)

class ResearchReportViewSet(viewsets.ModelViewSet):
    """ViewSet for ResearchReport model."""
    
    serializer_class = ResearchReportSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    http_method_names = ['get', 'post', 'patch', 'delete', 'head', 'options']
    
    def get_permissions(self):
        """
        Override to allow token auth via query parameter for EventSource connections
        which can't send Authorization headers
        """
        if self.action == 'stream_report' and self.request.method == 'GET' and 'token' in self.request.GET:
            # Authenticate with token immediately, before permission checks
            user = self.perform_token_authentication(self.request)
            if user:
                # Successfully authenticated with token
                return []
        return super().get_permissions()
    
    def get_queryset(self):
        """Get reports for the current user."""
        user = self.request.user
        logger.debug(f"Getting reports for user {user.username} (ID: {user.id})")
        
        # If we're accessing the stream action with a specific ID, first check if this report exists
        if self.action == 'stream_report' and self.kwargs.get('pk'):
            report_id = self.kwargs.get('pk')
            # Check if report exists at all
            try:
                report = ResearchReport.objects.get(id=report_id)
                logger.debug(f"Found report {report_id}, belongs to user {report.user.id}")
                
                # Return all reports (will be filtered by permission checks later)
                # This prevents 404 errors when the report exists but permissions need to be checked
                return ResearchReport.objects.all()
            except ResearchReport.DoesNotExist:
                logger.warning(f"Report {report_id} does not exist")
                # Return empty queryset
                return ResearchReport.objects.none()
        
        # For normal operations, only return user's reports
        return ResearchReport.objects.filter(user=user)
    
    def get_serializer_class(self):
        """Get the appropriate serializer based on the action."""
        if self.action == 'create':
            return ResearchReportCreateSerializer
        elif self.action == 'partial_update':
            return ResearchReportUpdateSerializer
        return ResearchReportSerializer
    
    def create(self, request, *args, **kwargs):
        """Create a new research report."""
        try:
            # Extract topic from request
            topic = request.data.get('topic')
            if not topic:
                return Response({'error': 'Topic is required'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Get the authenticated user
            user = request.user
            
            # Check if the user has remaining queries
            if not user.is_staff and not user.is_superuser:  # Skip check for admins
                remaining_queries = user.get_remaining_queries()
                if remaining_queries <= 0:
                    return Response(
                        {'error': 'You have used all your available queries for this month'},
                        status=status.HTTP_403_FORBIDDEN
                    )
            
            # Get configuration options from request data
            config_data = request.data.get('config', {})
            logger.info(f"Received config from frontend: {config_data}")
            
            # Extract specific values with defaults
            search_api = config_data.get('search_api', 'tavily')
            planner_provider = config_data.get('planner_provider', 'openai')
            planner_model = config_data.get('planner_model', 'gpt-4o')
            writer_provider = config_data.get('writer_provider', 'openai')
            writer_model = config_data.get('writer_model', 'gpt-4o')
            
            logger.info(f"Using search_api: {search_api}, planner: {planner_provider}/{planner_model}, writer: {writer_provider}/{writer_model}")
            
            # Create config dictionary
            config = {
                "search_api": search_api,
                "planner_provider": planner_provider,
                "planner_model": planner_model,
                "writer_provider": writer_provider,
                "writer_model": writer_model,
            }
            
            # Fix Ollama model names if needed
            fixed_config = self._fix_ollama_model_names(config)
            
            # Check for the lite mode flag
            lite_mode = request.data.get('lite_mode', False)
            
            # In lite mode, use the same lite settings for both planner and writer
            if lite_mode and lite_mode.lower() == 'true':
                fixed_config["planner_provider"] = "ollama"
                fixed_config["planner_model"] = "llama3.2:1b"
                fixed_config["writer_provider"] = "ollama"
                fixed_config["writer_model"] = "llama3.2:1b"
                fixed_config["max_search_depth"] = 1
                logger.info("Using lite mode settings with llama3.2:1b")
            
            # For direct routes, return the response immediately
            if request.query_params.get('direct') == 'true':
                # Initialize research manager
                manager = ResearchManager(fixed_config)
                thread_id, thread = asyncio.run(manager.start_research(topic))
                
                # Return the report data
                serializer = ResearchReportSerializer(report)
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            
            # Initialize research manager
            manager = ResearchManager(fixed_config)
            thread_id, thread = asyncio.run(manager.start_research(topic))
            
            # Create a new research report
            report = ResearchReport.objects.create(
                user=user,
                topic=topic,
                thread_id=thread_id,
                status='PLANNING',
                config=fixed_config  # Use fixed config
            )
            
            # Save thread state to Redis
            save_thread_state(thread_id, thread)
            
            # Increment the user's query count
            user.increment_query_count()
            
            # Create the query record
            query = Query.objects.create(
                user=user,
                query_text=topic,
                diversity_score=0.0,  # Not applicable for research
                response={},  # Empty for research type
                system_prompt="Research Report"
            )
            report.query = query
            report.save()
            
            # Return the created report
            serializer = ResearchReportSerializer(report)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error creating research report: {str(e)}", exc_info=True)
            return Response(
                {"error": f"Failed to create research report: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'], url_path='feedback')
    def provide_feedback(self, request, pk=None):
        """Provide feedback on a research plan."""
        try:
            report = self.get_object()
            
            # Validate feedback
            serializer = ResearchFeedbackSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            feedback = serializer.validated_data['feedback']
            
            # Ensure report is in planning status
            if report.status != 'PLANNING':
                return Response(
                    {"error": "Feedback can only be provided in planning stage"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get thread state from Redis
            thread = get_thread_state(report.thread_id)
            if not thread:
                logger.warning(f"Thread state missing for report {report.id} (thread_id: {report.thread_id})")
                
                # If a report exists but thread state is missing, create a new thread state
                if report.status == 'PLANNING' and report.plan:
                    logger.info(f"Creating new thread state for report {report.id}")
                    # Create thread with all required properties
                    thread = ensure_thread_has_required_props({}, report.topic, report.thread_id)
                    # Save this thread state to Redis
                    save_thread_state(report.thread_id, thread)
                else:
                    return Response(
                        {"error": "Research session expired or not found"},
                        status=status.HTTP_404_NOT_FOUND
                    )
            else:
                # Ensure thread has all required properties
                logger.info(f"Ensuring thread has all required properties for feedback")
                thread = ensure_thread_has_required_props(thread, report.topic, report.thread_id)
                save_thread_state(report.thread_id, thread)
            
            # Return streaming response
            def feedback_stream():
                # Send an initial keep-alive comment to establish the connection
                yield ":\n\n"
                
                # Send connection established event
                event_data = {
                    'type': 'connection_established',
                    'message': 'Connected to research service'
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Initialize research manager with report config
                manager = ResearchManager(report.config)
                
                try:
                    # Create a new event loop for this stream
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Since provide_feedback returns an async generator, we need to iterate through it
                    async def process_events():
                        # Get the async generator directly
                        feedback_generator = manager.provide_feedback(thread, feedback)
                        
                        # Process each event from the generator
                        async for event in feedback_generator:
                            # Process each event
                            try:
                                # Handle non-dict events
                                if not isinstance(event, dict):
                                    serialized_event = {"raw_event": str(event)}
                                    logger.warning(f"Non-dict event in feedback stream: {type(event)}")
                                else:
                                    serialized_event = event
                                
                                # Add metadata
                                serialized_event['__meta'] = {
                                    'time': time.time(),
                                    'thread_id': report.thread_id  # Use report.thread_id instead of thread_id
                                }
                                
                                # Check for updated plan
                                if 'plan_updated' in serialized_event and serialized_event.get('sections'):
                                    # Update report with new plan from feedback
                                    sections = serialized_event.get('sections', [])
                                    if sections:
                                        logger.info(f"Updating report plan with {len(sections)} sections")
                                        report.plan = sections
                                        report.save()
                                        logger.info(f"Updated report plan with {len(sections)} sections from feedback")
                                
                                # Format for SSE
                                try:
                                    sse_data = f"data: {json.dumps(serialized_event)}\n\n"
                                    logger.debug(f"Successfully formatted SSE data")
                                    yield sse_data
                                except TypeError as e:
                                    # Handle JSON serialization errors
                                    logger.error(f"JSON serialization error: {e}, event: {type(serialized_event)}")
                                    # Create a safe version without problematic fields
                                    safe_event = {"type": "event", "message": "Event received but could not be fully serialized"}
                                    for key, value in serialized_event.items():
                                        try:
                                            # Test if this key/value can be serialized
                                            json.dumps({key: value})
                                            safe_event[key] = value
                                        except:
                                            safe_event[f"{key}_error"] = f"Could not serialize value of type {type(value)}"
                                    
                                    yield f"data: {json.dumps(safe_event)}\n\n"
                                
                                # Keepalive after important events
                                if isinstance(event, dict) and (
                                    'sections' in event or 
                                    'completed_sections' in event or
                                    'compile_final_report' in event
                                ):
                                    yield ":\n\n"  # Keepalive after important updates
                                    last_keepalive = time.time()
                            except Exception as e:
                                logger.error(f"Error processing feedback event: {e}")
                                # Return a simple error event
                                yield f"data: {json.dumps({'type': 'error', 'message': f'Error processing event: {str(e)}'})}\n\n"
                    
                    # Run the async function and collect all event data
                    all_events = loop.run_until_complete(collect_events(process_events()))
                    
                    # Always save the thread state after processing
                    try:
                        # Save thread state back to Redis
                        logger.info(f"Saving thread state to Redis after processing for thread_id: {report.thread_id}")
                        save_thread_state(report.thread_id, thread)
                    except Exception as e:
                        logger.error(f"Error saving thread state: {e}", exc_info=True)
                    
                    # Yield each event to the SSE stream
                    for event_data in all_events:
                        yield event_data
                        
                except Exception as e:
                    logger.error(f"Error in feedback stream: {str(e)}", exc_info=True)
                    error_data = {
                        'type': 'error',
                        'message': f'Error processing feedback: {str(e)}'
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                
                # Send a final message before closing
                yield f"data: {json.dumps({'type': 'stream_complete'})}\n\n"
            
            return StreamingHttpResponse(
                feedback_stream(),
                content_type='text/event-stream'
            )
            
        except Exception as e:
            logger.error(f"Error providing feedback: {str(e)}", exc_info=True)
            return Response(
                {"error": f"Failed to provide feedback: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'], url_path='approve')
    def approve_plan(self, request, pk=None):
        """Approve the research plan and start report generation."""
        try:
            report = self.get_object()
            
            # Ensure report is in planning status
            if report.status != 'PLANNING':
                return Response(
                    {"error": "Approval can only be done in planning stage"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get thread state from Redis
            thread = get_thread_state(report.thread_id)
            if not thread:
                logger.warning(f"Thread state missing for report {report.id} (thread_id: {report.thread_id})")
                
                # If a report exists but thread state is missing, attempt to reconstruct minimal thread state
                if report.status == 'PLANNING' and report.plan:
                    logger.info(f"Creating fallback thread state for report {report.id}")
                    # Create minimal thread state with just the thread_id
                    thread = {
                        "configurable": {
                            "thread_id": report.thread_id,
                            "topic": report.topic
                        }
                    }
                    # Save this fallback thread state to Redis
                    save_thread_state(report.thread_id, thread)
                else:
                    return Response(
                        {"error": "Research session expired or not found"},
                        status=status.HTTP_404_NOT_FOUND
                    )
            
            # Ensure thread has the topic in configurable
            if "configurable" in thread and "topic" not in thread["configurable"]:
                logger.info(f"Adding missing topic to thread configurable: {report.topic}")
                thread["configurable"]["topic"] = report.topic
                save_thread_state(report.thread_id, thread)
            
            # Update report status
            report.status = 'RESEARCHING'
            report.save()

            # Additional validation and logging for thread contents
            if 'configurable' in thread:
                logger.info(f"Thread configurable before approval: {thread['configurable']}")
                
                # Ensure topic is in configurable
                if 'topic' not in thread['configurable']:
                    logger.warning(f"Topic missing from thread configurable, adding it now: {report.topic}")
                    thread['configurable']['topic'] = report.topic
                
                # Log the final state
                logger.info(f"Final thread configurable: {thread['configurable']}")
                save_thread_state(report.thread_id, thread)
            else:
                logger.error(f"Thread is missing 'configurable' field: {thread}")
                # Create configurable if it doesn't exist
                thread['configurable'] = {
                    "thread_id": report.thread_id,
                    "topic": report.topic,
                    "planner_model": "gpt-4o",
                    "writer_model": "gpt-4o"
                }
                logger.info(f"Created configurable for thread: {thread['configurable']}")
                save_thread_state(report.thread_id, thread)
            
            # Additional validation and logging for thread contents
            logger.info(f"Ensuring thread has all required properties before approval")
            thread = ensure_thread_has_required_props(thread, report.topic, report.thread_id)
            logger.info(f"Final thread configurable: {thread['configurable']}")
            save_thread_state(report.thread_id, thread)
            
            # Return streaming response
            def approve_stream():
                # Send an initial keep-alive comment to establish the connection
                yield ":\n\n"
                
                # Send connection established event
                event_data = {
                    'type': 'connection_established',
                    'message': 'Connected to research service'
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Initialize research manager with report config
                manager = ResearchManager(report.config)
                
                try:
                    # Create a new event loop for this stream
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Since approve_plan returns an async generator, we need to iterate through it
                    async def process_events():
                        # Get the async generator directly
                        approval_generator = manager.approve_plan(thread)
                        
                        # Process each event from the generator
                        async for event in approval_generator:
                            # Process each event
                            try:
                                # Handle non-dict events
                                if not isinstance(event, dict):
                                    serialized_event = {"raw_event": str(event)}
                                    logger.warning(f"Non-dict event in approval stream: {type(event)}")
                                else:
                                    serialized_event = event
                                
                                # Add additional debug info
                                logger.info(f"Processing approval event type: {type(event)}, keys: {event.keys() if isinstance(event, dict) else 'N/A'}")
                                
                                # Add metadata
                                serialized_event['__meta'] = {
                                    'time': time.time(),
                                    'thread_id': report.thread_id  # Use report.thread_id instead of thread_id
                                }
                                
                                # Check for completion
                                if 'compile_final_report' in serialized_event:
                                    # Update report
                                    if 'final_report' in serialized_event['compile_final_report']:
                                        logger.info("Found final report content, updating report")
                                        report.content = serialized_event['compile_final_report']['final_report']
                                        report.status = 'COMPLETED'
                                        report.completed_at = timezone.now()
                                        report.save()
                                
                                # For section building phase
                                if 'build_section_with_web_research' in serialized_event:
                                    logger.info("Section building phase detected")
                                    # Make sure we're in RESEARCHING state
                                    report.status = 'RESEARCHING'
                                    report.save()
                                
                                # Format for SSE
                                try:
                                    sse_data = f"data: {json.dumps(serialized_event)}\n\n"
                                    logger.debug(f"Successfully formatted SSE data")
                                    yield sse_data
                                except TypeError as e:
                                    # Handle JSON serialization errors
                                    logger.error(f"JSON serialization error: {e}, event: {type(serialized_event)}")
                                    # Create a safe version without problematic fields
                                    safe_event = {"type": "event", "message": "Event received but could not be fully serialized"}
                                    for key, value in serialized_event.items():
                                        try:
                                            # Test if this key/value can be serialized
                                            json.dumps({key: value})
                                            safe_event[key] = value
                                        except:
                                            safe_event[f"{key}_error"] = f"Could not serialize value of type {type(value)}"
                                    
                                    yield f"data: {json.dumps(safe_event)}\n\n"
                                
                                # Keepalive after important events
                                if isinstance(event, dict) and (
                                    'sections' in event or 
                                    'completed_sections' in event or
                                    'compile_final_report' in event
                                ):
                                    yield ":\n\n"  # Keepalive after important updates
                                    last_keepalive = time.time()
                            except Exception as e:
                                logger.error(f"Error processing approval event: {e}", exc_info=True)
                                # Return a simple error event
                                yield f"data: {json.dumps({'type': 'error', 'message': f'Error processing event: {str(e)}'})}\n\n"
                    
                    # Run the async function and collect all event data
                    all_events = loop.run_until_complete(collect_events())
                    
                    # Always save the thread state after processing
                    try:
                        # Save thread state back to Redis
                        logger.info(f"Saving thread state to Redis after processing for thread_id: {report.thread_id}")
                        save_thread_state(report.thread_id, thread)
                    except Exception as e:
                        logger.error(f"Error saving thread state: {e}", exc_info=True)
                    
                    # Yield each event to the SSE stream
                    for event_data in all_events:
                        yield event_data
                        
                except Exception as e:
                    logger.error(f"Error in approval stream: {str(e)}", exc_info=True)
                    error_data = {
                        'type': 'error',
                        'message': f'Error processing research: {str(e)}'
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    
                    # Update report status on error
                    report.status = 'FAILED'
                    report.save()
                
                # Send a final message before closing
                yield f"data: {json.dumps({'type': 'stream_complete'})}\n\n"

            return StreamingHttpResponse(
                approve_stream(),
                content_type='text/event-stream'
            )
            
        except Exception as e:
            logger.error(f"Error approving research plan: {str(e)}", exc_info=True)
            
            # Update report status on error
            try:
                report.status = 'FAILED'
                report.save()
            except:
                pass
                
            return Response(
                {"error": f"Failed to approve research plan: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(
        detail=True, 
        methods=['get'], 
        url_path='stream',
        url_name='report-stream',
        renderer_classes=[EventSourceRenderer]
    )
    def stream_report(self, request, pk=None):
        """Stream research report events through the entire research process."""
        try:
            logger.debug(f"Received stream request for report {pk}")
            report = self.get_object()
            logger.debug(f"Found report {report.id} with thread_id {report.thread_id}")
            
            thread_id = report.thread_id
            
            # Get thread state
            thread = get_thread_state(thread_id)
            if not thread:
                logger.debug(f"No thread state found for {thread_id}, creating new one")
                thread = ensure_thread_has_required_props({}, report.topic, thread_id)
                save_thread_state(thread_id, thread)
                logger.debug(f"Created and saved new thread state for {thread_id}")
            else:
                # Always ensure thread has the topic properly set
                thread = ensure_thread_has_required_props(thread, report.topic, thread_id)
                save_thread_state(thread_id, thread)
                logger.debug(f"Updated thread with topic at root and configurable")
            
            # Ensure topic is in the thread at root level for LangGraph compatibility
            if 'topic' not in thread:
                thread['topic'] = report.topic
                logger.debug(f"Added topic to root level of thread: {report.topic}")
            
            # Initialize research manager
            manager = ResearchManager(report.config)
            logger.debug(f"Initialized ResearchManager with config: {report.config}")
            
            # Update report status immediately to start research
            report.status = 'RESEARCHING'
            report.save(update_fields=['status'])
            logger.debug(f"Updated report status to RESEARCHING")
            
            def event_stream():
                # Send initial connection event with keepalives
                logger.debug("Starting event stream")
                yield f"data: {json.dumps({'type': 'connection_established'})}\n\n"
                yield ":\n\n"  # First keepalive
                yield ":\n\n"  # Double keepalive to ensure connection
                
                # Send current status with force_update flag
                status_event = {
                    'type': 'report_status',
                    'status': 'RESEARCHING',
                    'plan': report.plan or [],
                    'force_update': True,
                    'timestamp': time.time()
                }
                yield f"data: {json.dumps(status_event)}\n\n"
                yield ":\n\n"  # Keepalive
                
                # Create event loop for async operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Track state to prevent loops
                sections_received = set()
                plan_received = False
                completion_detected = False
                last_keepalive = time.time()
                keepalive_interval = 5  # Send keepalive every 5 seconds
                
                # Create sync_to_async wrappers for database operations
                update_report_status = sync_to_async(lambda r, s: setattr(r, 'status', s) or r.save(update_fields=['status']))
                update_report_content = sync_to_async(lambda r, c, s: setattr(r, 'content', c) and setattr(r, 'status', s) and setattr(r, 'completed_at', timezone.now()) and r.save(update_fields=['status', 'content', 'completed_at']))
                create_or_update_section = sync_to_async(lambda r, n, c: ResearchSection.objects.update_or_create(report=r, name=n, defaults={'content': c}))
                save_report_plan = sync_to_async(lambda r, p: setattr(r, 'plan', p) and r.save(update_fields=['plan']))
                
                try:
                    # Define async processing function that will be passed to collect_events
                    async def process_events():
                        nonlocal sections_received, plan_received, completion_detected, last_keepalive
                        
                        # Ensure topic is set in the thread before streaming
                        if 'configurable' in thread:
                            thread['configurable']['topic'] = report.topic
                        thread['topic'] = report.topic
                        
                        # Log important debug info before streaming
                        logger.debug(f"Thread keys before stream_results: {list(thread.keys())}")
                        logger.debug(f"Topic in thread root: {thread.get('topic')}")
                        
                        try:
                            # Stream research results for the entire process
                            async for event in manager.stream_results(report.topic, thread):
                                try:
                                    # Add timestamp to all events
                                    if isinstance(event, dict):
                                        event['timestamp'] = time.time()
                                        
                                        # Send keepalive if needed
                                        current_time = time.time()
                                        if current_time - last_keepalive > keepalive_interval:
                                            yield ":\n\n"  # Keepalive
                                            last_keepalive = current_time
                                        
                                        # Process each event type appropriately
                                        # Handle plan events
                                        if 'sections' in event and not plan_received:
                                            plan_received = True
                                            # Update report plan in database
                                            await save_report_plan(report, event['sections'])
                                            logger.info(f"Saved plan with {len(event['sections'])} sections")
                                            
                                            # Add custom fields to indicate plan is ready and auto-approved
                                            event['auto_approved'] = True
                                            event['type'] = event.get('type') or 'plan_ready'
                                            event['force_update'] = True
                                        
                                        # Check for interrupt events that might contain the plan
                                        if '__interrupt__' in event and not plan_received:
                                            # Extract sections from the interrupt if possible
                                            interrupt_data = event['__interrupt__']
                                            logger.info(f"Processing interrupt data: {type(interrupt_data)}")
                                            
                                            try:
                                                # Try to extract sections from different interrupt formats
                                                extracted_sections = None
                                                
                                                if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
                                                    if 'value' in interrupt_data[0]:
                                                        value = interrupt_data[0]['value']
                                                        if isinstance(value, dict) and 'sections' in value:
                                                            extracted_sections = value['sections']
                                                        elif isinstance(value, str) and 'Section:' in value:
                                                            extracted_sections = extract_sections_from_plan(value)
                                                elif hasattr(interrupt_data, '__iter__') and not isinstance(interrupt_data, (dict, str)):
                                                    # This might be a tuple of Interrupt objects
                                                    logger.info(f"Processing iterative interrupt: {len(interrupt_data)} items")
                                                    for item in interrupt_data:
                                                        # Check if it's an Interrupt object
                                                        if hasattr(item, 'value'):
                                                            value = item.value
                                                            if isinstance(value, dict) and 'sections' in value:
                                                                extracted_sections = value['sections']
                                                                logger.info(f"Found sections in Interrupt object")
                                                                break
                                                            elif isinstance(value, str) and 'Section:' in value:
                                                                extracted_sections = extract_sections_from_plan(value)
                                                                logger.info(f"Extracted sections from Interrupt string")
                                                                break
                                                
                                                # If we found sections, update the report
                                                if extracted_sections:
                                                    plan_received = True
                                                    await save_report_plan(report, extracted_sections)
                                                    logger.info(f"Extracted plan from interrupt with {len(extracted_sections)} sections")
                                                    
                                                    # Replace interrupt with sections for better frontend handling
                                                    event = {
                                                        'type': 'plan_ready',
                                                        'sections': extracted_sections,
                                                        'auto_approved': True,
                                                        'force_update': True,
                                                        'timestamp': time.time()
                                                    }
                                            except Exception as extract_err:
                                                logger.error(f"Error extracting sections from interrupt: {str(extract_err)}")
                                        
                                        # Handle report status updates
                                        if event.get('type') == 'report_status':
                                            status = event.get('status')
                                            if status and status != report.status:
                                                await update_report_status(report, status)
                                                logger.info(f"Updated report status to {status}")
                                            
                                            # Always ensure force_update flag is set
                                            event['force_update'] = True
                                        
                                        # Handle section completion 
                                        if 'completed_sections' in event:
                                            sections = event.get('completed_sections', [])
                                            for section in sections:
                                                # Extract section info
                                                if hasattr(section, 'name') and hasattr(section, 'content'):
                                                    name = section.name
                                                    content = section.content
                                                elif isinstance(section, dict):
                                                    name = section.get('name', '')
                                                    content = section.get('content', '')
                                                else:
                                                    continue
                                                    
                                                # Skip if already processed
                                                if name in sections_received:
                                                    continue
                                                    
                                                # Mark as received and save to database
                                                sections_received.add(name)
                                                logger.info(f"Saving completed section: {name}")
                                                
                                                # Create or update section in database
                                                await create_or_update_section(report, name, content)
                                        
                                        # If we're receiving completed sections, we're in research phase
                                        if report.status != 'COMPLETED':
                                            await update_report_status(report, 'RESEARCHING')
                                            
                                        # Make sure event has a type
                                        if 'type' not in event:
                                            event['type'] = 'section_completed'
                                        
                                        # Add type to events that don't have one
                                        if 'type' not in event:
                                            if 'build_section_with_web_research' in event:
                                                event['type'] = 'researching_section'
                                            elif 'compile_final_report' in event:
                                                event['type'] = 'report_completed'
                                        
                                        # Handle final report completion
                                        if ('compile_final_report' in event and 
                                                'final_report' in event.get('compile_final_report', {}) and 
                                                not completion_detected):
                                            completion_detected = True
                                            final_report = event['compile_final_report']['final_report']
                                            
                                            # Update report status and content
                                            await update_report_content(report, final_report, 'COMPLETED')
                                            
                                            logger.info(f"Completed report: {report.id}")
                                            
                                            # Add completion marker to event
                                            event['report_complete'] = True
                                            event['report_id'] = report.id
                                            event['type'] = 'report_completed'
                                            event['force_update'] = True
                                    
                                    # Format for SSE - with special handling for non-JSON-serializable objects
                                    try:
                                        # Check for Interrupt objects or other non-serializable types
                                        if not isinstance(event, (dict, list, str, int, float, bool, type(None))):
                                            # Convert to a simple event dict
                                            yield f"data: {json.dumps({'type': 'message', 'content': str(event), 'timestamp': time.time()})}\n\n"
                                        elif isinstance(event, dict) and '__interrupt__' in event:
                                            # Handle special case for __interrupt__ field
                                            safe_event = event.copy()
                                            # Convert interrupt to safe representation
                                            interrupt_data = event['__interrupt__']
                                            
                                            if hasattr(interrupt_data, '__iter__') and not isinstance(interrupt_data, (str, dict)):
                                                # For tuples or lists of interrupts
                                                interrupt_list = []
                                                for item in interrupt_data:
                                                    if hasattr(item, 'value'):
                                                        # Extract Interrupt.value
                                                        interrupt_list.append({"value": str(item.value)})
                                                    else:
                                                        interrupt_list.append({"value": str(item)})
                                                safe_event['__interrupt__'] = interrupt_list
                                            else:
                                                # Single item
                                                safe_event['__interrupt__'] = [{"value": str(interrupt_data)}]
                                                
                                            yield f"data: {json.dumps(safe_event)}\n\n"
                                        else:
                                            # Regular serializable object
                                            yield f"data: {json.dumps(event)}\n\n"
                                            
                                    except TypeError as json_err:
                                        logger.error(f"JSON serialization error: {json_err}, event type: {type(event)}")
                                        # Create a sanitized version of the event
                                        safe_event = {"type": "event", "message": f"Event received but could not be serialized: {type(event).__name__}", "timestamp": time.time()}
                                        
                                        # Try to copy serializable fields
                                        if isinstance(event, dict):
                                            for key, value in event.items():
                                                try:
                                                    # Test if this key/value can be serialized
                                                    json.dumps({key: value})
                                                    safe_event[key] = value
                                                except (TypeError, ValueError):
                                                    safe_event[f"{key}_type"] = str(type(value).__name__)
                                        
                                        yield f"data: {json.dumps(safe_event)}\n\n"
                                except Exception as e:
                                    logger.error(f"Error processing event: {str(e)}", exc_info=True)
                                    # Create a well-structured error event
                                    error_event = {
                                        'type': 'error',
                                        'message': f"Error processing research event: {str(e)}",
                                        'error_type': e.__class__.__name__,
                                        'timestamp': time.time()
                                    }
                                    yield f"data: {json.dumps(error_event)}\n\n"
                                    yield ":\n\n"  # Keepalive after error
                        except Exception as stream_error:
                            logger.error(f"Error in research stream: {str(stream_error)}", exc_info=True)
                            # Create a detailed error message for the client
                            error_event = {
                                'type': 'error',
                                'message': f"Research stream error: {str(stream_error)}",
                                'error_type': stream_error.__class__.__name__,
                                'details': traceback.format_exc()[:500],  # Include truncated stack trace
                                'timestamp': time.time()
                            }
                            yield f"data: {json.dumps(error_event)}\n\n"
                            yield ":\n\n"  # Keepalive after error
                    
                    # Collect events from the process_events generator
                    async_events = loop.run_until_complete(collect_events(process_events()))
                    
                    # Yield each collected event
                    for event in async_events:
                        yield event
                    
                    # Send a final completion event if needed
                    if not completion_detected:
                        logger.info(f"Stream complete but no completion event detected, sending final status")
                        yield f"data: {json.dumps({'type': 'stream_complete', 'message': 'Research stream completed', 'timestamp': time.time()})}\n\n"
                        
                except Exception as e:
                    logger.error(f"Error in stream: {str(e)}", exc_info=True)
                    
                    # Update report status on error
                    report.status = 'FAILED'
                    report.save(update_fields=['status'])
                    
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'status': 'FAILED', 'timestamp': time.time()})}\n\n"
                finally:
                    loop.close()
                    
                    # Send final keepalive before closing
                    yield ":\n\n"
                    
                    # Save thread state to persist any changes
                    save_thread_state(thread_id, thread)
            
            # Create streaming response with headers
            response = StreamingHttpResponse(
                event_stream(),
                content_type='text/event-stream'
            )
            
            # Add required headers
            response['Cache-Control'] = 'no-cache, no-transform'
            response['X-Accel-Buffering'] = 'no'
            response['Connection'] = 'keep-alive'
            
            return response
            
        except Exception as e:
            logger.error(f"Error setting up stream: {str(e)}", exc_info=True)
            
            # Update report status on error
            try:
                # Use sync operation here since we're not in an async context
                report.status = 'FAILED'
                report.save(update_fields=['status'])
            except:
                pass
                
            return Response(
                {"error": f"Failed to stream report: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def perform_token_authentication(self, request):
        """
        Authenticate using token from query parameter
        """
        token = request.GET.get('token')
        if not token:
            return None
            
        try:
            from rest_framework_simplejwt.tokens import AccessToken
            from django.contrib.auth import get_user_model

            # Decode token
            validated_token = AccessToken(token)
            user_id = validated_token.get('user_id')
            
            # Get user
            User = get_user_model()
            user = User.objects.get(id=user_id)
            
            # Don't try to set is_authenticated as it's a property
            # Instead, just set the user on the request
            request.user = user
            # Add _authentication_backend to user so DRF knows user is authenticated
            setattr(user, 'backend', 'django.contrib.auth.backends.ModelBackend')
            
            logger.debug(f"Token authentication successful for user {user.username} (id: {user_id})")
            return user
        except Exception as e:
            logger.error(f"Token auth error: {str(e)}", exc_info=True)
            return None

    # Define a helper method to fix Ollama model names
    def _fix_ollama_model_names(self, config):
        """Helper to fix incorrect Ollama model names in config."""
        if not isinstance(config, dict):
            return config
            
        # Make a copy to avoid modifying the original
        fixed_config = config.copy()
        
        # Fix planner model
        if fixed_config.get("planner_provider") == "ollama":
            planner_model = fixed_config.get("planner_model", "")
            if planner_model in ["llama3", "llama3.2"]:
                logger.warning(f"Fixing incorrect Ollama model name: {planner_model} -> llama3.2:1b")
                fixed_config["planner_model"] = "llama3.2:1b"
                
        # Fix writer model
        if fixed_config.get("writer_provider") == "ollama":
            writer_model = fixed_config.get("writer_model", "")
            if writer_model in ["llama3", "llama3.2"]:
                logger.warning(f"Fixing incorrect Ollama model name: {writer_model} -> llama3.2:1b")
                fixed_config["writer_model"] = "llama3.2:1b"
                
        return fixed_config

class StartResearchView(APIView):
    """API view to start research and get streaming updates."""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    renderer_classes = [EventSourceRenderer, renderers.JSONRenderer]
    
    # Helper method to fix Ollama model names
    def _fix_ollama_model_names(self, config):
        """Helper to fix incorrect Ollama model names in config."""
        if not isinstance(config, dict):
            return config
            
        # Make a copy to avoid modifying the original
        fixed_config = config.copy()
        
        # Fix planner model
        if fixed_config.get("planner_provider") == "ollama":
            planner_model = fixed_config.get("planner_model", "")
            if planner_model in ["llama3", "llama3.2"]:
                logger.warning(f"Fixing incorrect Ollama model name: {planner_model} -> llama3.2:1b")
                fixed_config["planner_model"] = "llama3.2:1b"
                
        # Fix writer model
        if fixed_config.get("writer_provider") == "ollama":
            writer_model = fixed_config.get("writer_model", "")
            if writer_model in ["llama3", "llama3.2"]:
                logger.warning(f"Fixing incorrect Ollama model name: {writer_model} -> llama3.2:1b")
                fixed_config["writer_model"] = "llama3.2:1b"
                
        return fixed_config
    
    def post(self, request):
        """Start research on a topic and stream updates."""
        try:
            # Extract parameters from request
            topic = request.data.get('topic', '')
            if not topic:
                return Response({'error': 'Topic is required'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Get configuration options
            search_api = request.data.get('search_api', 'tavily')
            planner_provider = request.data.get('planner_provider', 'openai')
            planner_model = request.data.get('planner_model', 'gpt-4o')
            writer_provider = request.data.get('writer_provider', 'openai')
            writer_model = request.data.get('writer_model', 'gpt-4o')
            
            # Create config
            config = {
                'search_api': search_api,
                'planner_provider': planner_provider,
                'planner_model': planner_model,
                'writer_provider': writer_provider,
                'writer_model': writer_model,
            }
            
            # Fix Ollama model names
            fixed_config = self._fix_ollama_model_names(config)
            
            # Initialize research manager
            manager = ResearchManager(fixed_config)
            
            # Create a thread ID
            thread_id = str(uuid.uuid4())
            
            # Initialize the report
            thread_id, thread = asyncio.run(manager.start_research(topic))
            
            # Create the report in the database
            report = ResearchReport.objects.create(
                user=request.user,
                topic=topic,
                thread_id=thread_id,
                status='PLANNING',
                config=fixed_config  # Use fixed config
            )
            
            # Save thread state to Redis
            save_thread_state(thread_id, thread)
            
            # Create a new event loop for streaming
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Define a function to process events before sending
            def process_event(event):
                # Check for error events
                if event.get('type') == 'error':
                    logger.error(f"Research error: {event.get('message')}")
                    # Use sync operation as we're in a synchronous context here
                    report.status = 'FAILED'
                    report.save()
                    return event
                
                # Check for section data
                if 'sections' in event:
                    # Extract plan data and save
                    sections_data = []
                    sections = event.get('sections', [])
                    
                    # Process sections
                    for section in sections:
                        if isinstance(section, dict):
                            sections_data.append({
                                'name': section.get('name', ''),
                                'description': section.get('description', ''),
                                'research': section.get('research', True)
                            })
                    
                    # Update the report with the plan
                    if sections_data:
                        # Use sync operation as we're in a synchronous context here
                        report.plan = {'sections': sections_data}
                        report.status = 'RESEARCHING'
                        report.save()
                
                # Check for section writing updates
                if 'build_section_with_web_research' in event and 'section' in event.get('build_section_with_web_research', {}):
                    # Update report status
                    report.status = 'RESEARCHING'
                    report.save()
                    
                    # Extract section details
                    section_data = event['build_section_with_web_research'].get('section', {})
                    section_name = section_data.get('name', '')
                    
                    if section_name:
                        # Add a section update event
                        event['section_update'] = {
                            'section_name': section_name,
                            'status': 'researching'
                        }
                
                # Check for completed sections
                if 'completed_sections' in event:
                    sections = event.get('completed_sections', [])
                    for section in sections:
                        if isinstance(section, dict):
                            name = section.get('name', '')
                            content = section.get('content', '')
                            
                            if name and content:
                                # Create or update section in the database
                                ResearchSection.objects.update_or_create(
                                    report=report,
                                    name=name,
                                    defaults={'content': content}
                                )
                
                # Check for final report
                if 'compile_final_report' in event and 'final_report' in event.get('compile_final_report', {}):
                    # Extract the final report content
                    final_report = event['compile_final_report']['final_report']
                    
                    # Update the report
                    report.content = final_report
                    report.status = 'COMPLETED'
                    report.completed_at = timezone.now()
                    report.save()
                    
                    # Add completion information
                    event['report_complete'] = {
                        'report_id': report.id
                    }
                
                return event
            
            # Set up metadata for all events
            metadata = {
                'time': time.time(),
                'thread_id': thread_id,
                'report_id': report.id
            }
            
            # Create an async function to handle the streaming
            async def run_research_stream():
                # Use the handle_sse_stream helper to manage the SSE connection
                async for event_data in handle_sse_stream(
                    manager.run_research(topic),
                    process_event_fn=process_event,
                    metadata=metadata,
                    keepalive_interval=10  # Send keep-alive every 10 seconds (reduced from 15)
                ):
                    yield event_data
                    
                # Save thread state when done
                save_thread_state(thread_id, thread)
            
            # Create the streamed response
            response = StreamingHttpResponse(
                loop.run_until_complete(collect_events(run_research_stream())),
                content_type='text/event-stream'
            )
            
            # Set up the response with proper headers
            response = setup_sse_response(response)
            
            # Add specific retry timeout (browser will reconnect after 2 seconds if connection lost)
            response['retry'] = '2000'
            
            # Add CORS headers if needed
            if 'HTTP_ORIGIN' in request.META:
                response['Access-Control-Allow-Origin'] = request.META['HTTP_ORIGIN']
                response['Access-Control-Allow-Credentials'] = 'true'
                
            return response
            
        except Exception as e:
            logger.error(f"Error streaming research: {str(e)}", exc_info=True)
            
            # Update report status on error
            try:
                report.status = 'FAILED'
                report.save()
            except:
                pass
                
            return Response(
                {"error": f"Failed to stream research: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ResearchFeedbackView(APIView):
    """API view to provide feedback on a research plan."""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    renderer_classes = [EventSourceRenderer, renderers.JSONRenderer]
    
    def get_permissions(self):
        """
        Override to allow token auth via query parameter for EventSource connections
        which can't send Authorization headers
        """
        if self.request.method == 'POST' and 'token' in self.request.GET:
            # Authenticate with token immediately, before permission checks
            user = self.perform_token_authentication(self.request)
            if user:
                # Successfully authenticated with token
                return []
        return super().get_permissions()
    
    def perform_token_authentication(self, request):
        """
        Authenticate using token from query parameter
        """
        token = request.GET.get('token')
        if not token:
            return None
            
        try:
            from rest_framework_simplejwt.tokens import AccessToken
            from django.contrib.auth import get_user_model
            
            # Decode token
            validated_token = AccessToken(token)
            user_id = validated_token.get('user_id')
            
            # Get user
            User = get_user_model()
            user = User.objects.get(id=user_id)
            
            # Don't try to set is_authenticated as it's a property
            # Instead, just set the user on the request
            request.user = user
            # Add _authentication_backend to user so DRF knows user is authenticated
            setattr(user, 'backend', 'django.contrib.auth.backends.ModelBackend')
            
            logger.debug(f"Token authentication successful for user {user.username} (id: {user_id})")
            return user
        except Exception as e:
            logger.error(f"Token auth error: {str(e)}", exc_info=True)
            return None
    
    def post(self, request, thread_id):
        """Submit feedback on a research plan."""
        try:
            # Handle token in query parameter for EventSource
            if 'token' in request.GET:
                # Always try token auth if token is present, even if user is already authenticated
                user = self.perform_token_authentication(request)
                if not user:
                    logger.error("Token authentication failed")
                    return Response(
                        {"error": "Invalid token"},
                        status=status.HTTP_401_UNAUTHORIZED
                    )
                logger.debug(f"User authenticated from token: {user.username} (ID: {user.id})")
            elif not request.user.is_authenticated:
                # No token and not authenticated
                return Response(
                    {"error": "Authentication required"},
                    status=status.HTTP_401_UNAUTHORIZED
                )
                    
            # Verify the thread belongs to the current user
            report = get_object_or_404(
                ResearchReport, 
                thread_id=thread_id,
                user=request.user
            )
            
            # Validate feedback
            serializer = ResearchFeedbackSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
                
            feedback = serializer.validated_data['feedback']
            
            # Ensure report is in planning status
            if report.status != 'PLANNING':
                return Response(
                    {"error": "Feedback can only be provided in planning stage"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get thread state from Redis
            thread = get_thread_state(thread_id)
            if not thread:
                logger.warning(f"Thread state missing for report {report.id} (thread_id: {report.thread_id})")
                
                # If a report exists but thread state is missing, attempt to reconstruct minimal thread state
                if report.status == 'PLANNING' and report.plan:
                    logger.info(f"Creating fallback thread state for report {report.id}")
                    # Create minimal thread state with just the thread_id
                    thread = {
                        "configurable": {
                            "thread_id": report.thread_id,
                            "topic": report.topic,
                            "planner_model": "gpt-4o",
                            "writer_model": "gpt-4o"
                        }
                    }
                    # Save this fallback thread state to Redis
                    save_thread_state(report.thread_id, thread)
                else:
                    return Response(
                        {"error": "Research session expired or not found"},
                        status=status.HTTP_404_NOT_FOUND
                    )
            
            # Ensure thread has the topic in configurable
            if "configurable" in thread and "topic" not in thread["configurable"]:
                logger.info(f"Adding missing topic to thread configurable: {report.topic}")
                thread["configurable"]["topic"] = report.topic
                save_thread_state(report.thread_id, thread)
            
            # Define streaming function for feedback
            def feedback_stream():
                # Send an initial keep-alive comment to establish the connection
                yield ":\n\n"
                
                # Send connection established event
                event_data = {
                    'type': 'connection_established',
                    'message': 'Connected to research service'
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Initialize research manager with report config
                manager = ResearchManager(report.config)
                
                try:
                    # Create a new event loop for this stream
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Since provide_feedback returns an async generator, we need to iterate through it
                    async def process_events():
                        # Get the async generator directly
                        feedback_generator = manager.provide_feedback(thread, feedback)
                        
                        # Process each event from the generator
                        async for event in feedback_generator:
                            # Process each event
                            try:
                                # Handle non-dict events
                                if not isinstance(event, dict):
                                    serialized_event = {"raw_event": str(event)}
                                    logger.warning(f"Non-dict event in feedback stream: {type(event)}")
                                else:
                                    serialized_event = event
                                
                                # Add metadata
                                serialized_event['__meta'] = {
                                    'time': time.time(),
                                    'thread_id': report.thread_id  # Use report.thread_id instead of thread_id
                                }
                                
                                # Check for updated plan
                                if 'plan_updated' in serialized_event and serialized_event.get('sections'):
                                    # Update report with new plan from feedback
                                    sections = serialized_event.get('sections', [])
                                    if sections:
                                        logger.info(f"Updating report plan with {len(sections)} sections")
                                        report.plan = sections
                                        report.save()
                                        logger.info(f"Updated report plan with {len(sections)} sections from feedback")
                                
                                # Format for SSE
                                try:
                                    # Convert non-serializable objects to strings
                                    if not isinstance(event, (dict, list, str, int, float, bool, type(None))):
                                        serialized_event = {
                                            "type": "message",
                                            "content": str(event),
                                            "timestamp": time.time()
                                        }
                                        yield f"data: {json.dumps(serialized_event)}\n\n"
                                    else:
                                        # Handle dict with potentially non-serializable values
                                        if isinstance(event, dict):
                                            # Create a copy we can modify safely
                                            safe_event = {}
                                            for k, v in event.items():
                                                # Special handling for __interrupt__ which can contain Interrupt objects
                                                if k == '__interrupt__':
                                                    # Convert to simple list of dicts format that frontend expects
                                                    if hasattr(v, '__iter__') and not isinstance(v, (str, dict)):
                                                        interrupt_list = []
                                                        for item in v:
                                                            if hasattr(item, 'value'):
                                                                # Handle Interrupt objects with a value attribute
                                                                interrupt_list.append({"value": str(item.value)})
                                                            else:
                                                                # Handle other types
                                                                interrupt_list.append({"value": str(item)})
                                                        safe_event[k] = interrupt_list
                                                    else:
                                                        # Simple value
                                                        safe_event[k] = [{"value": str(v)}]
                                                elif isinstance(v, (dict, list, str, int, float, bool, type(None))):
                                                    safe_event[k] = v
                                                else:
                                                    # Convert non-serializable values to strings
                                                    safe_event[k] = str(v)
                                            yield f"data: {json.dumps(safe_event)}\n\n"
                                        else:
                                            # Should be directly serializable now
                                            yield f"data: {json.dumps(event)}\n\n"
                                except TypeError as json_err:
                                    logger.error(f"JSON serialization error: {json_err}, event type: {type(event)}")
                                    # Create a safe serializable version as fallback
                                    safe_event = {
                                        "type": "error",
                                        "message": f"Failed to serialize event: {str(json_err)}",
                                        "timestamp": time.time()
                                    }
                                    yield f"data: {json.dumps(safe_event)}\n\n"
                                
                                # Keepalive after important events
                                if isinstance(event, dict) and (
                                    'sections' in event or 
                                    'completed_sections' in event or
                                    'compile_final_report' in event
                                ):
                                    yield ":\n\n"  # Keepalive after important updates
                                    last_keepalive = time.time()
                            except Exception as e:
                                logger.error(f"Error processing feedback event: {e}")
                                # Return a simple error event
                                yield f"data: {json.dumps({'type': 'error', 'message': f'Error processing event: {str(e)}'})}\n\n"
                    
                    # Run the async function and collect all event data
                    all_events = loop.run_until_complete(collect_events())
                    
                    # Always save the thread state after processing
                    try:
                        # Save thread state back to Redis
                        logger.info(f"Saving thread state to Redis after processing for thread_id: {report.thread_id}")
                        save_thread_state(report.thread_id, thread)
                    except Exception as e:
                        logger.error(f"Error saving thread state: {e}", exc_info=True)
                    
                    # Yield each event to the SSE stream
                    for event_data in all_events:
                        yield event_data
                        
                except Exception as e:
                    logger.error(f"Error in feedback stream: {str(e)}", exc_info=True)
                    error_data = {
                        'type': 'error',
                        'message': f'Error processing feedback: {str(e)}'
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                
                # Send a final message before closing
                yield f"data: {json.dumps({'type': 'stream_complete'})}\n\n"
            
            response = StreamingHttpResponse(
                feedback_stream(),
                content_type='text/event-stream'
            )
            
            # Add headers needed for SSE
            response['Cache-Control'] = 'no-cache'
            response['X-Accel-Buffering'] = 'no'
            response['Connection'] = 'keep-alive'
            
            # Add CORS headers if needed
            if 'HTTP_ORIGIN' in request.META:
                response['Access-Control-Allow-Origin'] = request.META['HTTP_ORIGIN']
                response['Access-Control-Allow-Credentials'] = 'true'
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
            return Response(
                {"error": f"Failed to process feedback: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ApproveResearchPlanView(APIView):
    """API view to approve a research plan."""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    renderer_classes = [EventSourceRenderer, renderers.JSONRenderer]
    
    def get_permissions(self):
        """
        Override to allow token auth via query parameter for EventSource connections
        which can't send Authorization headers
        """
        if self.request.method == 'POST' and 'token' in self.request.GET:
            # Authenticate with token immediately, before permission checks
            user = self.perform_token_authentication(self.request)
            if user:
                # Successfully authenticated with token
                return []
        return super().get_permissions()
    
    def perform_token_authentication(self, request):
        """
        Authenticate using token from query parameter
        """
        token = request.GET.get('token')
        if not token:
            return None
            
        try:
            from rest_framework_simplejwt.tokens import AccessToken
            from django.contrib.auth import get_user_model
            
            # Decode token
            validated_token = AccessToken(token)
            user_id = validated_token.get('user_id')
            
            # Get user
            User = get_user_model()
            user = User.objects.get(id=user_id)
            
            # Don't try to set is_authenticated as it's a property
            # Instead, just set the user on the request
            request.user = user
            # Add _authentication_backend to user so DRF knows user is authenticated
            setattr(user, 'backend', 'django.contrib.auth.backends.ModelBackend')
            
            logger.debug(f"Token authentication successful for user {user.username} (id: {user_id})")
            return user
        except Exception as e:
            logger.error(f"Token auth error: {str(e)}", exc_info=True)
            return None
    
    def post(self, request, thread_id):
        """Approve a research plan."""
        try:
            # Handle token in query parameter for EventSource
            if 'token' in request.GET:
                # Always try token auth if token is present, even if user is already authenticated
                user = self.perform_token_authentication(request)
                if not user:
                    logger.error("Token authentication failed")
                    return Response(
                        {"error": "Invalid token"},
                        status=status.HTTP_401_UNAUTHORIZED
                    )
                logger.debug(f"User authenticated from token: {user.username} (ID: {user.id})")
            elif not request.user.is_authenticated:
                # No token and not authenticated
                return Response(
                    {"error": "Authentication required"},
                    status=status.HTTP_401_UNAUTHORIZED
                )
                    
            # Verify the thread belongs to the current user
            report = get_object_or_404(
                ResearchReport, 
                thread_id=thread_id,
                user=request.user
            )
            
            # Ensure report is in planning status
            if report.status != 'PLANNING':
                return Response(
                    {"error": "Approval can only be done in planning stage"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get thread state from Redis
            thread = get_thread_state(thread_id)
            if not thread:
                logger.warning(f"Thread state missing for report {report.id} (thread_id: {report.thread_id})")
                
                # If a report exists but thread state is missing, attempt to reconstruct minimal thread state
                if report.status == 'PLANNING' and report.plan:
                    logger.info(f"Creating fallback thread state for report {report.id}")
                    # Create minimal thread state with just the thread_id
                    thread = {
                        "configurable": {
                            "thread_id": report.thread_id,
                            "topic": report.topic
                        }
                    }
                    # Save this fallback thread state to Redis
                    save_thread_state(report.thread_id, thread)
                else:
                    return Response(
                        {"error": "Research session expired or not found"},
                        status=status.HTTP_404_NOT_FOUND
                    )
            
            # Ensure thread has the topic in configurable
            if "configurable" in thread and "topic" not in thread["configurable"]:
                logger.info(f"Adding missing topic to thread configurable: {report.topic}")
                thread["configurable"]["topic"] = report.topic
                save_thread_state(report.thread_id, thread)
            
            # Update report status
            report.status = 'RESEARCHING'
            report.save()

            # Additional validation and logging for thread contents
            if 'configurable' in thread:
                logger.info(f"Thread configurable before approval: {thread['configurable']}")
                
                # Ensure topic is in configurable
                if 'topic' not in thread['configurable']:
                    logger.warning(f"Topic missing from thread configurable, adding it now: {report.topic}")
                    thread['configurable']['topic'] = report.topic
                
                # Log the final state
                logger.info(f"Final thread configurable: {thread['configurable']}")
                save_thread_state(report.thread_id, thread)
            else:
                logger.error(f"Thread is missing 'configurable' field: {thread}")
                # Create configurable if it doesn't exist
                thread['configurable'] = {
                    "thread_id": report.thread_id,
                    "topic": report.topic,
                    "planner_model": "gpt-4o",
                    "writer_model": "gpt-4o"
                }
                logger.info(f"Created configurable for thread: {thread['configurable']}")
                save_thread_state(report.thread_id, thread)
            
            # Additional validation and logging for thread contents
            logger.info(f"Ensuring thread has all required properties before approval")
            thread = ensure_thread_has_required_props(thread, report.topic, report.thread_id)
            logger.info(f"Final thread configurable: {thread['configurable']}")
            save_thread_state(report.thread_id, thread)
            
            # Return streaming response
            def approve_stream():
                # Send an initial keep-alive comment to establish the connection
                yield ":\n\n"
                
                # Send connection established event
                event_data = {
                    'type': 'connection_established',
                    'message': 'Connected to research service'
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Initialize research manager with report config
                manager = ResearchManager(report.config)
                
                try:
                    # Create a new event loop for this stream
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Since approve_plan returns an async generator, we need to iterate through it
                    async def process_events():
                        # Get the async generator directly
                        approval_generator = manager.approve_plan(thread)
                        
                        # Process each event from the generator
                        async for event in approval_generator:
                            # Process each event
                            try:
                                # Handle non-dict events
                                if not isinstance(event, dict):
                                    serialized_event = {"raw_event": str(event)}
                                    logger.warning(f"Non-dict event in approval stream: {type(event)}")
                                else:
                                    serialized_event = event
                                
                                # Add additional debug info
                                logger.info(f"Processing approval event type: {type(event)}, keys: {event.keys() if isinstance(event, dict) else 'N/A'}")
                                
                                # Add metadata
                                serialized_event['__meta'] = {
                                    'time': time.time(),
                                    'thread_id': report.thread_id  # Use report.thread_id instead of thread_id
                                }
                                
                                # Check for completion
                                if 'compile_final_report' in serialized_event:
                                    # Update report
                                    if 'final_report' in serialized_event['compile_final_report']:
                                        logger.info("Found final report content, updating report")
                                        report.content = serialized_event['compile_final_report']['final_report']
                                        report.status = 'COMPLETED'
                                        report.completed_at = timezone.now()
                                        report.save()
                                
                                # For section building phase
                                if 'build_section_with_web_research' in serialized_event:
                                    logger.info("Section building phase detected")
                                    # Make sure we're in RESEARCHING state
                                    report.status = 'RESEARCHING'
                                    report.save()
                                
                                # Format for SSE
                                try:
                                    sse_data = f"data: {json.dumps(serialized_event)}\n\n"
                                    logger.debug(f"Successfully formatted SSE data")
                                    yield sse_data
                                except TypeError as e:
                                    # Handle JSON serialization errors
                                    logger.error(f"JSON serialization error: {e}, event: {type(serialized_event)}")
                                    # Create a safe version without problematic fields
                                    safe_event = {"type": "event", "message": "Event received but could not be fully serialized"}
                                    for key, value in serialized_event.items():
                                        try:
                                            # Test if this key/value can be serialized
                                            json.dumps({key: value})
                                            safe_event[key] = value
                                        except:
                                            safe_event[f"{key}_error"] = f"Could not serialize value of type {type(value)}"
                                    
                                    yield f"data: {json.dumps(safe_event)}\n\n"
                                
                                # Keepalive after important events
                                if isinstance(event, dict) and (
                                    'sections' in event or 
                                    'completed_sections' in event or
                                    'compile_final_report' in event
                                ):
                                    yield ":\n\n"  # Keepalive after important updates
                                    last_keepalive = time.time()
                            except Exception as e:
                                logger.error(f"Error processing approval event: {e}", exc_info=True)
                                # Return a simple error event
                                yield f"data: {json.dumps({'type': 'error', 'message': f'Error processing event: {str(e)}'})}\n\n"
                    
                    # Run the async function and collect all event data
                    all_events = loop.run_until_complete(collect_events())
                    
                    # Always save the thread state after processing
                    try:
                        # Save thread state back to Redis
                        logger.info(f"Saving thread state to Redis after processing for thread_id: {report.thread_id}")
                        save_thread_state(report.thread_id, thread)
                    except Exception as e:
                        logger.error(f"Error saving thread state: {e}", exc_info=True)
                    
                    # Yield each event to the SSE stream
                    for event_data in all_events:
                        yield event_data
                        
                except Exception as e:
                    logger.error(f"Error in approval stream: {str(e)}", exc_info=True)
                    error_data = {
                        'type': 'error',
                        'message': f'Error processing research: {str(e)}'
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    
                    # Update report status on error
                    report.status = 'FAILED'
                    report.save()
                
                # Send a final message before closing
                yield f"data: {json.dumps({'type': 'stream_complete'})}\n\n"

            return StreamingHttpResponse(
                approve_stream(),
                content_type='text/event-stream'
            )
            
        except Exception as e:
            logger.error(f"Error approving research plan: {str(e)}", exc_info=True)
            
            # Update report status on error
            try:
                report.status = 'FAILED'
                report.save()
            except:
                pass
                
            return Response(
                {"error": f"Failed to approve research plan: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ResearchModelTestView(APIView):
    """API view to test the research model configuration."""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def post(self, request):
        """Test the planner model and return results."""
        try:
            # Create a research manager
            manager = ResearchManager()
            
            # Get OpenAI API key status
            openai_valid, openai_msg, openai_details = verify_openai_api_key()
            
            # Get test results
            results = manager.test_planner()
            
            return Response({
                "status": "success",
                "message": "Model test completed", 
                "results": results,
                "openai_status": {
                    "valid": openai_valid,
                    "message": openai_msg,
                    "details": openai_details
                }
            })
            
        except Exception as e:
            logger.error(f"Error testing models: {str(e)}", exc_info=True)
            return Response(
                {"error": f"Failed to test models: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class SSEHealthCheckView(APIView):
    """API view to test SSE connections and verify the server is working properly."""
    
    permission_classes = [AllowAny]
    renderer_classes = [EventSourceRenderer, renderers.JSONRenderer]
    
    def get(self, request):
        """Test the SSE connection with a simple event stream."""
        try:
            # Get test parameters
            test_duration = int(request.GET.get('duration', 30))  # Default 30 second test
            keepalive_interval = int(request.GET.get('keepalive', 5))  # Default 5 second keepalives
            event_count = int(request.GET.get('events', 10))  # Default 10 events
            
            # Cap to reasonable limits
            test_duration = min(max(test_duration, 5), 120)  # Between 5-120 seconds
            keepalive_interval = min(max(keepalive_interval, 1), 15)  # Between 1-15 seconds
            event_count = min(max(event_count, 3), 50)  # Between 3-50 events
            
            # Calculate delay between events to distribute over the test duration
            event_delay = test_duration / event_count
            
            # Create a new event loop for streaming
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Define an async generator for testing SSE
            async def sse_test_generator():
                start_time = time.time()
                
                # Yield events with calculated delay
                for i in range(event_count):
                    elapsed = time.time() - start_time
                    remaining = test_duration - elapsed
                    
                    # Calculate time to sleep (ensure we don't exceed test duration)
                    sleep_time = min(event_delay, remaining / (event_count - i)) if i < event_count - 1 else 0
                    
                    # Add a delay to simulate real events
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                    
                    # Send diagnostic data
                    yield {
                        'type': 'test_event',
                        'counter': i + 1,
                        'total': event_count,
                        'elapsed_seconds': round(time.time() - start_time, 2),
                        'timestamp': time.time(),
                        'test_config': {
                            'duration': test_duration,
                            'keepalive': keepalive_interval,
                            'event_count': event_count
                        }
                    }
                
                # Final completion event
                yield {
                    'type': 'test_complete',
                    'message': 'SSE connection test completed successfully',
                    'total_time': round(time.time() - start_time, 2),
                    'events_sent': event_count
                }
            
            # Create the metadata
            metadata = {
                'test_id': str(uuid.uuid4()),
                'timestamp': time.time(),
                'client_ip': request.META.get('REMOTE_ADDR', 'unknown'),
                'client_agent': request.META.get('HTTP_USER_AGENT', 'unknown')
            }
            
            logger.info(f"Starting SSE health check: duration={test_duration}s, keepalive={keepalive_interval}s, events={event_count}")
            
            # Create the streamed response
            response = StreamingHttpResponse(
                loop.run_until_complete(collect_events(
                    handle_sse_stream(
                        sse_test_generator(),
                        metadata=metadata,
                        keepalive_interval=keepalive_interval
                    )
                )),
                content_type='text/event-stream'
            )
            
            # Set up the response with proper headers
            response = setup_sse_response(response)
            
            # Add CORS headers if needed
            if 'HTTP_ORIGIN' in request.META:
                response['Access-Control-Allow-Origin'] = request.META['HTTP_ORIGIN']
                response['Access-Control-Allow-Credentials'] = 'true'
            
            return response
            
        except Exception as e:
            logger.error(f"Error in SSE health check: {str(e)}", exc_info=True)
            return Response(
                {"error": f"Failed to run SSE health check: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )