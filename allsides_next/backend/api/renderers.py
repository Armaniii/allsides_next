from rest_framework.renderers import BaseRenderer

# class EventStreamRenderer(BaseRenderer):
#     """Custom renderer for Server-Sent Events (SSE)."""
#     media_type = 'text/event-stream'
#     format = 'eventstream'
#     charset = None  # SSE responses are plain bytes

#     def render(self, data, accepted_media_type=None, renderer_context=None):
#         """
#         Return empty string as the actual streaming is handled by StreamingHttpResponse.
#         This renderer only exists to satisfy DRF's content negotiation.
#         """
#         return '' 