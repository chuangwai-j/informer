from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/flight/tracking/(?P<flight_id>\w+)/$', consumers.FlightTrackingConsumer.as_asgi()),
    re_path(r'ws/flights/all/$', consumers.AllFlightsConsumer.as_asgi()),
    re_path(r'ws/prediction/updates/$', consumers.PredictionUpdateConsumer.as_asgi()),
]