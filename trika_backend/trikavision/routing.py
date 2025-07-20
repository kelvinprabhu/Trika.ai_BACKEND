from django.urls import re_path
from . import consumer
from . import visonConsumers
from django.urls import path
from trikavision.consumers.posture_consumer import PostureConsumer
websocket_urlpatterns = [
    re_path(r"ws/test/$", consumer.TestConsumer.as_asgi()),
    re_path(r"ws/posture-demo/$", visonConsumers.PostureConsumer.as_asgi()),
    re_path(r"ws/posture/$", PostureConsumer.as_asgi()),
]

