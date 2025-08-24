from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='trikabot-index'),
    path('chat', views.Chat, name='trikabot-chat'),
    path('generate-challenge/', views.generate_challenge, name="generate_challenge"),
]
 