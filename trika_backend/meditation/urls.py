from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='meditation-index'),
    path('generate/', views.generate_meditation_session, name='generate-meditation-session'),
]
