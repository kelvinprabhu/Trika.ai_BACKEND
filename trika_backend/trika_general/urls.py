from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='trika_general-index'),
]
