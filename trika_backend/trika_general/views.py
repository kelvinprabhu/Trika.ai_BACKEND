from django.shortcuts import render

# Create your views here.


def index(request):
    from django.http import JsonResponse
    return JsonResponse({"message": "trika_general is working"})
