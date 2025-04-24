from django.shortcuts import render
from django.http import JsonResponse

# Create your views here.


def index(request):
    return render(request,"chat_app/chatbox.html")


def response(request):
    # return render(request,"chat_app/chatbox.html",)
    if request.method=="POST":
        return JsonResponse({'response':'hi'})
    return JsonResponse({'error':'Invalid method'})