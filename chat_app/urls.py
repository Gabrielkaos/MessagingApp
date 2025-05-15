
from django.urls import path

from . import views


app_name = 'chat_app'
urlpatterns = [
    path("",views.index, name="index"),
    path("response",views.response, name="response"),
    path("train",views.train, name="train"),
    path("add",views.add, name="add")
]
