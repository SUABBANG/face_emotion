from django.urls import path

from videoapp.views import video_emotion

urlpatterns = [
    path('video_emotion/', video_emotion.as_view(), name='video_emotion'),
]