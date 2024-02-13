
from django.contrib import admin
from django.urls import path, include

from face_emotion_project import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path("videoapp/",include("videoapp.urls")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
