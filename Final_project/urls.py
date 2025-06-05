"""Final_project URL Configuration"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from speed_estimation import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('speed_estimation/video_feed/', views.video_feed, name='video_feed'),
    path('speed_estimation/get_stats/', views.get_stats, name='get_stats'),
    path('', include('user_app.urls')),
]

# Add static and media file serving for development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
