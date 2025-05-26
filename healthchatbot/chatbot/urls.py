# chatbot/urls.py - Version đơn giản để fix lỗi
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()

# Chỉ đăng ký các ViewSets cơ bản
try:
    router.register(r'diseases', views.DiseaseViewSet)
    router.register(r'symptoms', views.SymptomViewSet)
    router.register(r'chatbot', views.ChatbotViewSet, basename='chatbot')
    router.register(r'sources', views.SourceViewSet, basename='sources')
except AttributeError as e:
    print(f"Warning: Some ViewSets not available: {e}")

urlpatterns = [
    path('api/', include(router.urls)),
    path('', views.chatbot_view, name='chatbot'),
    
    # Test endpoint (optional)
    path('api/test-extraction/', views.test_extraction, name='test_extraction'),
]