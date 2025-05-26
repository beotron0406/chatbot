# chatbot/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'diseases', views.DiseaseViewSet)
router.register(r'symptoms', views.SymptomViewSet)
router.register(r'chatbot', views.ChatbotViewSet, basename='chatbot')
router.register(r'sources', views.SourceViewSet, basename='sources')
urlpatterns = [
    path('api/', include(router.urls)),
    path('', views.chatbot_view, name='chatbot'),
]