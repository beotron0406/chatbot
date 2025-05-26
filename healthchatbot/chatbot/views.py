# chatbot/views.py
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import render
from django.utils import timezone
import uuid
from .models import URLSource
from .models import Disease, Symptom, ChatSession, ChatMessage
from .serializers import DiseaseSerializer, SymptomSerializer, ChatSessionSerializer, ChatMessageSerializer
from .nlp_processor import NLPProcessor
from django.utils import timezone

# ViewSet cho Disease
class DiseaseViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Disease.objects.all()
    serializer_class = DiseaseSerializer
    
    def get_queryset(self):
        queryset = Disease.objects.all()
        name = self.request.query_params.get('name')
        if name:
            queryset = queryset.filter(name__icontains=name)
        return queryset

# ViewSet cho Symptom
class SymptomViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Symptom.objects.all()
    serializer_class = SymptomSerializer
    
    def get_queryset(self):
        queryset = Symptom.objects.all()
        name = self.request.query_params.get('name')
        if name:
            queryset = queryset.filter(name__icontains=name)
        return queryset

# ViewSet cho Chatbot
class ChatbotViewSet(viewsets.ViewSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nlp_processor = NLPProcessor()
        
    @action(detail=False, methods=['post'])
    def message(self, request):
        session_id = request.data.get('session_id')
        message = request.data.get('message')
        
        if not message:
            return Response({"error": "No message provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Tạo hoặc lấy session
        if not session_id:
            session_id = str(uuid.uuid4())
            session = ChatSession.objects.create(session_id=session_id)
        else:
            try:
                session = ChatSession.objects.get(session_id=session_id)
            except ChatSession.DoesNotExist:
                session = ChatSession.objects.create(session_id=session_id)
        
        # Lưu tin nhắn người dùng
        user_message = ChatMessage.objects.create(
            session=session,
            sender='user',
            message=message
        )
        
        # Xử lý tin nhắn và tạo phản hồi
        response_text = self.nlp_processor.process_query(message)
        
        # Lưu tin nhắn bot
        bot_message = ChatMessage.objects.create(
            session=session,
            sender='bot',
            message=response_text
        )
        
        return Response({
            'session_id': session_id,
            'response': response_text
        })
    @action(detail=False, methods=['post'])
    def update_knowledge(self, request):
        """Cập nhật knowledge base từ URL"""
        urls = request.data.get('urls', [])
        
        try:
            # Lưu URL vào database nếu chưa có
            for url in urls:
                URLSource.objects.get_or_create(url=url)
            
            # Cập nhật knowledge base
            total_diseases = self.nlp_processor.fetch_and_update_knowledge_base(urls)
            
            # Cập nhật thông tin URLs
            for url in urls:
                source = URLSource.objects.get(url=url)
                source.last_updated = timezone.now()
                source.save()
            
            return Response({
                'success': True,
                'message': f'Successfully updated knowledge base with {total_diseases} diseases'
            })
        except Exception as e:
            return Response({
                'success': False,
                'message': f'Error updating knowledge base: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
   
# View cho Template
def chatbot_view(request):
    return render(request, 'chatbot/chatbot.html')
class SourceViewSet(viewsets.ViewSet):
    def list(self, request):
        """Lấy danh sách nguồn dữ liệu"""
        sources = URLSource.objects.all().order_by('-last_updated')
        data = [{
            'url': source.url,
            'last_updated': source.last_updated,
            'success_count': source.success_count,
            'active': source.active
        } for source in sources]
        
        return Response(data)
    
    @action(detail=False, methods=['post'])
    def delete(self, request):
        """Xóa một nguồn dữ liệu"""
        url = request.data.get('url')
        
        if not url:
            return Response({
                'success': False,
                'message': 'URL is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            source = URLSource.objects.get(url=url)
            source.delete()
            
            return Response({
                'success': True,
                'message': f'Successfully deleted source: {url}'
            })
        except URLSource.DoesNotExist:
            return Response({
                'success': False,
                'message': f'Source not found: {url}'
            }, status=status.HTTP_404_NOT_FOUND)