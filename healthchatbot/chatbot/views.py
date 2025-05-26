# chatbot/views.py - Version cơ bản để fix lỗi
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from django.shortcuts import render
from django.utils import timezone
import uuid
import logging

# Import models
from .models import Disease, Symptom, ChatSession, ChatMessage, URLSource

# Import serializers
from .serializers import DiseaseSerializer, SymptomSerializer, ChatSessionSerializer, ChatMessageSerializer

# Import processors (with error handling)
try:
    from .nlp_processor import ImprovedNLPProcessor
except ImportError:
    # Fallback to original processor if new one not available
    try:
        from .nlp_processor import NLPProcessor as ImprovedNLPProcessor
    except ImportError:
        ImprovedNLPProcessor = None

logger = logging.getLogger(__name__)

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
        try:
            if ImprovedNLPProcessor:
                self.nlp_processor = ImprovedNLPProcessor()
                logger.info("ImprovedNLPProcessor initialized successfully")
            else:
                self.nlp_processor = None
                logger.warning("NLP Processor not available")
        except Exception as e:
            logger.error(f"Error initializing NLP Processor: {e}")
            self.nlp_processor = None
        
    @action(detail=False, methods=['post'])
    def message(self, request):
        """Xử lý tin nhắn từ người dùng"""
        try:
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
            if self.nlp_processor:
                try:
                    response_text = self.nlp_processor.process_query(message)
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    response_text = "Xin lỗi, đã xảy ra lỗi khi xử lý tin nhắn của bạn. Vui lòng thử lại."
            else:
                response_text = "Hệ thống đang gặp sự cố. Vui lòng thử lại sau."
            
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
            
        except Exception as e:
            logger.error(f"Error in message endpoint: {e}")
            return Response({
                'error': 'Đã xảy ra lỗi hệ thống'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['post'])
    def update_knowledge(self, request):
        """Cập nhật knowledge base từ URL"""
        try:
            urls = request.data.get('urls', [])
            
            if not urls:
                return Response({
                    'success': False,
                    'message': 'Vui lòng cung cấp ít nhất một URL'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Kiểm tra và làm sạch URLs
            clean_urls = []
            for url in urls:
                url = url.strip()
                if url and url.startswith(('http://', 'https://')):
                    clean_urls.append(url)
                else:
                    logger.warning(f"Invalid URL format: {url}")
            
            if not clean_urls:
                return Response({
                    'success': False,
                    'message': 'Không có URL hợp lệ nào được cung cấp'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Lưu URL vào database nếu chưa có
            for url in clean_urls:
                URLSource.objects.get_or_create(url=url)
            
            # Cập nhật knowledge base
            if self.nlp_processor:
                try:
                    total_diseases = self.nlp_processor.fetch_and_update_knowledge_base(clean_urls)
                except Exception as e:
                    logger.error(f"Error updating knowledge base: {e}")
                    total_diseases = 0
            else:
                return Response({
                    'success': False,
                    'message': 'NLP Processor chưa được khởi tạo'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Cập nhật thông tin URLs
            for url in clean_urls:
                try:
                    source = URLSource.objects.get(url=url)
                    source.last_updated = timezone.now()
                    source.save()
                except URLSource.DoesNotExist:
                    logger.warning(f"URL source not found: {url}")
            
            message = f'Đã cập nhật thành công knowledge base với {total_diseases} bệnh từ {len(clean_urls)} URL'
            logger.info(message)
            
            return Response({
                'success': True,
                'message': message,
                'imported_diseases': total_diseases,
                'processed_urls': len(clean_urls)
            })
            
        except Exception as e:
            error_message = f'Lỗi khi cập nhật knowledge base: {str(e)}'
            logger.error(error_message)
            return Response({
                'success': False,
                'message': error_message
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Lấy thống kê về knowledge base"""
        try:
            stats = {
                'total_diseases': Disease.objects.count(),
                'total_symptoms': Symptom.objects.count(),
                'total_sessions': ChatSession.objects.count(),
                'total_messages': ChatMessage.objects.count(),
                'total_sources': URLSource.objects.count(),
                'active_sources': URLSource.objects.filter(active=True).count(),
            }
            
            return Response(stats)
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return Response({
                'error': 'Lỗi khi lấy thống kê'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ViewSet cho Source management
class SourceViewSet(viewsets.ViewSet):
    def list(self, request):
        """Lấy danh sách nguồn dữ liệu"""
        try:
            sources = URLSource.objects.all().order_by('-last_updated')
            data = [{
                'url': source.url,
                'last_updated': source.last_updated,
                'success_count': source.success_count,
                'active': source.active
            } for source in sources]
            
            return Response(data)
            
        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            return Response({
                'error': 'Lỗi khi lấy danh sách nguồn'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['post'])
    def delete(self, request):
        """Xóa một nguồn dữ liệu"""
        try:
            url = request.data.get('url')
            
            if not url:
                return Response({
                    'success': False,
                    'message': 'URL is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            source = URLSource.objects.get(url=url)
            source.delete()
            
            logger.info(f"Deleted source: {url}")
            
            return Response({
                'success': True,
                'message': f'Successfully deleted source: {url}'
            })
            
        except URLSource.DoesNotExist:
            return Response({
                'success': False,
                'message': f'Source not found: {url}'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error deleting source: {e}")
            return Response({
                'success': False,
                'message': f'Error deleting source: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# View cho Template
def chatbot_view(request):
    """Render trang chatbot"""
    return render(request, 'chatbot/chatbot.html')

# Test endpoints (optional - có thể thêm sau)
@api_view(['POST'])
def test_extraction(request):
    """Test endpoint để debug quá trình extraction từ URL"""
    try:
        url = request.data.get('url')
        
        if not url:
            return Response({
                'success': False,
                'error': 'URL is required'
            }, status=400)
        
        # Basic validation
        if not url.startswith(('http://', 'https://')):
            return Response({
                'success': False,
                'error': 'Invalid URL format'
            }, status=400)
        
        # Simple test implementation
        import requests
        from bs4 import BeautifulSoup
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Basic analysis
        page_analysis = {
            'title': soup.title.text.strip() if soup.title else 'No title',
            'content_length': len(response.text),
            'paragraphs': len(soup.find_all('p')),
            'headers': {
                'h1': len(soup.find_all('h1')),
                'h2': len(soup.find_all('h2')),
                'h3': len(soup.find_all('h3'))
            }
        }
        
        return Response({
            'success': True,
            'url': url,
            'page_analysis': page_analysis
        })
        
    except Exception as e:
        logger.error(f'Error in test extraction: {e}')
        return Response({
            'success': False,
            'error': f'Error: {str(e)}'
        }, status=500)