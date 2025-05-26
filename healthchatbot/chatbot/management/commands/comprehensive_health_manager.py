# chatbot/management/commands/comprehensive_health_manager.py
# Version đơn giản để test và debug

from django.core.management.base import BaseCommand
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Healthcare knowledge base management (basic version)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--action',
            type=str,
            choices=['update-all', 'stats', 'test', 'debug'],
            help='Action to perform',
            required=True
        )
        
        parser.add_argument(
            '--urls',
            nargs='+',
            type=str,
            help='List of URLs for testing'
        )

    def handle(self, *args, **kwargs):
        action = kwargs['action']
        urls = kwargs.get('urls', [])
        
        self.stdout.write(f'🚀 Executing action: {action}')
        
        try:
            if action == 'debug':
                self.debug_system()
            elif action == 'stats':
                self.show_statistics()
            elif action == 'test':
                self.test_basic_functionality(urls)
            elif action == 'update-all':
                self.update_knowledge_base(urls)
            else:
                self.stdout.write(self.style.ERROR(f'Unknown action: {action}'))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error: {str(e)}'))
            logger.error(f'Management command error: {e}', exc_info=True)
    
    def debug_system(self):
        """Debug hệ thống để tìm lỗi"""
        self.stdout.write('🔍 Debugging system...')
        
        try:
            # Test import models
            from chatbot.models import Disease, Symptom, URLSource, ChatSession
            self.stdout.write('✅ Models import successfully')
            
            # Test import views
            from chatbot import views
            self.stdout.write('✅ Views import successfully')
            
            # Check ViewSets
            viewsets = ['DiseaseViewSet', 'SymptomViewSet', 'ChatbotViewSet', 'SourceViewSet']
            for viewset in viewsets:
                if hasattr(views, viewset):
                    self.stdout.write(f'✅ {viewset} exists')
                else:
                    self.stdout.write(f'❌ {viewset} missing')
            
            # Test import processors
            try:
                from chatbot.nlp_processor import ImprovedNLPProcessor
                self.stdout.write('✅ ImprovedNLPProcessor import successfully')
            except ImportError as e:
                self.stdout.write(f'⚠️ ImprovedNLPProcessor import failed: {e}')
                
                # Try original processor
                try:
                    from chatbot.nlp_processor import NLPProcessor
                    self.stdout.write('✅ Original NLPProcessor available')
                except ImportError as e2:
                    self.stdout.write(f'❌ No NLP processor available: {e2}')
            
            # Test database
            disease_count = Disease.objects.count()
            symptom_count = Symptom.objects.count()
            source_count = URLSource.objects.count()
            
            self.stdout.write(f'📊 Database status:')
            self.stdout.write(f'  - Diseases: {disease_count}')
            self.stdout.write(f'  - Symptoms: {symptom_count}')
            self.stdout.write(f'  - Sources: {source_count}')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Debug failed: {str(e)}'))
    
    def show_statistics(self):
        """Hiển thị thống kê cơ bản"""
        self.stdout.write('📊 System Statistics:')
        
        try:
            from chatbot.models import Disease, Symptom, URLSource, ChatSession, ChatMessage
            
            stats = {
                'diseases': Disease.objects.count(),
                'symptoms': Symptom.objects.count(),
                'sources': URLSource.objects.count(),
                'sessions': ChatSession.objects.count(),
                'messages': ChatMessage.objects.count(),
            }
            
            for key, value in stats.items():
                self.stdout.write(f'  {key.capitalize()}: {value}')
                
            # Active sources
            active_sources = URLSource.objects.filter(active=True).count()
            self.stdout.write(f'  Active Sources: {active_sources}')
            
            self.stdout.write('✅ Statistics retrieved successfully')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error getting statistics: {str(e)}'))
    
    def test_basic_functionality(self, urls):
        """Test chức năng cơ bản"""
        self.stdout.write('🧪 Testing basic functionality...')
        
        if not urls:
            urls = ['https://vnvc.vn/cac-benh-truyen-nhiem-thuong-gap/']
        
        for url in urls:
            self.stdout.write(f'📡 Testing URL: {url}')
            
            try:
                import requests
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    self.stdout.write(f'  ✅ URL accessible (status: {response.status_code})')
                    self.stdout.write(f'  📄 Content length: {len(response.text)} characters')
                    
                    # Basic content check
                    content = response.text.lower()
                    health_keywords = ['bệnh', 'triệu chứng', 'điều trị', 'y tế']
                    found_keywords = [kw for kw in health_keywords if kw in content]
                    
                    if found_keywords:
                        self.stdout.write(f'  🏥 Health keywords found: {", ".join(found_keywords)}')
                    else:
                        self.stdout.write(f'  ⚠️ No health keywords found')
                        
                else:
                    self.stdout.write(f'  ❌ URL not accessible (status: {response.status_code})')
                    
            except Exception as e:
                self.stdout.write(f'  ❌ Error accessing URL: {str(e)}')
    
    def update_knowledge_base(self, urls):
        """Cập nhật knowledge base (version cơ bản)"""
        self.stdout.write('🔄 Updating knowledge base...')
        
        try:
            # Try to import and use processor
            from chatbot.nlp_processor import ImprovedNLPProcessor
            processor = ImprovedNLPProcessor()
            self.stdout.write('✅ Using ImprovedNLPProcessor')
            
        except ImportError:
            try:
                from chatbot.nlp_processor import NLPProcessor
                processor = NLPProcessor()
                self.stdout.write('⚠️ Using original NLPProcessor')
            except ImportError:
                self.stdout.write('❌ No NLP processor available')
                return
        
        if not urls:
            urls = [
                'https://vnvc.vn/cac-benh-truyen-nhiem-thuong-gap/',
                'https://hellobacsi.com/benh-truyen-nhiem/'
            ]
        
        try:
            total_diseases = processor.fetch_and_update_knowledge_base(urls)
            self.stdout.write(
                self.style.SUCCESS(f'✅ Knowledge base updated: {total_diseases} diseases')
            )
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Update failed: {str(e)}'))
            
            # Try manual update
            self.stdout.write('🔄 Attempting manual update...')
            self.manual_update(urls)
    
    def manual_update(self, urls):
        """Update thủ công để debug"""
        for url in urls:
            self.stdout.write(f'📡 Manually processing: {url}')
            
            try:
                import requests
                from bs4 import BeautifulSoup
                
                response = requests.get(url, timeout=30)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Basic extraction
                title = soup.title.text if soup.title else 'No title'
                paragraphs = soup.find_all('p')
                
                self.stdout.write(f'  📄 Title: {title}')
                self.stdout.write(f'  📝 Paragraphs found: {len(paragraphs)}')
                
                # Look for health content
                health_content = []
                for p in paragraphs[:10]:  # Check first 10 paragraphs
                    text = p.get_text().strip()
                    if any(word in text.lower() for word in ['bệnh', 'triệu chứng', 'điều trị']):
                        health_content.append(text[:100] + '...')
                
                if health_content:
                    self.stdout.write(f'  🏥 Health content samples:')
                    for content in health_content[:3]:
                        self.stdout.write(f'    - {content}')
                else:
                    self.stdout.write(f'  ⚠️ No obvious health content found')
                    
            except Exception as e:
                self.stdout.write(f'  ❌ Manual processing failed: {str(e)}')