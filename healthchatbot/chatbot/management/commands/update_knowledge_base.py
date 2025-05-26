from django.core.management.base import BaseCommand
from chatbot.nlp_processor import NLPProcessor

class Command(BaseCommand):
    help = 'Update knowledge base from URLs'

    def add_arguments(self, parser):
        parser.add_argument('--urls', nargs='+', type=str, help='List of URLs to fetch data from')

    def handle(self, *args, **kwargs):
        urls = kwargs.get('urls')
        
        self.stdout.write(f'Starting knowledge base update...')
        
        processor = NLPProcessor()
        total_diseases = processor.fetch_and_update_knowledge_base(urls)
        
        self.stdout.write(self.style.SUCCESS(f'Successfully updated knowledge base with {total_diseases} diseases'))