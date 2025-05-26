import requests
from bs4 import BeautifulSoup
import re
from django.core.management.base import BaseCommand
from chatbot.models import Disease, Symptom, DiseaseSymptom, Complication, Treatment, Prevention, Vaccine

class Command(BaseCommand):
    help = 'Import disease data from URL'

    def add_arguments(self, parser):
        parser.add_argument('url', type=str, help='URL to fetch data from')

    def handle(self, *args, **kwargs):
        url = kwargs['url']
        
        try:
            # Lấy nội dung từ URL
            self.stdout.write(f'Fetching data from {url}...')
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Kiểm tra lỗi
            
            # Xử lý nội dung tùy thuộc vào định dạng
            if url.endswith('.html') or 'html' in url:
                content = self.extract_from_html(response.text)
            else:
                content = response.text
            
            # Tiến hành xử lý dữ liệu và nhập vào database
            diseases_data = self.extract_disease_info(content)
            self.import_data_to_db(diseases_data)
            
            self.stdout.write(self.style.SUCCESS(f'Successfully imported {len(diseases_data)} diseases from {url}'))
            
        except requests.RequestException as e:
            self.stdout.write(self.style.ERROR(f'Error fetching URL: {e}'))
    
    def extract_from_html(self, html_content):
        # Sử dụng BeautifulSoup để trích xuất nội dung từ HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Xóa các script và style để tránh nhiễu
        for script in soup(["script", "style"]):
            script.extract()
        
        # Lấy text từ trang web
        text = soup.get_text()
        
        # Xử lý text (loại bỏ khoảng trắng thừa, dòng trống)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_disease_info(self, text):
        diseases = []
        
        # Tìm phần tài liệu liệt kê các bệnh
        disease_section = re.search(r'Các bệnh truyền nhiễm thường gặp(.*?)Làm sao để phòng ngừa', text, re.DOTALL)
        
        if not disease_section:
            # Thử tìm với cách khác nếu không tìm thấy với pattern trên
            disease_section = re.search(r'([\d]+\.\s+[\w\s]+)\n', text, re.DOTALL)
        
        if disease_section:
            disease_text = disease_section.group(1)
            
            # Tìm các phần bệnh riêng lẻ - cải thiện regex để phù hợp với format
            disease_patterns = re.finditer(r'(\d+)\.\s+(.*?)(?=\d+\.\s+|\Z)', disease_text, re.DOTALL)
            
            for pattern in disease_patterns:
                number = pattern.group(1)
                content = pattern.group(2).strip()
                
                # Tách tên và mô tả bệnh
                name_match = re.match(r'([^\n]+)', content)
                name = name_match.group(1) if name_match else f"Bệnh {number}"
                
                # Trích xuất các thông tin khác
                description = ""
                causes = ""
                symptoms = []
                complications = []
                treatments = []
                preventions = []
                vaccines = []
                is_contagious = False
                
                # Tìm mô tả
                desc_match = re.search(r'là.*?(?=\.|$)', content)
                if desc_match:
                    description = desc_match.group(0).strip()
                else:
                    # Thử cách khác nếu không tìm thấy với pattern trên
                    paragraphs = content.split('\n\n')
                    if len(paragraphs) > 1:
                        description = paragraphs[0].strip()
                
                # Tìm nguyên nhân
                cause_match = re.search(r'do (.*?) gây ra', content)
                if cause_match:
                    causes = cause_match.group(1).strip()
                
                # Tìm triệu chứng
                symptom_section = re.search(r'triệu chứng.*?:(.*?)(?=\.|$)', content, re.IGNORECASE | re.DOTALL)
                if symptom_section:
                    symptom_text = symptom_section.group(1)
                    symptom_list = re.findall(r'[-•]\s*(.*?)(?=[-•]|\Z)', symptom_text)
                    if not symptom_list:
                        symptom_list = [s.strip() for s in symptom_text.split(',')]
                    symptoms = [s.strip() for s in symptom_list if s.strip()]
                
                # Tìm biến chứng
                complication_section = re.search(r'biến chứng.*?:(.*?)(?=\.|$)', content, re.IGNORECASE | re.DOTALL)
                if complication_section:
                    complication_text = complication_section.group(1)
                    complication_list = re.findall(r'[-•]\s*(.*?)(?=[-•]|\Z)', complication_text)
                    if not complication_list:
                        complication_list = [c.strip() for c in complication_text.split(',')]
                    complications = [c.strip() for c in complication_list if c.strip()]
                
                # Tìm vắc-xin
                vaccine_match = re.search(r'vắc[- ]?xin.*?:(.*?)(?=\.|$)', content, re.IGNORECASE | re.DOTALL)
                if vaccine_match:
                    vaccine_text = vaccine_match.group(1)
                    vaccine_list = re.findall(r'[-•]\s*(.*?)(?=[-•]|\Z)', vaccine_text)
                    if not vaccine_list:
                        vaccine_list = [v.strip() for v in vaccine_text.split(',')]
                    vaccines = [v.strip() for v in vaccine_list if v.strip()]
                
                # Tìm phương pháp phòng ngừa
                prevention_section = re.search(r'phòng ngừa.*?:(.*?)(?=\.|$)', content, re.IGNORECASE | re.DOTALL)
                if prevention_section:
                    prevention_text = prevention_section.group(1)
                    prevention_list = re.findall(r'[-•]\s*(.*?)(?=[-•]|\Z)', prevention_text)
                    if not prevention_list:
                        prevention_list = [p.strip() for p in prevention_text.split(',')]
                    preventions = [p.strip() for p in prevention_list if p.strip()]
                
                # Xác định tính lây nhiễm
                if re.search(r'lây.*?(qua|bởi|từ)', content, re.IGNORECASE):
                    is_contagious = True
                
                diseases.append({
                    'name': name,
                    'description': description,
                    'causes': causes,
                    'symptoms': symptoms,
                    'complications': complications,
                    'treatments': treatments,
                    'preventions': preventions,
                    'vaccines': vaccines,
                    'is_contagious': is_contagious
                })
                
                self.stdout.write(f'Extracted information for: {name}')
        else:
            self.stdout.write(self.style.WARNING('Could not find disease section in the content'))
        
        return diseases
    
    def import_data_to_db(self, diseases_data):
        for data in diseases_data:
            # Tạo hoặc cập nhật bệnh
            disease, created = Disease.objects.update_or_create(
                name=data['name'],
                defaults={
                    'description': data['description'],
                    'causes': data['causes'],
                    'is_contagious': data['is_contagious']
                }
            )
            
            # Thêm triệu chứng
            for symptom_name in data['symptoms']:
                symptom, _ = Symptom.objects.get_or_create(name=symptom_name)
                DiseaseSymptom.objects.get_or_create(disease=disease, symptom=symptom)
            
            # Thêm biến chứng
            for complication_name in data['complications']:
                complication, _ = Complication.objects.get_or_create(name=complication_name)
                disease.complications.add(complication)
            
            # Thêm phương pháp phòng ngừa
            for prevention_method in data['preventions']:
                prevention, _ = Prevention.objects.get_or_create(method=prevention_method, 
                                                               defaults={'description': ''})
                disease.preventions.add(prevention)
            
            # Thêm vắc-xin
            for vaccine_name in data['vaccines']:
                vaccine, _ = Vaccine.objects.get_or_create(name=vaccine_name)
                disease.vaccines.add(vaccine)
            
            self.stdout.write(f"Imported {disease.name}")