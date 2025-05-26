import re
import string
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode
import numpy as np
from django.utils import timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .models import Disease, Symptom, DiseaseSymptom, Complication, Prevention, Vaccine, URLSource

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedNLPProcessor:
    def __init__(self):
        self.symptom_vectorizer = None
        self.symptom_vectors = None
        self.symptoms = None
        self.init_symptoms()
        
        self.disease_vectorizer = None
        self.disease_vectors = None
        self.diseases = None
        self.init_diseases()
        
        # Danh sách từ khóa để nhận dạng bệnh
        self.disease_keywords = [
            'bệnh', 'viêm', 'nhiễm', 'sốt', 'đau', 'hội chứng', 'ung thư', 
            'tiểu đường', 'cao huyết áp', 'tim mạch', 'phổi', 'gan', 'thận',
            'dạ dày', 'ruột', 'da', 'mắt', 'tai', 'mũi', 'họng', 'cảm'
        ]
        
        # Từ khóa triệu chứng
        self.symptom_keywords = [
            'triệu chứng', 'biểu hiện', 'dấu hiệu', 'nhận biết', 'cảm giác',
            'đau', 'sốt', 'ho', 'chảy nước mũi', 'mệt mỏi', 'buồn nôn'
        ]
    
    def init_symptoms(self):
        """Khởi tạo vector cho triệu chứng"""
        try:
            self.symptoms = list(Symptom.objects.all())
            
            if not self.symptoms:
                logger.warning("No symptoms found in database")
                return
            
            symptom_texts = [self.preprocess_text(s.name + " " + (s.description or "")) 
                           for s in self.symptoms]
            
            self.symptom_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            self.symptom_vectors = self.symptom_vectorizer.fit_transform(symptom_texts)
            logger.info(f"Initialized {len(self.symptoms)} symptoms")
        except Exception as e:
            logger.error(f"Error initializing symptoms: {e}")
    
    def init_diseases(self):
        """Khởi tạo vector cho bệnh"""
        try:
            self.diseases = list(Disease.objects.all())
            
            if not self.diseases:
                logger.warning("No diseases found in database")
                return
            
            disease_texts = []
            for disease in self.diseases:
                text = disease.name + " " + disease.description
                
                # Thêm triệu chứng vào text
                for link in disease.symptoms_link.all():
                    text += " " + link.symptom.name
                
                disease_texts.append(self.preprocess_text(text))
            
            self.disease_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            self.disease_vectors = self.disease_vectorizer.fit_transform(disease_texts)
            logger.info(f"Initialized {len(self.diseases)} diseases")
        except Exception as e:
            logger.error(f"Error initializing diseases: {e}")
    
    def preprocess_text(self, text):
        """Tiền xử lý văn bản"""
        if not text:
            return ""
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Loại bỏ dấu tiếng Việt
        text = unidecode(text)
        
        # Loại bỏ số và dấu câu
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Loại bỏ khoảng trắng thừa
        text = ' '.join(text.split())
        
        return text
    
    def fetch_and_update_knowledge_base(self, urls=None):
        """Lấy dữ liệu từ URL và cập nhật knowledge base"""
        if urls is None:
            urls = [
                'https://vnvc.vn/cac-benh-truyen-nhiem-thuong-gap/',
                'https://www.vinmec.com/vi/benh/benh-truyen-nhiem-1/',
                'https://nhathuoclongchau.com.vn/bai-viet/cac-benh-truyen-nhiem-thuong-gap.html'
            ]
        
        total_diseases = 0
        
        for url in urls:
            try:
                logger.info(f'Fetching data from {url}...')
                
                # Headers để tránh bị block
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'vi-VN,vi;q=0.8,en-US;q=0.5,en;q=0.3',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive'
                }
                
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                response.encoding = 'utf-8'
                
                # Xử lý nội dung HTML
                diseases_data = self.extract_from_url_improved(url, response.text)
                
                # Cập nhật database
                imported_count = 0
                for data in diseases_data:
                    try:
                        # Tạo hoặc cập nhật bệnh
                        disease, created = Disease.objects.update_or_create(
                            name=data['name'],
                            defaults={
                                'description': data['description'][:1000] if data['description'] else '',
                                'causes': data.get('causes', '')[:1000],
                                'is_contagious': data.get('is_contagious', False),
                                'source_url': url
                            }
                        )
                        
                        # Thêm triệu chứng
                        for symptom_name in data.get('symptoms', []):
                            if symptom_name and len(symptom_name.strip()) > 2:
                                symptom, _ = Symptom.objects.get_or_create(
                                    name=symptom_name[:200],
                                    defaults={'description': ''}
                                )
                                DiseaseSymptom.objects.get_or_create(
                                    disease=disease, 
                                    symptom=symptom
                                )
                        
                        # Thêm biến chứng
                        for complication_name in data.get('complications', []):
                            if complication_name and len(complication_name.strip()) > 2:
                                complication, _ = Complication.objects.get_or_create(
                                    name=complication_name[:200],
                                    defaults={'description': ''}
                                )
                                disease.complications.add(complication)
                        
                        # Thêm phương pháp phòng ngừa
                        for prevention_method in data.get('preventions', []):
                            if prevention_method and len(prevention_method.strip()) > 2:
                                prevention, _ = Prevention.objects.get_or_create(
                                    method=prevention_method[:200], 
                                    defaults={'description': ''}
                                )
                                disease.preventions.add(prevention)
                        
                        # Thêm vắc-xin
                        for vaccine_name in data.get('vaccines', []):
                            if vaccine_name and len(vaccine_name.strip()) > 2:
                                vaccine, _ = Vaccine.objects.get_or_create(
                                    name=vaccine_name[:200],
                                    defaults={'manufacturer': ''}
                                )
                                disease.vaccines.add(vaccine)
                        
                        logger.info(f"Imported {disease.name}")
                        imported_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error importing disease {data.get('name', 'Unknown')}: {e}")
                
                # Cập nhật thông tin nguồn URL
                try:
                    source, _ = URLSource.objects.get_or_create(url=url)
                    source.last_updated = timezone.now()
                    source.success_count = imported_count
                    source.save()
                except Exception as e:
                    logger.error(f"Error updating URL source: {e}")
                
                total_diseases += imported_count
                logger.info(f"Successfully imported {imported_count} diseases from {url}")
                
            except requests.RequestException as e:
                logger.error(f'Error fetching URL {url}: {e}')
            except Exception as e:
                logger.error(f'Unexpected error processing URL {url}: {e}')
        
        # Khởi tạo lại vectorizer và vectors sau khi cập nhật dữ liệu
        if total_diseases > 0:
            self.init_symptoms()
            self.init_diseases()
        
        return total_diseases
    
    def extract_from_url_improved(self, url, html_content):
        """Phương thức trích xuất cải tiến dựa trên URL và phân tích nội dung"""
        logger.info(f"Extracting information from {url}")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Loại bỏ script, style, nav, footer để có nội dung sạch hơn
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        # Xác định phương thức trích xuất dựa trên URL
        if 'vnvc.vn' in url:
            return self.extract_from_vnvc_improved(soup)
        elif 'vinmec.com' in url:
            return self.extract_from_vinmec_improved(soup)
        elif 'longchau.com' in url or 'nhathuoclongchau.com' in url:
            return self.extract_from_longchau_improved(soup)
        elif 'medda.vn' in url:
            return self.extract_from_medda_improved(soup)
        else:
            # Thử phương pháp phân tích tự động
            return self.extract_auto_analysis(soup)
    
    def extract_from_vnvc_improved(self, soup):
        """Trích xuất cải tiến từ trang VNVC"""
        diseases = []
        
        # Tìm nội dung chính
        main_content = (soup.find('div', class_='post-content') or 
                       soup.find('div', class_='content') or
                       soup.find('article') or
                       soup.find('main'))
        
        if not main_content:
            main_content = soup
        
        # Tìm các đoạn văn bản có chứa từ khóa bệnh
        paragraphs = main_content.find_all('p')
        
        current_disease = None
        disease_content = []
        
        for p in paragraphs:
            text = p.get_text().strip()
            if not text:
                continue
            
            # Kiểm tra xem có phải tiêu đề bệnh mới không
            if any(keyword in text.lower() for keyword in self.disease_keywords):
                # Lưu bệnh trước đó
                if current_disease and disease_content:
                    disease_info = self.parse_disease_content(current_disease, ' '.join(disease_content))
                    if disease_info:
                        diseases.append(disease_info)
                
                # Bắt đầu bệnh mới
                current_disease = text
                disease_content = []
            else:
                # Thêm nội dung vào bệnh hiện tại
                if current_disease:
                    disease_content.append(text)
        
        # Xử lý bệnh cuối cùng
        if current_disease and disease_content:
            disease_info = self.parse_disease_content(current_disease, ' '.join(disease_content))
            if disease_info:
                diseases.append(disease_info)
        
        # Nếu không tìm thấy bằng cách trên, thử tìm bằng headers
        if not diseases:
            diseases = self.extract_by_headers(soup)
        
        return diseases
    
    def extract_from_vinmec_improved(self, soup):
        """Trích xuất cải tiến từ trang Vinmec"""
        diseases = []
        
        # Tìm nội dung chính
        main_content = (soup.find('div', class_='detail-content') or
                       soup.find('div', class_='post-content') or
                       soup.find('article') or
                       soup.find('main'))
        
        if not main_content:
            main_content = soup
        
        # Tìm các tiêu đề và nội dung tương ứng
        headers = main_content.find_all(['h1', 'h2', 'h3', 'h4'])
        
        for header in headers:
            title = header.get_text().strip()
            
            # Kiểm tra xem tiêu đề có phải tên bệnh không
            if not any(keyword in title.lower() for keyword in self.disease_keywords):
                continue
            
            # Thu thập nội dung sau tiêu đề
            content_parts = []
            current = header.find_next_sibling()
            
            while current and current.name not in ['h1', 'h2', 'h3', 'h4']:
                if current.name in ['p', 'div', 'ul', 'ol']:
                    content_parts.append(current.get_text().strip())
                current = current.find_next_sibling()
            
            if content_parts:
                content = ' '.join(content_parts)
                disease_info = self.parse_disease_content(title, content)
                if disease_info:
                    diseases.append(disease_info)
        
        # Fallback: tìm theo danh sách
        if not diseases:
            diseases = self.extract_by_lists(soup)
        
        return diseases
    
    def extract_from_longchau_improved(self, soup):
        """Trích xuất cải tiến từ trang Long Châu"""
        diseases = []
        
        # Tìm nội dung chính
        main_content = (soup.find('div', class_='article-detail') or
                       soup.find('div', class_='post-content') or
                       soup.find('article') or
                       soup.find('main'))
        
        if not main_content:
            main_content = soup
        
        # Tìm các đoạn văn có chứa thông tin bệnh
        all_text = main_content.get_text()
        
        # Tách thành các đoạn
        sections = re.split(r'\n\s*\n', all_text)
        
        for section in sections:
            section = section.strip()
            if len(section) < 50:  # Bỏ qua đoạn quá ngắn
                continue
            
            # Tìm tên bệnh trong đoạn
            lines = section.split('\n')
            potential_disease_name = None
            
            for line in lines[:3]:  # Kiểm tra 3 dòng đầu
                line = line.strip()
                if any(keyword in line.lower() for keyword in self.disease_keywords):
                    potential_disease_name = line
                    break
            
            if potential_disease_name:
                disease_info = self.parse_disease_content(potential_disease_name, section)
                if disease_info:
                    diseases.append(disease_info)
        
        return diseases
    
    def extract_from_medda_improved(self, soup):
        """Trích xuất cải tiến từ trang Medda"""
        diseases = []
        
        text = soup.get_text()
        
        # Tìm các mục được đánh số
        numbered_patterns = re.finditer(r'(\d+)\.\s*([^\n\r]+)(?:\n|\r\n?)([^0-9]*?)(?=\d+\.\s|\Z)', text, re.DOTALL)
        
        for pattern in numbered_patterns:
            number = pattern.group(1)
            title = pattern.group(2).strip()
            content = pattern.group(3).strip()
            
            # Kiểm tra xem có phải tên bệnh không
            if any(keyword in title.lower() for keyword in self.disease_keywords):
                disease_info = self.parse_disease_content(title, content)
                if disease_info:
                    diseases.append(disease_info)
        
        return diseases
    
    def extract_by_headers(self, soup):
        """Trích xuất thông tin dựa trên headers"""
        diseases = []
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])
        
        for header in headers:
            title = header.get_text().strip()
            
            if any(keyword in title.lower() for keyword in self.disease_keywords) and len(title) > 5:
                # Thu thập nội dung sau header
                content_parts = []
                current = header.find_next_sibling()
                
                # Thu thập cho đến khi gặp header khác
                while current and current.name not in ['h1', 'h2', 'h3', 'h4', 'h5']:
                    if current.name in ['p', 'div']:
                        text = current.get_text().strip()
                        if text:
                            content_parts.append(text)
                    current = current.find_next_sibling()
                
                if content_parts:
                    content = ' '.join(content_parts)
                    disease_info = self.parse_disease_content(title, content)
                    if disease_info:
                        diseases.append(disease_info)
        
        return diseases
    
    def extract_by_lists(self, soup):
        """Trích xuất thông tin dựa trên danh sách"""
        diseases = []
        
        # Tìm các danh sách
        lists = soup.find_all(['ul', 'ol'])
        
        for list_elem in lists:
            items = list_elem.find_all('li')
            
            for item in items:
                text = item.get_text().strip()
                
                if any(keyword in text.lower() for keyword in self.disease_keywords) and len(text) > 10:
                    # Tách tên bệnh và mô tả
                    parts = text.split(':')
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        description = ':'.join(parts[1:]).strip()
                    else:
                        name = text
                        description = ""
                    
                    disease_info = self.parse_disease_content(name, description)
                    if disease_info:
                        diseases.append(disease_info)
        
        return diseases
    
    def extract_auto_analysis(self, soup):
        """Phân tích tự động cấu trúc trang"""
        diseases = []
        
        # Thử nhiều phương pháp khác nhau
        methods = [
            self.extract_by_headers,
            self.extract_by_lists,
            self.extract_by_paragraphs
        ]
        
        for method in methods:
            try:
                result = method(soup)
                if result:
                    diseases.extend(result)
                    break
            except Exception as e:
                logger.error(f"Error in auto analysis method: {e}")
                continue
        
        return diseases
    
    def extract_by_paragraphs(self, soup):
        """Trích xuất dựa trên phân tích đoạn văn"""
        diseases = []
        
        # Lấy tất cả đoạn văn
        paragraphs = soup.find_all('p')
        
        current_disease = None
        disease_content = []
        
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) < 20:
                continue
            
            # Kiểm tra xem có phải tiêu đề bệnh mới
            if (any(keyword in text.lower() for keyword in self.disease_keywords) and 
                len(text) < 100 and ':' not in text):
                
                # Lưu bệnh trước
                if current_disease and disease_content:
                    content = ' '.join(disease_content)
                    disease_info = self.parse_disease_content(current_disease, content)
                    if disease_info:
                        diseases.append(disease_info)
                
                # Bắt đầu bệnh mới
                current_disease = text
                disease_content = []
            else:
                # Thêm vào nội dung bệnh hiện tại
                if current_disease:
                    disease_content.append(text)
        
        # Xử lý bệnh cuối
        if current_disease and disease_content:
            content = ' '.join(disease_content)
            disease_info = self.parse_disease_content(current_disease, content)
            if disease_info:
                diseases.append(disease_info)
        
        return diseases
    
    def parse_disease_content(self, name, content):
        """Phân tích nội dung để trích xuất thông tin bệnh"""
        if not name or len(name.strip()) < 3:
            return None
        
        name = name.strip()
        if not content:
            content = ""
        
        # Trích xuất mô tả (lấy câu đầu tiên hoặc đoạn đầu)
        description = ""
        sentences = re.split(r'[.!?]', content)
        if sentences:
            description = sentences[0].strip()
        
        # Trích xuất triệu chứng
        symptoms = self.extract_symptoms_from_text(content)
        
        # Trích xuất nguyên nhân
        causes = self.extract_causes_from_text(content)
        
        # Trích xuất phòng ngừa
        preventions = self.extract_preventions_from_text(content)
        
        # Trích xuất biến chứng
        complications = self.extract_complications_from_text(content)
        
        # Trích xuất vắc-xin
        vaccines = self.extract_vaccines_from_text(content)
        
        # Xác định tính lây nhiễm
        is_contagious = self.is_contagious_disease(content)
        
        return {
            'name': name,
            'description': description,
            'causes': causes,
            'symptoms': symptoms,
            'complications': complications,
            'treatments': [],
            'preventions': preventions,
            'vaccines': vaccines,
            'is_contagious': is_contagious
        }
    
    def extract_symptoms_from_text(self, text):
        """Trích xuất triệu chứng từ text"""
        symptoms = []
        
        # Tìm các pattern về triệu chứng
        patterns = [
            r'triệu chứng[^:]*:([^.]*)',
            r'biểu hiện[^:]*:([^.]*)',
            r'dấu hiệu[^:]*:([^.]*)',
            r'bao gồm[^:]*:([^.]*)',
            r'như[^:]*:([^.]*)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower(), re.IGNORECASE)
            for match in matches:
                symptom_text = match.group(1).strip()
                # Tách các triệu chứng
                symptom_list = re.split(r'[,;]', symptom_text)
                for symptom in symptom_list:
                    symptom = symptom.strip()
                    if len(symptom) > 3 and len(symptom) < 100:
                        symptoms.append(symptom)
        
        # Tìm thêm triệu chứng bằng keyword matching
        symptom_keywords = ['đau', 'sốt', 'ho', 'chảy nước mũi', 'mệt mỏi', 'buồn nôn', 'tiêu chảy']
        words = text.lower().split()
        
        for i, word in enumerate(words):
            if any(keyword in word for keyword in symptom_keywords):
                # Lấy context xung quanh
                start = max(0, i-2)
                end = min(len(words), i+3)
                context = ' '.join(words[start:end])
                if len(context) > 5 and len(context) < 50:
                    symptoms.append(context)
        
        return list(set(symptoms))  # Loại bỏ duplicate
    
    def extract_causes_from_text(self, text):
        """Trích xuất nguyên nhân từ text"""
        patterns = [
            r'nguyên nhân[^:]*:([^.]*)',
            r'do ([^.]*) gây ra',
            r'bởi ([^.]*)',
            r'gây ra bởi ([^.]*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower(), re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def extract_preventions_from_text(self, text):
        """Trích xuất cách phòng ngừa từ text"""
        preventions = []
        
        patterns = [
            r'phòng ngừa[^:]*:([^.]*)',
            r'ngăn chặn[^:]*:([^.]*)',
            r'dự phòng[^:]*:([^.]*)',
            r'tránh ([^.]*)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower(), re.IGNORECASE)
            for match in matches:
                prevention_text = match.group(1).strip()
                prevention_list = re.split(r'[,;]', prevention_text)
                for prevention in prevention_list:
                    prevention = prevention.strip()
                    if len(prevention) > 5 and len(prevention) < 200:
                        preventions.append(prevention)
        
        return list(set(preventions))
    
    def extract_complications_from_text(self, text):
        """Trích xuất biến chứng từ text"""
        complications = []
        
        patterns = [
            r'biến chứng[^:]*:([^.]*)',
            r'tai biến[^:]*:([^.]*)',
            r'có thể dẫn đến ([^.]*)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower(), re.IGNORECASE)
            for match in matches:
                complication_text = match.group(1).strip()
                complication_list = re.split(r'[,;]', complication_text)
                for complication in complication_list:
                    complication = complication.strip()
                    if len(complication) > 5 and len(complication) < 200:
                        complications.append(complication)
        
        return list(set(complications))
    
    def extract_vaccines_from_text(self, text):
        """Trích xuất thông tin vắc-xin từ text"""
        vaccines = []
        
        patterns = [
            r'vắc[- ]?xin[^:]*:([^.]*)',
            r'tiêm chủng[^:]*:([^.]*)',
            r'vaccine[^:]*:([^.]*)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower(), re.IGNORECASE)
            for match in matches:
                vaccine_text = match.group(1).strip()
                vaccine_list = re.split(r'[,;]', vaccine_text)
                for vaccine in vaccine_list:
                    vaccine = vaccine.strip()
                    if len(vaccine) > 3 and len(vaccine) < 100:
                        vaccines.append(vaccine)
        
        return list(set(vaccines))
    
    def is_contagious_disease(self, text):
        """Xác định bệnh có lây nhiễm không"""
        contagious_keywords = [
            'lây', 'truyền nhiễm', 'virus', 'vi khuẩn', 'nhiễm trùng',
            'vi-rút', 'lây lan', 'lây truyền', 'dịch bệnh'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in contagious_keywords)
    
    def find_matching_symptoms(self, query, top_n=3):
        """Tìm triệu chứng phù hợp với query"""
        if not self.symptoms or not self.symptom_vectorizer:
            return []
        
        try:
            query = self.preprocess_text(query)
            query_vector = self.symptom_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.symptom_vectors).flatten()
            
            top_indices = np.argsort(similarities)[-top_n:][::-1]
            top_symptoms = [(self.symptoms[i], similarities[i]) for i in top_indices if similarities[i] > 0.1]
            
            return top_symptoms
        except Exception as e:
            logger.error(f"Error finding matching symptoms: {e}")
            return []
    
    def find_matching_diseases(self, query, top_n=3):
        """Tìm bệnh phù hợp với query"""
        if not self.diseases or not self.disease_vectorizer:
            return []
        
        try:
            query = self.preprocess_text(query)
            query_vector = self.disease_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.disease_vectors).flatten()
            
            top_indices = np.argsort(similarities)[-top_n:][::-1]
            top_diseases = [(self.diseases[i], similarities[i]) for i in top_indices if similarities[i] > 0.1]
            
            return top_diseases
        except Exception as e:
            logger.error(f"Error finding matching diseases: {e}")
            return []
    
    def process_query(self, query):
        """Xử lý query từ người dùng"""
        try:
            # Tìm triệu chứng phù hợp
            matching_symptoms = self.find_matching_symptoms(query)
            
            # Tìm bệnh phù hợp
            matching_diseases = self.find_matching_diseases(query)
            
            # Nếu có cả triệu chứng và bệnh phù hợp
            if matching_symptoms and matching_diseases:
                symptoms_text = ", ".join([s[0].name for s in matching_symptoms])
                diseases_text = ", ".join([d[0].name for d in matching_diseases])
                
                return (f"Tôi nhận thấy bạn có thể đang mô tả các triệu chứng: {symptoms_text}. "
                       f"Điều này có thể liên quan đến: {diseases_text}. "
                       f"Xin lưu ý đây chỉ là thông tin tham khảo, vui lòng tham khảo ý kiến bác sĩ.")
            
            # Nếu chỉ có triệu chứng phù hợp
            elif matching_symptoms:
                symptoms_text = ", ".join([s[0].name for s in matching_symptoms])
                
                # Tìm các bệnh liên quan đến các triệu chứng này
                related_diseases = set()
                for symptom, _ in matching_symptoms:
                    for link in symptom.diseases_link.all():
                        related_diseases.add(link.disease)
                
                if related_diseases:
                    diseases_text = ", ".join([d.name for d in related_diseases])
                    return (f"Tôi nhận thấy bạn có thể đang mô tả các triệu chứng: {symptoms_text}. "
                           f"Những triệu chứng này có thể liên quan đến: {diseases_text}. "
                           f"Xin lưu ý đây chỉ là thông tin tham khảo, vui lòng tham khảo ý kiến bác sĩ.")
                else:
                    return (f"Tôi nhận thấy bạn có thể đang mô tả các triệu chứng: {symptoms_text}. "
                           f"Tôi không có đủ thông tin để xác định bệnh cụ thể. "
                           f"Vui lòng mô tả chi tiết hơn hoặc tham khảo ý kiến bác sĩ.")
            
            # Nếu chỉ có bệnh phù hợp
            elif matching_diseases:
                disease = matching_diseases[0][0]  # Lấy bệnh phù hợp nhất
                
                # Tạo câu trả lời chi tiết về bệnh
                response = f"**{disease.name}**\n\n{disease.description}"
                
                # Thêm thông tin về triệu chứng
                symptom_links = disease.symptoms_link.all()
                if symptom_links:
                    symptoms_text = ", ".join([link.symptom.name for link in symptom_links])
                    response += f"\n\n**Triệu chứng thường gặp:** {symptoms_text}"
                
                # Thêm thông tin về biến chứng
                complications = disease.complications.all()
                if complications:
                    complications_text = ", ".join([c.name for c in complications])
                    response += f"\n\n**Biến chứng có thể xảy ra:** {complications_text}"
                
                # Thêm thông tin về cách phòng ngừa
                preventions = disease.preventions.all()
                if preventions:
                    preventions_text = ", ".join([p.method for p in preventions])
                    response += f"\n\n**Cách phòng ngừa:** {preventions_text}"
                
                # Thêm thông tin về vắc-xin nếu có
                vaccines = disease.vaccines.all()
                if vaccines:
                    vaccines_text = ", ".join([v.name for v in vaccines])
                    response += f"\n\n**Vắc-xin phòng bệnh:** {vaccines_text}"
                
                # Thêm thông tin về nguồn
                if disease.source_url:
                    response += f"\n\n**Nguồn tham khảo:** {disease.source_url}"
                
                response += "\n\n⚠️ **Lưu ý:** Đây chỉ là thông tin tham khảo, vui lòng tham khảo ý kiến bác sĩ."
                
                return response
            
            # Nếu không có kết quả phù hợp
            else:
                return ("Tôi không có đủ thông tin để xử lý yêu cầu của bạn. "
                       "Vui lòng mô tả chi tiết hơn về triệu chứng hoặc bệnh bạn đang tìm hiểu. "
                       "Bạn cũng có thể hỏi về:\n"
                       "- Các triệu chứng cụ thể (ví dụ: sốt, ho, đau đầu)\n"
                       "- Tên bệnh cụ thể\n"
                       "- Cách phòng ngừa bệnh")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."