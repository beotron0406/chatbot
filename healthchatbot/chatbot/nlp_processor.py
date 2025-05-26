import re
import string
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode
import numpy as np
from django.utils import timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Disease, Symptom, DiseaseSymptom, Complication, Prevention, Vaccine, URLSource

class NLPProcessor:
    def __init__(self):
        self.symptom_vectorizer = None
        self.symptom_vectors = None
        self.symptoms = None
        self.init_symptoms()
        
        self.disease_vectorizer = None
        self.disease_vectors = None
        self.diseases = None
        self.init_diseases()
    
    def init_symptoms(self):
        # Tải tất cả triệu chứng từ database
        self.symptoms = list(Symptom.objects.all())
        
        if not self.symptoms:
            return
        
        # Chuẩn bị dữ liệu cho vectorizer
        symptom_texts = [self.preprocess_text(s.name + " " + (s.description or "")) 
                         for s in self.symptoms]
        
        # Tạo vectorizer và vectors
        self.symptom_vectorizer = TfidfVectorizer()
        self.symptom_vectors = self.symptom_vectorizer.fit_transform(symptom_texts)
    
    def init_diseases(self):
        # Tải tất cả bệnh từ database
        self.diseases = list(Disease.objects.all())
        
        if not self.diseases:
            return
        
        # Chuẩn bị dữ liệu cho vectorizer
        disease_texts = []
        for disease in self.diseases:
            text = disease.name + " " + disease.description
            
            # Thêm triệu chứng vào text
            for link in disease.symptoms_link.all():
                text += " " + link.symptom.name
            
            disease_texts.append(self.preprocess_text(text))
        
        # Tạo vectorizer và vectors
        self.disease_vectorizer = TfidfVectorizer()
        self.disease_vectors = self.disease_vectorizer.fit_transform(disease_texts)
    
    def preprocess_text(self, text):
        # Chuyển về chữ thường
        text = text.lower()
        
        # Loại bỏ dấu tiếng Việt
        text = unidecode(text)
        
        # Loại bỏ số và dấu câu
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def fetch_and_update_knowledge_base(self, urls=None):
        """Lấy dữ liệu từ URL và cập nhật knowledge base"""
        if urls is None:
            urls = [
                'https://vnvc.vn/benh-truyen-nhiem/',
                'https://www.vinmec.com/vi/benh/benh-truyen-nhiem-1/',
                'https://suckhoedoisong.vn/benh-truyen-nhiem/'
            ]
        
        total_diseases = 0
        
        for url in urls:
            try:
                # Lấy nội dung từ URL
                print(f'Fetching data from {url}...')
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Xử lý nội dung HTML
                content = self.extract_from_html(response.text)
                
                # Trích xuất thông tin bệnh
                diseases_data = self.extract_disease_info(content)
                
                # Cập nhật database
                for data in diseases_data:
                    # Thêm URL nguồn
                    data['source_url'] = url
                    
                    # Tạo hoặc cập nhật bệnh
                    disease, created = Disease.objects.update_or_create(
                        name=data['name'],
                        defaults={
                            'description': data['description'],
                            'causes': data['causes'],
                            'is_contagious': data['is_contagious'],
                            'source_url': data['source_url']
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
                        prevention, _ = Prevention.objects.get_or_create(
                            method=prevention_method, 
                            defaults={'description': ''}
                        )
                        disease.preventions.add(prevention)
                    
                    # Thêm vắc-xin
                    for vaccine_name in data['vaccines']:
                        vaccine, _ = Vaccine.objects.get_or_create(name=vaccine_name)
                        disease.vaccines.add(vaccine)
                    
                    total_diseases += 1
                    print(f"Imported {disease.name}")
                
            except Exception as e:
                print(f'Error processing URL {url}: {e}')
        
        # Khởi tạo lại vectorizer và vectors sau khi cập nhật dữ liệu
        self.init_symptoms()
        self.init_diseases()
        
        return total_diseases
    
    def extract_from_html(self, html_content):
        """Trích xuất nội dung văn bản từ HTML"""
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
        """Trích xuất thông tin bệnh từ text"""
        diseases = []
        
        # Tìm phần tài liệu liệt kê các bệnh
        disease_section = re.search(r'Các bệnh truyền nhiễm thường gặp(.*?)Làm sao để phòng ngừa', text, re.DOTALL)
        
        if not disease_section:
            # Thử tìm với cách khác nếu không tìm thấy với pattern trên
            disease_section = re.search(r'([\d]+\.\s+[\w\s]+)\n', text, re.DOTALL)
        
        if disease_section:
            disease_text = disease_section.group(1)
            
            # Tìm các phần bệnh riêng lẻ
            disease_patterns = re.finditer(r'(\d+)\.\s+(.*?)(?=\d+\.\s+|\Z)', disease_text, re.DOTALL)
            
            for pattern in disease_patterns:
                number = pattern.group(1)
                content = pattern.group(2).strip()
                
                # Tách tên và mô tả bệnh
                name_match = re.match(r'([^\n]+)', content)
                name = name_match.group(1) if name_match else f"Bệnh {number}"
                
                # [Phần code trích xuất thông tin chi tiết - giữ nguyên như trước]
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
        
        return diseases

    # [Các phương thức khác trong NLPProcessor - giữ nguyên]
    def find_matching_symptoms(self, query, top_n=3):
        if not self.symptoms or not self.symptom_vectorizer:
            return []
        
        # Tiền xử lý query
        query = self.preprocess_text(query)
        
        # Chuyển query thành vector
        query_vector = self.symptom_vectorizer.transform([query])
        
        # Tính độ tương đồng
        similarities = cosine_similarity(query_vector, self.symptom_vectors).flatten()
        
        # Lấy các triệu chứng có độ tương đồng cao nhất
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        top_symptoms = [(self.symptoms[i], similarities[i]) for i in top_indices if similarities[i] > 0.1]
        
        return top_symptoms
    
    def find_matching_diseases(self, query, top_n=3):
        if not self.diseases or not self.disease_vectorizer:
            return []
        
        # Tiền xử lý query
        query = self.preprocess_text(query)
        
        # Chuyển query thành vector
        query_vector = self.disease_vectorizer.transform([query])
        
        # Tính độ tương đồng
        similarities = cosine_similarity(query_vector, self.disease_vectors).flatten()
        
        # Lấy các bệnh có độ tương đồng cao nhất
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        top_diseases = [(self.diseases[i], similarities[i]) for i in top_indices if similarities[i] > 0.1]
        
        return top_diseases
    
    def process_query(self, query):
        # [Giữ nguyên phương thức xử lý query]
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
            response = f"Bệnh {disease.name}: {disease.description}"
            
            # Thêm thông tin về triệu chứng
            symptom_links = disease.symptoms_link.all()
            if symptom_links:
                symptoms_text = ", ".join([link.symptom.name for link in symptom_links])
                response += f"\n\nTriệu chứng thường gặp: {symptoms_text}"
            
            # Thêm thông tin về biến chứng
            complications = disease.complications.all()
            if complications:
                complications_text = ", ".join([c.name for c in complications])
                response += f"\n\nBiến chứng có thể xảy ra: {complications_text}"
            
            # Thêm thông tin về cách phòng ngừa
            preventions = disease.preventions.all()
            if preventions:
                preventions_text = ", ".join([p.method for p in preventions])
                response += f"\n\nCách phòng ngừa: {preventions_text}"
            
            # Thêm thông tin về vắc-xin nếu có
            vaccines = disease.vaccines.all()
            if vaccines:
                vaccines_text = ", ".join([v.name for v in vaccines])
                response += f"\n\nVắc-xin phòng bệnh: {vaccines_text}"
            
            # Thêm thông tin về nguồn
            if disease.source_url:
                response += f"\n\nNguồn tham khảo: {disease.source_url}"
            
            response += "\n\nXin lưu ý đây chỉ là thông tin tham khảo, vui lòng tham khảo ý kiến bác sĩ."
            
            return response
        
        # Nếu không có kết quả phù hợp
        else:
            return "Tôi không có đủ thông tin để xử lý yêu cầu của bạn. Vui lòng mô tả chi tiết hơn về triệu chứng hoặc bệnh bạn đang tìm hiểu."
        
    def fetch_and_update_knowledge_base(self, urls=None):
        """Lấy dữ liệu từ URL và cập nhật knowledge base"""
        if urls is None:
            urls = [
                'https://vnvc.vn/cac-benh-truyen-nhiem-thuong-gap/',
                # Thêm các URL mặc định khác
            ]
        
        total_diseases = 0
        
        for url in urls:
            try:
                # Lấy nội dung từ URL
                print(f'Fetching data from {url}...')
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Xử lý nội dung HTML
                content = self.extract_from_html(response.text)
                
                # Trích xuất thông tin bệnh dựa vào URL
                diseases_data = self.extract_from_url(url, content)
                
                # Cập nhật database
                for data in diseases_data:
                    # Tạo hoặc cập nhật bệnh
                    disease, created = Disease.objects.update_or_create(
                        name=data['name'],
                        defaults={
                            'description': data['description'],
                            'causes': data.get('causes', ''),
                            'is_contagious': data.get('is_contagious', False),
                            'source_url': url
                        }
                    )
                    
                    # Thêm triệu chứng
                    for symptom_name in data.get('symptoms', []):
                        if symptom_name:
                            symptom, _ = Symptom.objects.get_or_create(name=symptom_name)
                            DiseaseSymptom.objects.get_or_create(disease=disease, symptom=symptom)
                    
                    # Thêm biến chứng
                    for complication_name in data.get('complications', []):
                        if complication_name:
                            complication, _ = Complication.objects.get_or_create(name=complication_name)
                            disease.complications.add(complication)
                    
                    # Thêm phương pháp phòng ngừa
                    for prevention_method in data.get('preventions', []):
                        if prevention_method:
                            prevention, _ = Prevention.objects.get_or_create(
                                method=prevention_method, 
                                defaults={'description': ''}
                            )
                            disease.preventions.add(prevention)
                    
                    # Thêm vắc-xin
                    for vaccine_name in data.get('vaccines', []):
                        if vaccine_name:
                            vaccine, _ = Vaccine.objects.get_or_create(name=vaccine_name)
                            disease.vaccines.add(vaccine)
                    
                    print(f"Imported {disease.name}")
                    total_diseases += 1
                
                # Cập nhật thông tin nguồn URL
                try:
                    source, _ = URLSource.objects.get_or_create(url=url)
                    source.last_updated = timezone.now()
                    source.success_count = len(diseases_data)
                    source.save()
                except Exception as e:
                    print(f"Error updating URL source: {e}")
                
                print(f"Successfully imported {len(diseases_data)} diseases from {url}")
                
            except Exception as e:
                print(f'Error processing URL {url}: {e}')
        
        # Khởi tạo lại vectorizer và vectors sau khi cập nhật dữ liệu
        self.init_symptoms()
        self.init_diseases()
        
        return total_diseases

    def extract_from_url(self, url, content):
        """Chọn phương thức trích xuất phù hợp dựa trên URL và nội dung"""
        print(f"Extracting information from {url}")
        
        # Xác định phương thức trích xuất dựa trên URL
        if 'vnvc.vn' in url:
            return self.extract_disease_info_vnvc(content)
        elif 'vinmec.com' in url:
            return self.extract_disease_info_vinmec(content)
        elif 'nhathuoclongchau.com' in url or 'longchau.com' in url:
            return self.extract_disease_info_longchau(content)
        elif 'medda.vn' in url:
            return self.extract_disease_info_medda(content)
        else:
            # Thử phân tích nội dung để xác định cấu trúc
            soup = BeautifulSoup(content, 'html.parser')
            
            # Kiểm tra có phải trang Long Châu không
            if soup.find('div', class_='article-detail') or "Long Châu" in content:
                return self.extract_disease_info_longchau(content)
            # Kiểm tra có phải trang Medda không
            elif re.search(r'\d+\.\s+[\w\s,]+\n+', content):
                return self.extract_disease_info_medda(content)
            # Kiểm tra có phải trang VNVC không
            elif "Các bệnh truyền nhiễm thường gặp" in content:
                return self.extract_disease_info_vnvc(content)
            # Nếu không xác định được, dùng phương thức chung
            else:
                return self.extract_disease_info_general(content)

    def extract_disease_info_vnvc(self, content):
        """Trích xuất thông tin bệnh từ trang VNVC"""
        # Đây là phương thức trích xuất hiện tại của bạn
        diseases = []
        
        # Tìm phần tài liệu liệt kê các bệnh
        disease_section = re.search(r'Các bệnh truyền nhiễm thường gặp(.*?)Làm sao để phòng ngừa', content, re.DOTALL)
        
        if not disease_section:
            # Thử tìm với cách khác
            disease_section = re.search(r'([\d]+\.\s+[\w\s]+)\n', content, re.DOTALL)
        
        if disease_section:
            disease_text = disease_section.group(1)
            
            # Tìm các phần bệnh riêng lẻ
            disease_patterns = re.finditer(r'(\d+)\.\s+(.*?)(?=\d+\.\s+|\Z)', disease_text, re.DOTALL)
            
            for pattern in disease_patterns:
                number = pattern.group(1)
                content = pattern.group(2).strip()
                
                # Tách tên và mô tả bệnh
                name_match = re.match(r'([^\n]+)', content)
                name = name_match.group(1) if name_match else f"Bệnh {number}"
                
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
        
        return diseases

    def extract_disease_info_vinmec(self, content):
        """Trích xuất thông tin bệnh từ trang Vinmec"""
        diseases = []
        soup = BeautifulSoup(content, 'html.parser')
        
        # Tìm phần nội dung chính
        article = soup.find('div', class_='detail-content')
        if not article:
            return diseases
        
        # Tìm các tiêu đề có thể chứa tên bệnh
        headers = article.find_all(['h2', 'h3'])
        
        for header in headers:
            disease_name = header.text.strip()
            
            # Kiểm tra xem tiêu đề có phải là tên bệnh không
            if not any(keyword in disease_name.lower() for keyword in ['bệnh', 'viêm', 'nhiễm', 'sốt', 'hội chứng']):
                continue
            
            # Lấy nội dung mô tả (đoạn văn bản sau tiêu đề)
            description = ""
            current = header.find_next_sibling()
            while current and current.name in ['p', 'div', 'ul']:
                if current.name == 'p':
                    description += current.text.strip() + " "
                current = current.find_next_sibling()
            
            # Kiểm tra xem có phải là bệnh truyền nhiễm
            is_contagious = any(term in description.lower() for term in ['lây', 'truyền nhiễm', 'vi khuẩn', 'virus'])
            
            # Tìm triệu chứng
            symptoms = []
            symptom_section = article.find(text=re.compile('triệu chứng|biểu hiện', re.IGNORECASE))
            if symptom_section and symptom_section.parent:
                symptom_list = symptom_section.parent.find_next('ul')
                if symptom_list:
                    for li in symptom_list.find_all('li'):
                        symptoms.append(li.text.strip())
            
            # Thêm bệnh vào danh sách
            diseases.append({
                'name': disease_name,
                'description': description.strip(),
                'causes': "",
                'symptoms': symptoms,
                'complications': [],
                'treatments': [],
                'preventions': [],
                'vaccines': [],
                'is_contagious': is_contagious
            })
            
            print(f"Extracted information for: {disease_name}")
        
        return diseases

    def extract_disease_info_longchau(self, content):
        """Trích xuất thông tin bệnh từ trang Nhà thuốc Long Châu"""
        diseases = []
        soup = BeautifulSoup(content, 'html.parser')
        
        # Tìm nội dung chính của bài viết
        article_content = soup.find('div', class_='article-detail')
        if not article_content:
            # Thử tìm với các class khác nếu không tìm thấy
            article_content = soup.find('div', class_='post-content')
        
        if not article_content:
            print("Không tìm thấy nội dung chính của bài viết")
            return diseases
        
        # Tìm tất cả các tiêu đề bệnh trong nội dung
        # Long Châu thường sử dụng các thẻ h2, h3 hoặc các đoạn text in đậm
        disease_headers = article_content.find_all(['h2', 'h3', 'strong'])
        
        for header in disease_headers:
            text = header.text.strip()
            
            # Lọc ra các tiêu đề bệnh (thường có từ "bệnh", "viêm", "đau", v.v.)
            if any(keyword in text.lower() for keyword in ['bệnh', 'viêm', 'đau', 'cảm', 'sốt', 'tiểu đường', 'tim mạch', 'sỏi', 'khớp']):
                disease_name = text
                
                # Tìm nội dung mô tả (đoạn văn bản sau tiêu đề)
                description = ""
                next_element = header.find_next(['p', 'div'])
                
                # Thu thập tất cả đoạn văn bản cho đến khi gặp tiêu đề tiếp theo
                while next_element and next_element.name in ['p', 'div'] and not next_element.find(['h2', 'h3', 'strong']):
                    if next_element.text.strip():
                        description += next_element.text.strip() + " "
                    next_element = next_element.find_next_sibling()
                
                # Kiểm tra xem đây có phải là bệnh truyền nhiễm không
                is_contagious = any(term in description.lower() for term in [
                    'lây', 'truyền nhiễm', 'virus', 'vi khuẩn', 'nhiễm trùng', 
                    'vi-rút', 'nhiễm', 'lây lan', 'lây truyền'
                ])
                
                # Trích xuất thông tin về triệu chứng (nếu có)
                symptoms = []
                symptom_text = ""
                symptom_patterns = ['triệu chứng', 'biểu hiện', 'dấu hiệu', 'nhận biết']
                
                # Tìm đoạn văn bản chứa thông tin về triệu chứng
                for pattern in symptom_patterns:
                    if pattern in description.lower():
                        # Lấy nội dung sau pattern
                        symptom_match = re.search(r'{}.*?[:;](.+?)(?=\.|$)'.format(pattern), 
                                                description.lower(), re.IGNORECASE | re.DOTALL)
                        if symptom_match:
                            symptom_text = symptom_match.group(1).strip()
                            break
                
                # Phân tích triệu chứng từ đoạn text
                if symptom_text:
                    # Tách thành các triệu chứng riêng biệt
                    # Thử tách theo dấu phẩy
                    if ',' in symptom_text:
                        symptoms = [s.strip() for s in symptom_text.split(',') if s.strip()]
                    # Thử tách theo dấu chấm
                    elif '.' in symptom_text:
                        symptoms = [s.strip() for s in symptom_text.split('.') if s.strip()]
                
                # Trích xuất thông tin về nguyên nhân
                causes = ""
                cause_patterns = ['nguyên nhân', 'do', 'gây ra bởi', 'gây ra do']
                
                for pattern in cause_patterns:
                    if pattern in description.lower():
                        cause_match = re.search(r'{}.*?[:;](.+?)(?=\.|$)'.format(pattern), 
                                            description.lower(), re.IGNORECASE | re.DOTALL)
                        if cause_match:
                            causes = cause_match.group(1).strip()
                            break
                
                # Trích xuất thông tin về phòng ngừa
                preventions = []
                prevention_text = ""
                prevention_patterns = ['phòng ngừa', 'ngăn chặn', 'phòng bệnh', 'dự phòng', 'giảm thiểu']
                
                for pattern in prevention_patterns:
                    if pattern in description.lower():
                        prevention_match = re.search(r'{}.*?[:;](.+?)(?=\.|$)'.format(pattern), 
                                                description.lower(), re.IGNORECASE | re.DOTALL)
                        if prevention_match:
                            prevention_text = prevention_match.group(1).strip()
                            break
                
                if prevention_text:
                    if ',' in prevention_text:
                        preventions = [p.strip() for p in prevention_text.split(',') if p.strip()]
                    elif '.' in prevention_text:
                        preventions = [p.strip() for p in prevention_text.split('.') if p.strip()]
                
                # Thêm bệnh vào danh sách
                diseases.append({
                    'name': disease_name,
                    'description': description,
                    'causes': causes,
                    'symptoms': symptoms,
                    'complications': [],
                    'treatments': [],
                    'preventions': preventions,
                    'vaccines': [],
                    'is_contagious': is_contagious
                })
                
                print(f"Extracted information for: {disease_name}")
        
        return diseases
        """Trích xuất thông tin bệnh từ trang Nhà thuốc Long Châu"""
        diseases = []
        soup = BeautifulSoup(content, 'html.parser')
        
        # Tìm phần nội dung chính
        article = soup.find('div', class_='post-content')
        if not article:
            return diseases
        
        # Tìm các tiêu đề có thể chứa tên bệnh
        headers = article.find_all(['h2', 'h3'])
        
        for header in headers:
            disease_name = header.text.strip()
            
            # Bỏ qua các tiêu đề không phải tên bệnh
            if not any(keyword in disease_name.lower() for keyword in ['bệnh', 'viêm', 'nhiễm', 'sốt']):
                continue
            
            # Trích xuất thông tin tương tự như Vinmec
            description = ""
            current = header.find_next_sibling()
            if current and current.name == 'p':
                description = current.text.strip()
            
            # Kiểm tra xem có phải là bệnh truyền nhiễm
            is_contagious = any(term in description.lower() for term in ['lây', 'truyền nhiễm', 'vi khuẩn', 'virus'])
            
            # Thêm bệnh vào danh sách
            diseases.append({
                'name': disease_name,
                'description': description,
                'causes': "",
                'symptoms': [],
                'complications': [],
                'treatments': [],
                'preventions': [],
                'vaccines': [],
                'is_contagious': is_contagious
            })
            
            print(f"Extracted information for: {disease_name}")
        
        return diseases

    def extract_disease_info_medda(self, content):
        """Trích xuất thông tin bệnh từ trang Medda"""
        diseases = []
        
        # Tìm các mục bệnh bằng regex - tìm mẫu như "1. Tên bệnh", "2. Tên bệnh"
        disease_patterns = re.finditer(r'(\d+)\.\s+([\w\s,]+)(?:\n+|\r\n+)', content, re.DOTALL)
        
        for pattern in disease_patterns:
            number = pattern.group(1)
            disease_name = pattern.group(2).strip()
            
            # Tìm nội dung mô tả của bệnh
            start_pos = pattern.end()
            
            # Tìm vị trí của bệnh tiếp theo hoặc kết thúc văn bản
            next_pattern = re.search(r'{}\.'.format(int(number) + 1), content)
            if next_pattern:
                end_pos = next_pattern.start()
            else:
                end_pos = len(content)
            
            # Lấy mô tả
            description = content[start_pos:end_pos].strip()
            
            # Tìm thông tin về triệu chứng (nếu có)
            symptoms = []
            symptom_match = re.search(r'triệu chứng.*?[:;](.*?)(?=\.|$)', description, re.IGNORECASE | re.DOTALL)
            if symptom_match:
                symptom_text = symptom_match.group(1).strip()
                symptoms = [s.strip() for s in symptom_text.split(',') if s.strip()]
            
            # Kiểm tra xem có phải là bệnh truyền nhiễm không
            is_contagious = any(term in description.lower() for term in ['lây', 'truyền nhiễm', 'virus', 'vi khuẩn', 'nhiễm trùng'])
            
            # Tìm thông tin về nguyên nhân
            causes = ""
            cause_match = re.search(r'nguyên nhân.*?[:;](.*?)(?=\.|$)', description, re.IGNORECASE | re.DOTALL)
            if cause_match:
                causes = cause_match.group(1).strip()
            
            # Tìm thông tin về phòng ngừa
            preventions = []
            prevention_match = re.search(r'phòng ngừa|phòng chống|ngăn chặn.*?[:;](.*?)(?=\.|$)', description, re.IGNORECASE | re.DOTALL)
            if prevention_match:
                prevention_text = prevention_match.group(1).strip()
                preventions = [p.strip() for p in prevention_text.split(',') if p.strip()]
            
            # Thêm bệnh vào danh sách
            diseases.append({
                'name': disease_name,
                'description': description,
                'causes': causes,
                'symptoms': symptoms,
                'complications': [],
                'treatments': [],
                'preventions': preventions,
                'vaccines': [],
                'is_contagious': is_contagious
            })
            
            print(f"Extracted information for: {disease_name}")
        
        return diseases

    def extract_disease_info_general(self, content):
        """Phương thức trích xuất chung cho các trang web khác"""
        diseases = []
        soup = BeautifulSoup(content, 'html.parser')
        
        # Tìm tất cả các tiêu đề
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        
        for header in headers:
            text = header.text.strip().lower()
            
            # Kiểm tra xem tiêu đề có chứa từ khóa liên quan đến bệnh không
            if any(keyword in text for keyword in ['bệnh', 'viêm', 'nhiễm', 'sốt', 'hội chứng']):
                disease_name = header.text.strip()
                
                # Lấy nội dung mô tả
                description = ""
                element = header.find_next_sibling()
                if element and element.name == 'p':
                    description = element.text.strip()
                
                # Kiểm tra xem có phải là bệnh truyền nhiễm
                content_text = " ".join([p.text for p in header.find_next_siblings('p', limit=3)])
                is_contagious = any(term in content_text.lower() for term in ['lây', 'truyền nhiễm', 'vi khuẩn', 'virus'])
                
                # Thêm bệnh vào danh sách
                diseases.append({
                    'name': disease_name,
                    'description': description,
                    'causes': "",
                    'symptoms': [],
                    'complications': [],
                    'treatments': [],
                    'preventions': [],
                    'vaccines': [],
                    'is_contagious': is_contagious
                })
                
                print(f"Extracted information for: {disease_name}")
        
        return diseases