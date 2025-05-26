import re
import string
from unidecode import unidecode
import logging

logger = logging.getLogger(__name__)

class VietnameseMedicalProcessor:
    """Processor chuyên biệt cho xử lý văn bản y tế tiếng Việt"""
    
    def __init__(self):
        # Từ điển thuật ngữ y tế tiếng Việt
        self.medical_terms = {
            # Các bệnh phổ biến
            'diseases': [
                'cảm cúm', 'viêm phổi', 'viêm gan', 'viêm não', 'viêm màng não',
                'sốt xuất huyết', 'sốt rét', 'sốt phát ban', 'bệnh dại', 'bạch hầu',
                'ho gà', 'sởi', 'quai bị', 'thủy đậu', 'zona', 'tay chân miệng',
                'tiêu chảy', 'kiết lỵ', 'tả', 'viêm ruột thừa', 'viêm dạ dày',
                'loét dạ dày', 'viêm đại tràng', 'viêm khớp', 'gout', 'lupus',
                'tiểu đường', 'huyết áp cao', 'tim mạch', 'đột quỵ', 'nhồi máu cơ tim',
                'ung thư', 'bướu', 'u nang', 'polyp', 'sỏi thận', 'sỏi mật',
                'viêm phế quản', 'hen suyễn', 'lao phổi', 'copd', 'viêm xoang',
                'viêm họng', 'viêm amidan', 'viêm tai giữa', 'viêm kết mạc',
                'đau nửa đầu', 'đau đầu căng thẳng', 'mất ngủ', 'trầm cảm',
                'lo âu', 'stress', 'rối loạn tâm lý', 'tự kỷ', 'adhd'
            ],
            
            # Triệu chứng
            'symptoms': [
                'sốt', 'ho', 'khó thở', 'đau ngực', 'đau bụng', 'đau đầu',
                'chóng mặt', 'buồn nôn', 'nôn', 'tiêu chảy', 'táo bón',
                'mệt mỏi', 'yếu', 'đau cơ', 'đau khớp', 'phát ban', 'ngứa',
                'chảy nước mũi', 'nghẹt mũi', 'hắt hơi', 'đau họng',
                'khàn tiếng', 'nuốt khó', 'ợ chua', 'đầy hơi', 'khó tiêu',
                'đau lưng', 'đau cổ', 'tê bì', 'run', 'co giật',
                'mất ý thức', 'hôn mê', 'suy giảm trí nhớ', 'lẫn',
                'mất ngủ', 'ác mông', 'lo lắng', 'buồn bã', 'cáu gắt'
            ],
            
            # Bộ phận cơ thể
            'body_parts': [
                'đầu', 'mặt', 'mắt', 'tai', 'mũi', 'miệng', 'răng', 'lưỡi',
                'cổ', 'vai', 'tay', 'cánh tay', 'cẳng tay', 'bàn tay', 'ngón tay',
                'ngực', 'lưng', 'bụng', 'hông', 'chân', 'đùi', 'cẳng chân',
                'bàn chân', 'ngón chân', 'tim', 'phổi', 'gan', 'thận', 'dạ dày',
                'ruột', 'túi mật', 'tuyến giáp', 'não', 'tủy sống', 'da'
            ],
            
            # Thuật ngữ điều trị
            'treatments': [
                'thuốc', 'kháng sinh', 'giảm đau', 'hạ sốt', 'chống viêm',
                'tiêm', 'uống', 'bôi', 'nhỏ', 'súc', 'xông', 'massage',
                'vật lý trị liệu', 'phẫu thuật', 'mổ', 'nội soi', 'sinh thiết',
                'xét nghiệm', 'chụp x-quang', 'siêu âm', 'ct scan', 'mri'
            ],
            
            # Thuật ngữ phòng ngừa
            'prevention': [
                'vắc xin', 'tiêm chủng', 'rửa tay', 'đeo khẩu trang',
                'cách ly', 'khử trùng', 'vệ sinh', 'dinh dưỡng', 'tập thể dục',
                'nghỉ ngơi', 'không hút thuốc', 'hạn chế rượu bia',
                'kiểm tra sức khỏe định kỳ', 'tầm soát', 'sàng lọc'
            ]
        }
        
        # Từ khóa để nhận dạng nguyên nhân
        self.cause_indicators = [
            'do', 'bởi', 'vì', 'gây ra bởi', 'gây ra do', 'nguyên nhân',
            'căn nguyên', 'tác nhân', 'vi khuẩn', 'virus', 'nấm', 'ký sinh trùng',
            'di truyền', 'môi trường', 'lối sống', 'stress', 'ô nhiễm'
        ]
        
        # Từ khóa để nhận dạng triệu chứng
        self.symptom_indicators = [
            'triệu chứng', 'biểu hiện', 'dấu hiệu', 'nhận biết', 'cảm giác',
            'cảm thấy', 'xuất hiện', 'có thể có', 'thường gặp', 'phổ biến'
        ]
        
        # Từ khóa để nhận dạng biến chứng
        self.complication_indicators = [
            'biến chứng', 'tai biến', 'hậu quả', 'ảnh hưởng', 'gây ra',
            'dẫn đến', 'có thể gây', 'nguy hiểm', 'nghiêm trọng', 'tử vong'
        ]
        
        # Từ khóa để nhận dạng điều trị
        self.treatment_indicators = [
            'điều trị', 'chữa trị', 'chữa', 'trị', 'thuốc', 'dùng thuốc',
            'uống thuốc', 'tiêm', 'phẫu thuật', 'mổ', 'can thiệp', 'liệu pháp'
        ]
        
        # Từ khóa để nhận dạng phòng ngừa
        self.prevention_indicators = [
            'phòng ngừa', 'phòng chống', 'ngăn ngừa', 'tránh', 'dự phòng',
            'vắc xin', 'tiêm chủng', 'bảo vệ', 'giảm nguy cơ', 'kiểm soát'
        ]
        
        # Pattern để xử lý số liệu và đơn vị
        self.measurement_patterns = [
            r'\d+\s*(?:độ|°C|mmHg|mg|ml|lít|kg|cm|m)',
            r'\d+\s*(?:ngày|tuần|tháng|năm)',
            r'\d+\s*(?:lần|viên|gói|thìa)',
            r'\d+[.,]\d+\s*(?:%|phần trăm|tỷ lệ)'
        ]
    
    def preprocess_text(self, text):
        """Tiền xử lý văn bản tiếng Việt cho lĩnh vực y tế"""
        if not text:
            return ""
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Chuẩn hóa dấu câu và khoảng trắng
        text = re.sub(r'\s+', ' ', text)  # Nhiều khoảng trắng thành 1
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)  # Giữ lại chữ cái tiếng Việt
        
        # Loại bỏ các từ không cần thiết
        stop_words = ['và', 'hoặc', 'cũng', 'như', 'là', 'có', 'được', 'sẽ', 'đã', 'này', 'đó']
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 1]
        
        return ' '.join(words)
    
    def extract_medical_entities(self, text):
        """Trích xuất các thực thể y tế từ văn bản"""
        entities = {
            'diseases': [],
            'symptoms': [],
            'body_parts': [],
            'treatments': [],
            'prevention_methods': [],
            'measurements': []
        }
        
        text_lower = text.lower()
        
        # Tìm bệnh
        for disease in self.medical_terms['diseases']:
            if disease in text_lower:
                entities['diseases'].append(disease)
        
        # Tìm triệu chứng
        for symptom in self.medical_terms['symptoms']:
            if symptom in text_lower:
                entities['symptoms'].append(symptom)
        
        # Tìm bộ phận cơ thể
        for part in self.medical_terms['body_parts']:
            if part in text_lower:
                entities['body_parts'].append(part)
        
        # Tìm phương pháp điều trị
        for treatment in self.medical_terms['treatments']:
            if treatment in text_lower:
                entities['treatments'].append(treatment)
        
        # Tìm phương pháp phòng ngừa
        for prevention in self.medical_terms['prevention']:
            if prevention in text_lower:
                entities['prevention_methods'].append(prevention)
        
        # Tìm số liệu và đơn vị
        for pattern in self.measurement_patterns:
            matches = re.findall(pattern, text)
            entities['measurements'].extend(matches)
        
        # Loại bỏ duplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def extract_structured_info(self, text):
        """Trích xuất thông tin có cấu trúc từ văn bản y tế"""
        info = {
            'causes': [],
            'symptoms': [],
            'complications': [],
            'treatments': [],
            'preventions': []
        }
        
        # Tách văn bản thành các câu
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            sentence_lower = sentence.lower()
            
            # Tìm nguyên nhân
            for indicator in self.cause_indicators:
                if indicator in sentence_lower:
                    # Trích xuất phần sau indicator
                    parts = sentence_lower.split(indicator, 1)
                    if len(parts) > 1:
                        cause = parts[1].strip()
                        if len(cause) > 5:
                            info['causes'].append(cause)
                    break
            
            # Tìm triệu chứng
            for indicator in self.symptom_indicators:
                if indicator in sentence_lower:
                    # Tìm các triệu chứng được liệt kê
                    symptom_part = sentence_lower.split(indicator, 1)
                    if len(symptom_part) > 1:
                        symptoms_text = symptom_part[1]
                        # Tách theo dấu phẩy hoặc "và"
                        symptoms = re.split(r'[,;]|và|hoặc', symptoms_text)
                        for symptom in symptoms:
                            symptom = symptom.strip()
                            if len(symptom) > 3 and len(symptom) < 100:
                                info['symptoms'].append(symptom)
                    break
            
            # Tìm biến chứng
            for indicator in self.complication_indicators:
                if indicator in sentence_lower:
                    comp_part = sentence_lower.split(indicator, 1)
                    if len(comp_part) > 1:
                        comp = comp_part[1].strip()
                        if len(comp) > 5:
                            info['complications'].append(comp)
                    break
            
            # Tìm điều trị
            for indicator in self.treatment_indicators:
                if indicator in sentence_lower:
                    treat_part = sentence_lower.split(indicator, 1)
                    if len(treat_part) > 1:
                        treatment = treat_part[1].strip()
                        if len(treatment) > 5:
                            info['treatments'].append(treatment)
                    break
            
            # Tìm phòng ngừa
            for indicator in self.prevention_indicators:
                if indicator in sentence_lower:
                    prev_part = sentence_lower.split(indicator, 1)
                    if len(prev_part) > 1:
                        prevention = prev_part[1].strip()
                        if len(prevention) > 5:
                            info['preventions'].append(prevention)
                    break
        
        # Làm sạch và loại bỏ duplicate
        for key in info:
            info[key] = list(set([item for item in info[key] if item]))
        
        return info
    
    def classify_disease_severity(self, text):
        """Phân loại mức độ nghiêm trọng của bệnh"""
        text_lower = text.lower()
        
        severe_keywords = [
            'nguy hiểm', 'nghiêm trọng', 'tử vong', 'cấp cứu', 'nặng',
            'ác tính', 'di căn', 'giai đoạn cuối', 'không thể chữa khỏi'
        ]
        
        moderate_keywords = [
            'trung bình', 'vừa phải', 'có thể điều trị', 'kiểm soát được',
            'mãn tính', 'tái phát', 'cần theo dõi'
        ]
        
        mild_keywords = [
            'nhẹ', 'đơn giản', 'dễ chữa', 'tự khỏi', 'không nguy hiểm',
            'thường gặp', 'bình thường', 'tạm thời'
        ]
        
        severe_count = sum(1 for keyword in severe_keywords if keyword in text_lower)
        moderate_count = sum(1 for keyword in moderate_keywords if keyword in text_lower)
        mild_count = sum(1 for keyword in mild_keywords if keyword in text_lower)
        
        if severe_count > 0:
            return 'severe'
        elif moderate_count > 0:
            return 'moderate'
        elif mild_count > 0:
            return 'mild'
        else:
            return 'unknown'
    
    def extract_age_demographics(self, text):
        """Trích xuất thông tin về nhóm tuổi bị ảnh hưởng"""
        age_patterns = [
            (r'trẻ em|trẻ nhỏ|em bé|bé|nhi', 'children'),
            (r'người lớn|người trưởng thành|thanh niên', 'adults'),
            (r'người già|người cao tuổi|lão nhân', 'elderly'),
            (r'thai phụ|phụ nữ mang thai|bà bầu', 'pregnant_women'),
            (r'phụ nữ|nữ giới', 'women'),
            (r'nam giới|đàn ông', 'men'),
            (r'(\d+)\s*-\s*(\d+)\s*tuổi', 'age_range'),
            (r'dưới\s*(\d+)\s*tuổi', 'under_age'),
            (r'trên\s*(\d+)\s*tuổi', 'over_age')
        ]
        
        demographics = []
        text_lower = text.lower()
        
        for pattern, demo_type in age_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if demo_type == 'age_range':
                    demographics.append(f'{demo_type}_{match.group(1)}_{match.group(2)}')
                elif demo_type in ['under_age', 'over_age']:
                    demographics.append(f'{demo_type}_{match.group(1)}')
                else:
                    demographics.append(demo_type)
        
        return list(set(demographics))
    
    def extract_transmission_info(self, text):
        """Trích xuất thông tin về cách lây truyền bệnh"""
        transmission_patterns = [
            (r'lây qua đường hô hấp|lây qua hơi thở|lây qua không khí', 'airborne'),
            (r'lây qua tiếp xúc|lây qua da|chạm vào', 'contact'),
            (r'lây qua đường tình dục|quan hệ tình dục', 'sexual'),
            (r'lây qua máu|truyền máu|kim tiêm', 'blood'),
            (r'lây qua nước|đường nước|uống nước', 'water'),
            (r'lây qua thức ăn|đường ăn uống|thực phẩm', 'food'),
            (r'muỗi cắn|côn trùng|vector', 'vector'),
            (r'từ mẹ sang con|lây thẳng đứng|thai kỳ', 'vertical')
        ]
        
        transmission_methods = []
        text_lower = text.lower()
        
        for pattern, method in transmission_patterns:
            if re.search(pattern, text_lower):
                transmission_methods.append(method)
        
        return list(set(transmission_methods))
    
    def process_medical_text(self, text):
        """Xử lý toàn diện văn bản y tế tiếng Việt"""
        result = {
            'preprocessed_text': self.preprocess_text(text),
            'medical_entities': self.extract_medical_entities(text),
            'structured_info': self.extract_structured_info(text),
            'severity': self.classify_disease_severity(text),
            'demographics': self.extract_age_demographics(text),
            'transmission': self.extract_transmission_info(text),
            'is_contagious': self.is_contagious_disease(text)
        }
        
        return result
    
    def is_contagious_disease(self, text):
        """Xác định bệnh có lây nhiễm không"""
        contagious_indicators = [
            'lây', 'truyền nhiễm', 'dịch bệnh', 'lan rộng', 'bùng phát',
            'virus', 'vi khuẩn', 'vi-rút', 'bacteria', 'nhiễm trùng',
            'cách ly', 'phong tỏa', 'tiếp xúc gần', 'đeo khẩu trang'
        ]
        
        non_contagious_indicators = [
            'không lây', 'không truyền nhiễm', 'di truyền', 'ung thư',
            'tim mạch', 'tiểu đường', 'thoái hóa', 'lão hóa', 'chấn thương'
        ]
        
        text_lower = text.lower()
        
        # Tính điểm cho lây nhiễm
        contagious_score = sum(1 for indicator in contagious_indicators if indicator in text_lower)
        non_contagious_score = sum(1 for indicator in non_contagious_indicators if indicator in text_lower)
        
        if contagious_score > non_contagious_score:
            return True
        elif non_contagious_score > contagious_score:
            return False
        else:
            return None  # Không xác định được
    
    def suggest_related_terms(self, query):
        """Đề xuất các thuật ngữ liên quan"""
        query_lower = query.lower()
        suggestions = {
            'diseases': [],
            'symptoms': [],
            'treatments': [],
            'prevention': []
        }
        
        # Tìm các thuật ngữ tương tự
        for category, terms in self.medical_terms.items():
            if category in suggestions:
                for term in terms:
                    # Kiểm tra độ tương đồng đơn giản
                    if any(word in term for word in query_lower.split()) or \
                       any(word in query_lower for word in term.split()):
                        suggestions[category].append(term)
        
        return suggestions