import requests
import logging
from datetime import datetime, timedelta
from django.utils import timezone
from django.db import transaction
from .models import URLSource, Disease, Symptom
from .vietnamese_medical_processor import VietnameseMedicalProcessor
from .nlp_processor import ImprovedNLPProcessor

logger = logging.getLogger(__name__)

class HealthSourceManager:
    """Quản lý toàn diện các nguồn thông tin y tế"""
    
    def __init__(self):
        self.medical_processor = VietnameseMedicalProcessor()
        self.nlp_processor = ImprovedNLPProcessor()
        
        # Danh sách nguồn đáng tin cậy cho y tế Việt Nam
        self.trusted_sources = {
            'government': [
                {
                    'url': 'https://moh.gov.vn/tin-tuc-su-kien',
                    'name': 'Bộ Y tế Việt Nam',
                    'reliability': 5,
                    'update_frequency': 'daily',
                    'content_types': ['news', 'guidelines', 'health_alerts']
                },
                {
                    'url': 'https://kcb.vn/',
                    'name': 'Cổng thông tin khám chữa bệnh',
                    'reliability': 5,
                    'update_frequency': 'daily',
                    'content_types': ['healthcare_facilities', 'medical_procedures']
                }
            ],
            'medical_centers': [
                {
                    'url': 'https://vnvc.vn/cac-benh-truyen-nhiem-thuong-gap/',
                    'name': 'VNVC - Trung tâm tiêm chủng',
                    'reliability': 4,
                    'update_frequency': 'weekly',
                    'content_types': ['infectious_diseases', 'vaccines', 'prevention']
                },
                {
                    'url': 'https://www.vinmec.com/vi/bai-viet/cac-benh-truyen-nhiem-thuong-gap-175',
                    'name': 'Vinmec International Hospital',
                    'reliability': 4,
                    'update_frequency': 'weekly',
                    'content_types': ['diseases', 'treatments', 'health_tips']
                },
                {
                    'url': 'https://benhvienbachmai.vn/tin-tuc-su-kien/',
                    'name': 'Bệnh viện Bạch Mai',
                    'reliability': 4,
                    'update_frequency': 'weekly',
                    'content_types': ['medical_news', 'treatments', 'specialties']
                },
                {
                    'url': 'https://www.108.vn/benh-vien/danh-sach-benh-vien',
                    'name': 'Bệnh viện Trung ương Quân đội 108',
                    'reliability': 4,
                    'update_frequency': 'weekly',
                    'content_types': ['medical_services', 'health_information']
                }
            ],
            'health_portals': [
                {
                    'url': 'https://suckhoedoisong.vn/benh-truyen-nhiem/',
                    'name': 'Sức khỏe đời sống',
                    'reliability': 3,
                    'update_frequency': 'daily',
                    'content_types': ['health_news', 'lifestyle', 'disease_info']
                },
                {
                    'url': 'https://hellobacsi.com/benh-truyen-nhiem/',
                    'name': 'Hello Bacsi',
                    'reliability': 3,
                    'update_frequency': 'daily',
                    'content_types': ['health_articles', 'medical_advice', 'symptoms']
                },
                {
                    'url': 'https://nhathuoclongchau.com.vn/bai-viet/cac-benh-truyen-nhiem-thuong-gap.html',
                    'name': 'Nhà thuốc Long Châu',
                    'reliability': 3,
                    'update_frequency': 'weekly',
                    'content_types': ['medications', 'health_tips', 'disease_prevention']
                }
            ],
            'international': [
                {
                    'url': 'https://medlineplus.gov/languages/vietnamese.html',
                    'name': 'MedlinePlus Vietnamese',
                    'reliability': 5,
                    'update_frequency': 'weekly',
                    'content_types': ['medical_encyclopedia', 'health_topics', 'drug_information']
                },
                {
                    'url': 'https://www.who.int/vietnam/health-topics',
                    'name': 'WHO Vietnam',
                    'reliability': 5,
                    'update_frequency': 'weekly',
                    'content_types': ['global_health', 'disease_outbreaks', 'health_guidelines']
                }
            ]
        }
    
    def validate_source_quality(self, url, content=None):
        """Đánh giá chất lượng nguồn thông tin"""
        quality_score = 0
        issues = []
        recommendations = []
        
        try:
            if not content:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; HealthBot/1.0; +https://example.com/bot)'
                }
                response = requests.get(url, headers=headers, timeout=15)
                content = response.text
            
            # Kiểm tra độ dài nội dung
            if len(content) < 1000:
                issues.append("Nội dung quá ngắn")
                quality_score -= 20
            else:
                quality_score += 10
            
            # Kiểm tra có phải nội dung y tế không
            medical_entities = self.medical_processor.extract_medical_entities(content)
            total_entities = sum(len(entities) for entities in medical_entities.values())
            
            if total_entities > 10:
                quality_score += 30
                recommendations.append("Nguồn chứa nhiều thông tin y tế hữu ích")
            elif total_entities > 5:
                quality_score += 15
            else:
                quality_score -= 10
                issues.append("Ít thông tin y tế")
            
            # Kiểm tra cấu trúc HTML
            if '<h1>' in content or '<h2>' in content:
                quality_score += 10
            if '<ul>' in content or '<ol>' in content:
                quality_score += 10
            
            # Kiểm tra có thông tin tác giả/nguồn không
            author_indicators = ['tác giả', 'bác sĩ', 'bs.', 'ths.', 'pgs.', 'gs.']
            if any(indicator in content.lower() for indicator in author_indicators):
                quality_score += 15
                recommendations.append("Có thông tin về tác giả/chuyên gia")
            
            # Kiểm tra ngày cập nhật
            date_patterns = [r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', r'\d{4}[/-]\d{1,2}[/-]\d{1,2}']
            has_date = any(re.search(pattern, content) for pattern in date_patterns)
            if has_date:
                quality_score += 10
                recommendations.append("Có thông tin về ngày đăng/cập nhật")
            
            # Xác định mức độ tin cậy
            if quality_score >= 70:
                reliability = 'high'
            elif quality_score >= 40:
                reliability = 'medium'
            else:
                reliability = 'low'
                issues.append("Chất lượng nội dung cần cải thiện")
            
            return {
                'quality_score': quality_score,
                'reliability': reliability,
                'issues': issues,
                'recommendations': recommendations,
                'medical_entities_count': total_entities
            }
            
        except Exception as e:
            logger.error(f"Error validating source quality: {e}")
            return {
                'quality_score': 0,
                'reliability': 'unknown',
                'issues': [f"Lỗi khi đánh giá: {str(e)}"],
                'recommendations': [],
                'medical_entities_count': 0
            }
    
    def auto_discover_health_sources(self, base_urls):
        """Tự động khám phá các nguồn y tế từ các URL cơ sở"""
        discovered_sources = []
        
        for base_url in base_urls:
            try:
                response = requests.get(base_url, timeout=15)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Tìm các link có chứa từ khóa y tế
                health_keywords = ['benh', 'suc-khoe', 'y-te', 'dieu-tri', 'phong-ngua', 'vaccine']
                
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    link_text = link.get_text().lower()
                    
                    # Kiểm tra xem link có liên quan đến y tế không
                    if any(keyword in href.lower() or keyword in link_text for keyword in health_keywords):
                        full_url = urljoin(base_url, href)
                        
                        # Validate URL
                        if self.is_valid_health_url(full_url):
                            discovered_sources.append({
                                'url': full_url,
                                'title': link.get_text().strip(),
                                'source_domain': urlparse(base_url).netloc
                            })
                
            except Exception as e:
                logger.error(f"Error discovering sources from {base_url}: {e}")
        
        return discovered_sources
    
    def is_valid_health_url(self, url):
        """Kiểm tra URL có hợp lệ cho nội dung y tế không"""
        try:
            parsed = urlparse(url)
            
            # Kiểm tra domain có đáng tin cậy không
            trusted_domains = [
                'moh.gov.vn', 'vinmec.com', 'vnvc.vn', 'benhvienbachmai.vn',
                'suckhoedoisong.vn', 'hellobacsi.com', 'longchau.com',
                'medlineplus.gov', 'who.int', 'cdc.gov'
            ]
            
            domain = parsed.netloc.lower()
            if any(trusted in domain for trusted in trusted_domains):
                return True
            
            # Kiểm tra path có chứa từ khóa y tế không
            health_paths = ['benh', 'suc-khoe', 'y-te', 'health', 'medical', 'disease']
            path = parsed.path.lower()
            if any(keyword in path for keyword in health_paths):
                return True
            
            return False
            
        except Exception:
            return False
    
    def prioritize_sources_for_update(self):
        """Sắp xếp ưu tiên các nguồn để cập nhật"""
        sources = URLSource.objects.filter(active=True)
        
        prioritized = []
        for source in sources:
            priority_score = 0
            
            # Tính điểm dựa trên thời gian cập nhật cuối
            if source.last_updated:
                days_since_update = (timezone.now() - source.last_updated).days
                if days_since_update > 7:
                    priority_score += days_since_update
            else:
                priority_score += 100  # Chưa bao giờ cập nhật
            
            # Tính điểm dựa trên độ tin cậy
            domain = urlparse(source.url).netloc.lower()
            for category, sources_list in self.trusted_sources.items():
                for trusted_source in sources_list:
                    if domain in trusted_source['url']:
                        priority_score += trusted_source['reliability'] * 10
                        break
            
            # Tính điểm dựa trên thành công trước đó
            if source.success_count > 0:
                priority_score += source.success_count
            
            prioritized.append((source, priority_score))
        
        # Sắp xếp theo điểm ưu tiên giảm dần
        prioritized.sort(key=lambda x: x[1], reverse=True)
        
        return [source for source, score in prioritized]
    
    def batch_update_sources(self, max_sources=10):
        """Cập nhật hàng loạt các nguồn theo thứ tự ưu tiên"""
        prioritized_sources = self.prioritize_sources_for_update()[:max_sources]
        
        results = {
            'updated': 0,
            'failed': 0,
            'total_diseases': 0,
            'details': []
        }
        
        for source in prioritized_sources:
            try:
                logger.info(f"Updating source: {source.url}")
                
                # Cập nhật từ nguồn
                diseases_count = self.nlp_processor.fetch_and_update_knowledge_base([source.url])
                
                # Cập nhật thông tin nguồn
                source.last_updated = timezone.now()
                source.success_count = diseases_count
                source.save()
                
                results['updated'] += 1
                results['total_diseases'] += diseases_count
                results['details'].append({
                    'url': source.url,
                    'status': 'success',
                    'diseases_count': diseases_count
                })
                
                logger.info(f"Successfully updated {source.url}: {diseases_count} diseases")
                
            except Exception as e:
                logger.error(f"Failed to update {source.url}: {e}")
                results['failed'] += 1
                results['details'].append({
                    'url': source.url,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results
    
    def get_source_statistics(self):
        """Lấy thống kê về các nguồn"""
        sources = URLSource.objects.all()
        
        stats = {
            'total_sources': sources.count(),
            'active_sources': sources.filter(active=True).count(),
            'sources_by_reliability': {},
            'recent_updates': 0,
            'never_updated': 0,
            'total_diseases_from_sources': 0
        }
        
        # Thống kê theo độ tin cậy
        for source in sources:
            quality = self.get_source_reliability(source.url)
            if quality not in stats['sources_by_reliability']:
                stats['sources_by_reliability'][quality] = 0
            stats['sources_by_reliability'][quality] += 1
        
        # Thống kê cập nhật gần đây
        one_week_ago = timezone.now() - timedelta(days=7)
        stats['recent_updates'] = sources.filter(last_updated__gte=one_week_ago).count()
        stats['never_updated'] = sources.filter(last_updated__isnull=True).count()
        
        # Tổng số bệnh từ các nguồn
        stats['total_diseases_from_sources'] = sum(source.success_count for source in sources)
        
        return stats
    
    def get_source_reliability(self, url):
        """Lấy độ tin cậy của nguồn"""
        domain = urlparse(url).netloc.lower()
        
        for category, sources_list in self.trusted_sources.items():
            for source in sources_list:
                if domain in source['url']:
                    reliability = source['reliability']
                    if reliability >= 4:
                        return 'high'
                    elif reliability >= 3:
                        return 'medium'
                    else:
                        return 'low'
        
        return 'unknown'
    
    def suggest_new_sources(self, topic=None):
        """Đề xuất nguồn mới dựa trên chủ đề"""
        suggestions = []
        
        if topic:
            topic_lower = topic.lower()
            
            # Đề xuất dựa trên chủ đề
            if 'truyền nhiễm' in topic_lower or 'dịch bệnh' in topic_lower:
                suggestions.extend([
                    'https://vnvc.vn/tin-tuc-su-kien/',
                    'https://nihe.org.vn/',
                    'https://www.who.int/vietnam/emergencies'
                ])
            
            elif 'tim mạch' in topic_lower:
                suggestions.extend([
                    'https://timmachquocgia.vn/',
                    'https://www.vinmec.com/vi/tim-mach/'
                ])
            
            elif 'ung thư' in topic_lower:
                suggestions.extend([
                    'https://benhvienk.vn/',
                    'https://www.vinmec.com/vi/ung-buou/'
                ])
        
        # Đề xuất chung
        suggestions.extend([
            'https://www.benhvienchoray.vn/',
            'https://bvdaihoc.vn/',
            'https://medlatec.vn/tin-tuc-y-te'
        ])
        
        return list(set(suggestions))
    
    def cleanup_inactive_sources(self, days_threshold=30):
        """Dọn dẹp các nguồn không hoạt động"""
        threshold_date = timezone.now() - timedelta(days=days_threshold)
        
        # Tìm các nguồn lâu không cập nhật
        inactive_sources = URLSource.objects.filter(
            last_updated__lt=threshold_date,
            success_count=0
        )
        
        cleanup_report = {
            'candidates_for_removal': [],
            'removed_count': 0
        }
        
        for source in inactive_sources:
            # Kiểm tra lần cuối xem nguồn có còn hoạt động không
            try:
                response = requests.head(source.url, timeout=10)
                if response.status_code >= 400:
                    cleanup_report['candidates_for_removal'].append({
                        'url': source.url,
                        'reason': f'HTTP {response.status_code}',
                        'last_updated': source.last_updated
                    })
            except Exception as e:
                cleanup_report['candidates_for_removal'].append({
                    'url': source.url,
                    'reason': str(e),
                    'last_updated': source.last_updated
                })
        
        return cleanup_report