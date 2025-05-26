from django.contrib import admin
from .models import Disease, Symptom, DiseaseSymptom, Complication, Treatment, Prevention, Vaccine, ChatSession, ChatMessage, URLSource

class DiseaseSymptomInline(admin.TabularInline):
    model = DiseaseSymptom
    extra = 1

class DiseaseAdmin(admin.ModelAdmin):
    inlines = [DiseaseSymptomInline]
    list_display = ('name', 'is_contagious', 'source_url')
    search_fields = ['name', 'description']
    list_filter = ('is_contagious',)

class SymptomAdmin(admin.ModelAdmin):
    list_display = ('name',)
    search_fields = ['name', 'description']

class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ('sender', 'message', 'timestamp')
    can_delete = False

class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'created_at', 'updated_at')
    inlines = [ChatMessageInline]
    readonly_fields = ('session_id', 'created_at', 'updated_at')

class URLSourceAdmin(admin.ModelAdmin):
    list_display = ('url', 'last_updated', 'success_count', 'active')
    list_filter = ('active', 'last_updated')
    search_fields = ('url',)
    actions = ['update_from_urls']
    
    def update_from_urls(self, request, queryset):
        from .nlp_processor import NLPProcessor
        processor = NLPProcessor()
        
        urls = [source.url for source in queryset.filter(active=True)]
        count = processor.fetch_and_update_knowledge_base(urls)
        
        self.message_user(request, f"Đã cập nhật thành công {count} bệnh từ {len(urls)} URL.")
    
    update_from_urls.short_description = "Cập nhật dữ liệu từ các URL đã chọn"

# Đăng ký model mới
admin.site.register(URLSource, URLSourceAdmin)

admin.site.register(Disease, DiseaseAdmin)
admin.site.register(Symptom, SymptomAdmin)
admin.site.register(Complication)
admin.site.register(Treatment)
admin.site.register(Prevention)
admin.site.register(Vaccine)
admin.site.register(ChatSession, ChatSessionAdmin)