from django.db import models

class Disease(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    causes = models.TextField(blank=True)
    is_contagious = models.BooleanField(default=False)
    source_url = models.URLField(blank=True, null=True)  # Trường URL nguồn
    
    def __str__(self):
        return self.name

class Symptom(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    
    def __str__(self):
        return self.name

class DiseaseSymptom(models.Model):
    disease = models.ForeignKey(Disease, on_delete=models.CASCADE, related_name='symptoms_link')
    symptom = models.ForeignKey(Symptom, on_delete=models.CASCADE, related_name='diseases_link')
    relevance_score = models.IntegerField(default=1)  # 1-10 để chỉ mức độ quan trọng của triệu chứng
    
    class Meta:
        unique_together = ('disease', 'symptom')

class Complication(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    diseases = models.ManyToManyField(Disease, related_name='complications')
    
    def __str__(self):
        return self.name

class Treatment(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    diseases = models.ManyToManyField(Disease, related_name='treatments')
    
    def __str__(self):
        return self.name

class Prevention(models.Model):
    method = models.CharField(max_length=200)
    description = models.TextField()
    diseases = models.ManyToManyField(Disease, related_name='preventions')
    
    def __str__(self):
        return self.method

class Vaccine(models.Model):
    name = models.CharField(max_length=200)
    manufacturer = models.CharField(max_length=200, blank=True)
    diseases = models.ManyToManyField(Disease, related_name='vaccines')
    
    def __str__(self):
        return self.name

class ChatSession(models.Model):
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Session {self.session_id}"

class ChatMessage(models.Model):
    SENDER_CHOICES = [
        ('user', 'User'),
        ('bot', 'Bot'),
    ]
    
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    sender = models.CharField(max_length=10, choices=SENDER_CHOICES)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.sender}: {self.message[:30]}..."
class URLSource(models.Model):
    url = models.URLField(unique=True)
    last_updated = models.DateTimeField(null=True, blank=True)
    success_count = models.IntegerField(default=0)
    active = models.BooleanField(default=True)
    
    def __str__(self):
        return self.url