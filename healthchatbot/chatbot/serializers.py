# chatbot/serializers.py
from rest_framework import serializers
from .models import Disease, Symptom, ChatSession, ChatMessage

class SymptomSerializer(serializers.ModelSerializer):
    class Meta:
        model = Symptom
        fields = ['id', 'name', 'description']

class DiseaseSerializer(serializers.ModelSerializer):
    symptoms = serializers.SerializerMethodField()
    complications = serializers.StringRelatedField(many=True)
    treatments = serializers.StringRelatedField(many=True)
    preventions = serializers.StringRelatedField(many=True)
    vaccines = serializers.StringRelatedField(many=True)
    
    class Meta:
        model = Disease
        fields = ['id', 'name', 'description', 'causes', 'is_contagious', 
                 'symptoms', 'complications', 'treatments', 'preventions', 'vaccines', 'source_url']
    
    def get_symptoms(self, obj):
        symptom_links = obj.symptoms_link.all()
        return [
            {
                'id': link.symptom.id,
                'name': link.symptom.name,
                'relevance': link.relevance_score
            }
            for link in symptom_links
        ]

class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ['id', 'sender', 'message', 'timestamp']

class ChatSessionSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = ChatSession
        fields = ['id', 'session_id', 'created_at', 'updated_at', 'messages']