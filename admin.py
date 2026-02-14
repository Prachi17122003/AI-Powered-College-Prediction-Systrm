from django.contrib import admin
from .models import College, Prediction, ContactMessage

@admin.register(College)
class CollegeAdmin(admin.ModelAdmin):
    list_display = ['name', 'location', 'tier', 'acceptance_rate', 'avg_sat_score']
    list_filter = ['tier', 'location']
    search_fields = ['name', 'location']
    ordering = ['name']

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'created_at']
    list_filter = ['created_at']
    search_fields = ['user__username']
    readonly_fields = ['created_at']

@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'created_at', 'is_read']
    list_filter = ['is_read', 'created_at']
    search_fields = ['name', 'email']
    readonly_fields = ['created_at']
