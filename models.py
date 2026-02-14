from django.db import models
from django.contrib.auth.models import User

class College(models.Model):
    """Model representing a college/university"""
    name = models.CharField(max_length=200)
    location = models.CharField(max_length=100)
    tier = models.CharField(max_length=50, choices=[
        ('Ivy League', 'Ivy League'),
        ('Top Tier', 'Top Tier'),
        ('Public Ivy', 'Public Ivy'),
        ('Top Public', 'Top Public'),
        ('Good Public', 'Good Public'),
        ('Private', 'Private'),
        ('Community', 'Community College')
    ])
    acceptance_rate = models.FloatField(default=0.0)
    avg_sat_score = models.IntegerField(default=0)
    avg_act_score = models.IntegerField(default=0)
    tuition_in_state = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    tuition_out_state = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    description = models.TextField(blank=True)
    website = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name

class Prediction(models.Model):
    """Model to store college predictions"""
    student_data = models.JSONField(help_text="Student profile data used for prediction")
    predicted_colleges = models.JSONField(help_text="Predicted colleges with probabilities")
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Prediction {self.id} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

class ContactMessage(models.Model):
    """Model to store contact form messages"""
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Message from {self.name} - {self.created_at.strftime('%Y-%m-%d')}"

class PredictionInput(models.Model):
    """Stores raw prediction input values per submission"""
    course = models.CharField(max_length=200, blank=True)
    exam = models.CharField(max_length=100, blank=True)
    rank = models.IntegerField(default=0)
    category = models.CharField(max_length=100, blank=True)
    gender = models.CharField(max_length=50, blank=True)
    domicile = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"PredictionInput {self.id} - {self.course} - {self.rank}"