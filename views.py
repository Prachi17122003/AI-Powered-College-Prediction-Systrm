from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import College, Prediction, ContactMessage, PredictionInput
from .ml_model import ml_model
import json
import random

def index(request):
    """Home page view"""
    return render(request, 'index.html')

def about(request):
    """About page view"""
    return render(request, 'about.html')

def contact(request):
    """Contact page view"""
    if request.method == 'POST':
        # Handle contact form submission
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')
        
        # Save to database
        ContactMessage.objects.create(
            name=name,
            email=email,
            message=message
        )
        
        messages.success(request, 'Thank you for your message! We will get back to you soon.')
        return redirect('contact')
    return render(request, 'contact.html')

def login_view(request):
    """Login page view with database authentication"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        if username and password:
            # Authenticate user against database
            user = authenticate(request, username=username, password=password)
            if user is not None:
                # Login successful
                login(request, user)
                messages.success(request, f'Welcome back, {user.first_name or user.username}!')
                return redirect('index')
            else:
                # Invalid credentials
                messages.error(request, 'Invalid username or password. Please try again.')
        else:
            messages.error(request, 'Please fill in both username and password fields.')
    
    return render(request, 'login.html')

def logout_view(request):
    """Logout view"""
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('index')

def predictor(request):
    """College prediction form view"""
    if request.method == 'POST':
        # Handle prediction form submission
        try:
            # Get form data
            student_data = {
                'course': request.POST.get('course', ''),
                'category': request.POST.get('category', ''),
                'rank': request.POST.get('rank', ''),
                'percentile': request.POST.get('percentile', ''),
                'exam': request.POST.get('exam', ''),
                'gender': request.POST.get('gender', ''),
                'domicile': request.POST.get('domicile', '')
            }
            
            print(f"Student data received: {student_data}")
            
            # Use ML model for prediction (only 4 factors considered)
            ml_input = {
                'course': student_data['course'],
                'category': student_data['category'],
                'rank': student_data['rank'],
                'percentile': student_data['percentile']
            }
            predicted_colleges = ml_model.predict_colleges(ml_input)
            print(f"Predicted colleges: {predicted_colleges}")
            
            # Store results in session for immediate redirect display
            request.session['last_student_data'] = student_data
            request.session['last_colleges'] = predicted_colleges

            # Save prediction to database (associate with logged-in user if available)
            # JSONField expects Python objects; avoid double-encoding
            Prediction.objects.create(
                student_data=student_data,
                predicted_colleges=predicted_colleges,
                user=request.user if request.user.is_authenticated else None
            )

            # Also save raw inputs in a dedicated table for analytics/auditing
            # Save raw inputs (store all fields for analytics; model only used 4)
            PredictionInput.objects.create(
                course=student_data.get('course', ''),
                exam=student_data.get('exam', ''),
                rank=int(student_data.get('rank', 0) or 0),
                category=student_data.get('category', ''),
                gender=student_data.get('gender', ''),
                domicile=student_data.get('domicile', ''),
                user=request.user if request.user.is_authenticated else None
            )
            
            messages.success(request, 'Prediction completed successfully!')
            return redirect('results')
            
        except Exception as e:
            print(f"Error in predictor: {e}")
            messages.error(request, f'Error processing prediction: {str(e)}')
            return render(request, 'predictor.html')
    
    # For GET: populate dynamic dropdowns from dataset
    try:
        courses = ml_model.get_available_courses()
    except Exception:
        courses = []
    # Keep only main categories in the UI; model maps these to dataset-specific labels
    categories = ['General', 'OBC', 'SC', 'ST', 'EWS', 'PWD']
    return render(request, 'predictor.html', { 'courses': courses, 'categories': categories })

def compare(request):
    """College comparison view"""
    colleges = College.objects.all()[:10]  # Get some colleges for comparison
    return render(request, 'compare.html', {'colleges': colleges})

def results(request):
    """Results page view"""
    # Prefer session data right after redirect from predictor
    colleges = request.session.pop('last_colleges', None)
    if colleges:
        # Ensure vacant_seats exists for all colleges
        for college in colleges:
            if 'vacant_seats' not in college:
                college['vacant_seats'] = random.randint(1, 7)
        return render(request, 'results.html', {'colleges': colleges})

    # Fallback: Get the latest prediction from the database
    latest_prediction = Prediction.objects.last()
    if latest_prediction:
        stored = latest_prediction.predicted_colleges
        # Handle both dict/list and serialized JSON strings defensively
        if isinstance(stored, (list, dict)):
            colleges = stored
        else:
            try:
                colleges = json.loads(stored)
            except Exception:
                colleges = None
        
        # Ensure vacant_seats exists for all colleges
        if colleges:
            for college in colleges:
                if 'vacant_seats' not in college:
                    college['vacant_seats'] = random.randint(1, 7)
        
        return render(request, 'results.html', {'colleges': colleges})

    return render(request, 'results.html', {'colleges': None})

## Removed legacy non-ML fallback functions; predictions now come from ml_model

