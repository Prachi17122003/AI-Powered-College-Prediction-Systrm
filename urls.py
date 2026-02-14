from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('predictor/', views.predictor, name='predictor'),
    path('compare/', views.compare, name='compare'),
    path('results/', views.results, name='results'),
]