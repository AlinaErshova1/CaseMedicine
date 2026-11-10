from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.login_view, name='login'),
    path('login/', views.login_view, name='login'),
    path('admin-panel/doctors/', views.admin_doctors, name='admin_doctors'),
    path('admin-panel/patients/', views.admin_patients, name='admin_patients'),
]