from django.urls import path
from . import views

app_name = 'patients'

urlpatterns = [
    path('', views.patient_list, name='list'),
    path('<str:patient_id>/', views.patient_detail, name='detail'),
]