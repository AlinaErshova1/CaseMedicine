from django.shortcuts import render

def admin_doctors(request):
    return render(request, 'admin/doctors.html')

def admin_patients(request):
    return render(request, 'patients/list.html')

def login_view(request):
    return render(request, 'login.html')