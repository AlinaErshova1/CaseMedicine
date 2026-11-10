from django.shortcuts import render


def patient_list(request):
    return render(request, 'patients/list.html')


def patient_detail(request, patient_id):
    return render(request, 'patients/detail.html', {'patient_id': patient_id})