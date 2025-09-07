from django.urls import path
from .views import predict_pest

urlpatterns = [
    path('predict/', predict_pest, name='predict_pest'),
]
