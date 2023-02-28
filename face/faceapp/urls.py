from django.urls import path,include
from . import views
urlpatterns = [path('',views.index1,name='index'),
path('take/',views.take,name='take'),
path('takeim/',views.TrackImages,name='takeim'),
path('takee/',views.Export,name='takee')]