from django.db import models

# Create your models here.
class Student(models.Model):
    rId=models.IntegerField( null=True, blank=True)
    name=models.CharField(max_length=200, null=True, blank=True)
    
    
       
     
class Attendence(models.Model):
     aid=models.IntegerField( null=True, blank=True)
     name=models.CharField(max_length=200, null=True, blank=True)
     Date=models.CharField(max_length=100, null=True, blank=True)
     Status=models.CharField(max_length=20, null=True, blank=True)