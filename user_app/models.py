from django.db import models
from django.utils import timezone

class Station(models.Model):
    areacode = models.PositiveIntegerField()
    location = models.CharField(max_length=80)
    mac_address = models.CharField(max_length=17)

    def __str__(self):
        return self.location

class Record(models.Model):
    stationID = models.ForeignKey('Station', on_delete=models.CASCADE)
    speed = models.IntegerField()
    date = models.DateField(default=timezone.now)
    count = models.IntegerField(default=1)
    licenseplate_no = models.CharField(max_length=50, null=True, blank=True)
    vehicle_image = models.ImageField(upload_to='Vehicle_images/', default=None, null=True, blank=True)
    license_plate_image = models.ImageField(upload_to='License_plate_images/', default=None, null=True, blank=True)
    vehicle_type = models.CharField(max_length=50, null=True, blank=True, default='unknown')
    detection_time = models.DateTimeField(default=timezone.now)
    max_speed = models.FloatField(null=True, blank=True, default=0.0)
    location = models.CharField(max_length=100, null=True, blank=True, default='unknown')
    direction = models.CharField(max_length=50, null=True, blank=True, default='unknown')
    plate_detection_time = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Record from {self.stationID}"
    

class Vehicle(models.Model):
    vehicle_id=models.IntegerField()
    vehicle_image=models.ImageField()

    def _str_(self):
        return self.vehicle_id
    
class LicensePlate(models.Model):
    plate_number = models.CharField(max_length=20, unique=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    vehicle_image = models.ImageField(upload_to='plates/', null=True, blank=True)  # Optional

    def __str__(self):
        return self.plate_number

   



    

