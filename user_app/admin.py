from django.contrib import admin
from user_app.models import Record, Station,LicensePlate,Vehicle

# Register your models here.
admin.site.register(Record)
admin.site.register(Station)
admin.site.register(Vehicle)
admin.site.register(LicensePlate)

