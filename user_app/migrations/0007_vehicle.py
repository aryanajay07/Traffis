# Generated by Django 4.2 on 2025-05-31 09:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user_app', '0006_licenseplate'),
    ]

    operations = [
        migrations.CreateModel(
            name='Vehicle',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('vehicle_id', models.IntegerField()),
                ('vehicle_image', models.ImageField(upload_to='')),
            ],
        ),
    ]
