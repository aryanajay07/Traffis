# Generated by Django 5.0.6 on 2024-06-28 12:39

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Station",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("areacode", models.PositiveIntegerField()),
                ("location", models.CharField(max_length=80)),
                ("mac_address", models.CharField(max_length=17)),
            ],
        ),
        migrations.CreateModel(
            name="Record",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("speed", models.IntegerField()),
                ("date", models.DateField()),
                ("count", models.IntegerField()),
                ("licenseplate_no", models.CharField(max_length=50, null=True)),
                (
                    "stationID",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="user_app.station",
                    ),
                ),
            ],
        ),
    ]
