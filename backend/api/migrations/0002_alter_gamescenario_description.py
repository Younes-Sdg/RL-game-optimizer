# Generated by Django 5.1.2 on 2024-10-23 22:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='gamescenario',
            name='description',
            field=models.CharField(max_length=400),
        ),
    ]
