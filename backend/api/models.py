from django.db import models

# Create your models here.

class GameScenario(models.Model):
    description = models.CharField(max_length=400)

