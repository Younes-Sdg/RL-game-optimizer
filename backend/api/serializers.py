from rest_framework import serializers
from .models import GameScenario

class GameScenarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = GameScenario
        fields = ['id','description']