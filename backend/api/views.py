# Create your views here.

from rest_framework import viewsets
from .models import GameScenario
from .serializers import GameScenarioSerializer

class GameScenarioViewSet(viewsets.ModelViewSet):
    queryset = GameScenario.objects.all()  # Retrieve all instances of GameScenario from the database
    serializer_class = GameScenarioSerializer  # Use the GameScenarioSerializer for converting data to and from JSON
