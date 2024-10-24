from rest_framework import viewsets
from .models import GameScenario
from .serializers import GameScenarioSerializer
from django.http import HttpResponse

class GameScenarioViewSet(viewsets.ModelViewSet):
    queryset = GameScenario.objects.all()
    serializer_class = GameScenarioSerializer


def home_view(request):

    return HttpResponse("welcome to the game optimizer API.")
