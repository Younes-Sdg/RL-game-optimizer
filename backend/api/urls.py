from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import GameScenarioViewSet

router = DefaultRouter()
router.register(r'game_scenarios', GameScenarioViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
