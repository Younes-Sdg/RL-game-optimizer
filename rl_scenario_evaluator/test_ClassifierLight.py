import unittest
from rl_scenario_evaluator.scenario_classifier import RLScenarioClassifierLight

class TestRLScenarioClassifierLight(unittest.TestCase):
    
    def setUp(self):
        """Set up the RLScenarioClassifierLight instance for testing."""
        self.classifier = RLScenarioClassifierLight()

    def test_initialization(self):
        """Test if the model and tokenizer are initialized correctly."""
        self.assertIsNotNone(self.classifier.model)
        self.assertIsNotNone(self.classifier.tokenizer)

    def test_initial_filter(self):
        """Test the initial filtering of a scenario."""
        scenario = "A robot needs to navigate through a warehouse."
        analysis = self.classifier.initial_filter(scenario)
        self.assertIn("characteristic_scores", analysis)
        self.assertIn("missing_characteristics", analysis)

    def test_classify_scenario(self):
        """Test the classification of a scenario."""
        scenario = "A robot needs to navigate through a warehouse, picking up items."
        result = self.classifier.classify_scenario(scenario)
        self.assertIn("final_classification", result)

if __name__ == '__main__':
    unittest.main()
