from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass

class RLScenarioClassifierLight:
    def __init__(self):
        # Initialize model and tokenizer
        self.model_name = "microsoft/deberta-v3-small"  
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            device_map="auto"
        )
        
        # Define core RL characteristics
        self.rl_characteristics = {
            "sequential_decisions": {
                "keywords": ["sequence", "steps", "over time", "continuous", "actions"],
                "weight": 0.25,
                "required": True,
                "prompt_description": "whether decisions need to be made sequentially over time"
            },
            "state_space": {
                "keywords": ["state", "condition", "situation", "environment", "context"],
                "weight": 0.2,
                "required": True,
                "prompt_description": "what variables describe the situation at each moment"
            },
            "action_space": {
                "keywords": ["action", "decision", "choice", "control", "strategy"],
                "weight": 0.2,
                "required": True,
                "prompt_description": "what actions or choices can be made"
            },
            "reward_mechanism": {
                "keywords": ["reward", "feedback", "score", "outcome", "performance"],
                "weight": 0.2,
                "required": True,
                "prompt_description": "how success or performance is measured"
            }
        }

    def generate_focused_prompt(self, scenario_text: str, missing_characteristic: str) -> str:
        """
        Generate a focused prompt for a single characteristic
        """
        char_info = self.rl_characteristics[missing_characteristic]
        
        prompt = f"""Analyze this scenario for reinforcement learning:

Scenario: {scenario_text}

Focus on: {char_info['prompt_description']}

Answer these questions:
1. What could be the {missing_characteristic} in this scenario?
2. How confident are you in this inference (high/medium/low)?
3. Why did you make this inference?

Format: Inference|Confidence|Reasoning"""

        return prompt

    def analyze_single_characteristic(self, scenario_text: str, characteristic: str) -> Dict:
        """
        Use the LLM to analyze a single characteristic
        """
        prompt = self.generate_focused_prompt(scenario_text, characteristic)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512).to(self.model.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=200,
            temperature=0.7,
            num_return_sequences=1
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response
        try:
            inference, confidence, reasoning = response.split("|")
            confidence_score = {
                "high": 0.9,
                "medium": 0.6,
                "low": 0.3
            }.get(confidence.lower().strip(), 0.5)
        except:
            # Fallback if parsing fails
            return {
                "inference": response,
                "confidence": 0.5,
                "reasoning": "Parsing failed"
            }
            
        return {
            "inference": inference.strip(),
            "confidence": confidence_score,
            "reasoning": reasoning.strip()
        }

    def llm_analysis(self, scenario_text: str, initial_analysis: Dict) -> Dict:
        """
        Analyze missing characteristics using the LLM
        """
        inferred_parameters = {}
        confidence_scores = {}
        reasoning = {}
        
        # Analyze each missing characteristic individually
        for characteristic in initial_analysis["missing_characteristics"]:
            result = self.analyze_single_characteristic(scenario_text, characteristic)
            inferred_parameters[characteristic] = result["inference"]
            confidence_scores[characteristic] = result["confidence"]
            reasoning[characteristic] = result["reasoning"]
        
        # Calculate overall confidence
        avg_confidence = (
            sum(confidence_scores.values()) / len(confidence_scores)
            if confidence_scores else 0.0
        )
        
        return {
            "inferred_parameters": inferred_parameters,
            "confidence_scores": confidence_scores,
            "reasoning": reasoning,
            "llm_confidence": avg_confidence
        }

    def initial_filter(self, scenario_text: str) -> Dict:
        """
        Initial rule-based filter
        """
        analysis = {
            "characteristic_scores": {},
            "missing_characteristics": [],
            "preliminary_score": 0.0
        }
        
        scenario_lower = scenario_text.lower()
        
        for char_name, char_info in self.rl_characteristics.items():
            keyword_matches = sum(1 for keyword in char_info["keywords"] 
                                if keyword in scenario_lower)
            score = min(keyword_matches / len(char_info["keywords"]), 1.0)
            weighted_score = score * char_info["weight"]
            
            analysis["characteristic_scores"][char_name] = weighted_score
            
            if char_info["required"] and score < 0.3:
                analysis["missing_characteristics"].append(char_name)
            
            analysis["preliminary_score"] += weighted_score
        
        return analysis

    def generate_suggestions(self, scenario_text: str, analysis: Dict) -> List[str]:
        """
        Generate improvement suggestions based on analysis
        """
        suggestions = []
        
        # Add suggestions based on missing characteristics
        for char in analysis["initial_analysis"]["missing_characteristics"]:
            char_info = self.rl_characteristics[char]
            suggestions.append(
                f"Consider clarifying {char_info['prompt_description']}"
            )
        
        # Add suggestions based on confidence scores
        low_confidence = [
            char for char, score in analysis["llm_analysis"]["confidence_scores"].items()
            if score < 0.6
        ]
        for char in low_confidence:
            suggestions.append(
                f"Provide more details about {self.rl_characteristics[char]['prompt_description']}"
            )
        
        return suggestions

    def classify_scenario(self, scenario_text: str) -> Dict:
        """
        Main classification pipeline
        """
        # Initial rule-based analysis
        initial_results = self.initial_filter(scenario_text)
        
        # LLM analysis if needed
        llm_results = self.llm_analysis(
            scenario_text, 
            initial_results
        ) if initial_results["missing_characteristics"] else {
            "inferred_parameters": {},
            "confidence_scores": {},
            "reasoning": {},
            "llm_confidence": 1.0
        }
        
        # Combine scores
        combined_confidence = (
            initial_results["preliminary_score"] * 0.6 +
            llm_results["llm_confidence"] * 0.4
        )
        
        # Generate suggestions
        suggestions = self.generate_suggestions(
            scenario_text,
            {"initial_analysis": initial_results, "llm_analysis": llm_results}
        )
        
        return {
            "is_suitable": combined_confidence >= 0.6,
            "confidence": combined_confidence,
            "initial_analysis": initial_results,
            "llm_analysis": llm_results,
            "suggestions": suggestions,
            "final_classification": (
                "RL-Suitable" if combined_confidence >= 0.6 
                else "RL-Unsuitable"
            )
        }

# Example usage
if __name__ == "__main__":
    classifier = RLScenarioClassifierLight()
    
    # Test scenarios
    scenarios = [
        """
        A robot needs to navigate through a warehouse, picking up items 
        while avoiding obstacles. The environment layout changes frequently.
        """,
        """
        A system needs to optimize the temperature of a room based on 
        occupancy and time of day, minimizing energy usage while 
        maintaining comfort.
        """
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}:")
        print("-" * 50)
        print(scenario.strip())
        print("-" * 50)
        
        result = classifier.classify_scenario(scenario)
        
        print(f"Classification: {result['final_classification']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        if result['llm_analysis']['inferred_parameters']:
            print("\nInferred Parameters:")
            for param, value in result['llm_analysis']['inferred_parameters'].items():
                confidence = result['llm_analysis']['confidence_scores'][param]
                print(f"- {param}: {value} (confidence: {confidence:.2f})")
        
        print("\nSuggestions:")
        for suggestion in result['suggestions']:
            print(f"- {suggestion}")