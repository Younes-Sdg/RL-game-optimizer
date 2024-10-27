import React, { useState } from 'react';
import { Workflow, Play } from 'lucide-react';
import './App.css';

const App = () => {
  const [input, setInput] = useState('');
  const [darkMode, setDarkMode] = useState(false);

  const scenarios = [
    "I'm in a bidding war with 2 competitors for a contract worth $100,000",
    "Two companies are competing for market share in a new product launch",
    "A group of friends is deciding on a vacation destination with different preferences",
    
  ];

  const handleAnalyze = () => {};

  const handleScenarioClick = (scenario) => {
    setInput(scenario);
  };

  const toggleDarkMode = () => {
    setDarkMode((prev) => !prev);
  };

  return (
    <div className={`container ${darkMode ? 'dark' : ''}`}>
      <div className="header">
        <Workflow className="lucide" style={{ color: '#3b82f6', width: '24px', height: '24px' }} />
        <h1>Strategy Optimizer</h1>
        <label className="toggle-slider">
          <input type="checkbox" checked={darkMode} onChange={toggleDarkMode} />
          <span className="slider"></span>
        </label>
        <span style={{ marginLeft: '10px' }}>{darkMode ? 'Dark Mode' : 'Light Mode'}</span>
      </div>

      <textarea
        placeholder="Describe your game scenario..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />

      <button className="analyze-button" onClick={handleAnalyze}>
        <Play className="w-4 h-4" />
        Analyze Game
      </button>

      <div className="suggestions-section">
        <h2>Try these scenarios:</h2>
        <div className="suggestions-grid">
          {scenarios.map((scenario, index) => (
            <button
              key={index}
              className="suggestion-button"
              onClick={() => handleScenarioClick(scenario)}
            >
              {scenario}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default App;
