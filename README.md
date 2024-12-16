# Modeling Judgements in Sequential Events

This repo contains the code for our 9.66 final project. Our project looks at how people make predictions in sequential events, comparing responses between random processes (coin flips) and skill-based processes (basketball shots).

## Project Overview

The project investigates two main questions:
1. How do people's prior beliefs about a process (luck-based vs. skill-based) influence their predictions about future outcomes?
2. How do these beliefs influence their ability to classify sequences as originating from a luck-based or skill-based process?

The study combines empirical data collection with computational modeling using WebPPL to understand human judgment patterns in sequential events.

## Repository Structure

```
project/
├── WebPPL_setup/
│   └── test.wppl           # Test file for WebPPL setup verification
├── data/
│   └── get_data.py         # Script to process experimental data into CSVs
├── models/
│   └── webppl/
│       └── base_model.wppl # Main WebPPL model implementation
├── analysis/
│   └── analyze_results.py  # Analysis script for model and human data comparison
├── results/                # Generated results (gitignored)
│   ├── sequential_predictions.csv
│   ├── domain_classification.csv
│   ├── skill_ratings.csv
│   ├── model_predictions.json
│   └── *.png              # Generated visualization plots
└── package.json           # Node.js package configuration
```

## Prerequisites

- Node.js (for running WebPPL)
- Python 3.x
- WebPPL (`npm install -g webppl`)

Python packages required:
```
pandas
numpy
matplotlib
seaborn
scipy
```

## Setup and Installation

1. Clone this repository
2. Install Node.js dependencies:
   ```bash
   npm install
   ```
3. Install Python dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy
   ```

## Running the Code

1. First, verify WebPPL is working correctly:
   ```bash
   cd WebPPL_setup
   webppl test.wppl
   ```

2. Generate the experimental data (input human study results in this code, then run it to turn it into appropriate CSV files):
   ```bash
   python data/get_data.py
   ```

3. Run the WebPPL model:
   ```bash
   cd models/webppl
   webppl base_model.wppl
   ```

4. Run the analysis script (after copying JSON string output from base_model.wppl to results/model_predictions.json):
   ```bash
   python analysis/analyze_results.py
   ```

The analysis script will generate several visualizations in the `results` directory:
- `streak_analysis.png`: Analysis of how streak length affects predictions
- `transition_analysis.png`: Analysis of prediction changes based on previous outcomes
- `sequence_classification.png`: Comparison of sequence classification results
- `streak_correlation.png`: Correlation between streak length and predictions (not used in report)

## Model Details

The computational model implements two key cognitive processes:
- Predicting the next outcome in a sequence
- Classifying sequences as luck-based (coin flips) or skill-based (basketball shots)

The model uses Bayesian principles and incorporates:
- Beta distributions for modeling beliefs about success probability
- Streak sensitivity parameters
- Domain-specific priors for coins vs. basketball

## Data Structure

The experimental data is processed into three main CSV files:
1. `sequential_predictions.csv`: Contains sequential prediction data
2. `domain_classification.csv`: Contains sequence classification results
3. `skill_ratings.csv`: Contains participant ratings of skill vs. luck (not used in final report)
