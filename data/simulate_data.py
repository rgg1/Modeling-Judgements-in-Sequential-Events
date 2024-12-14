import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create simulated human data
example_data = {
    'participant_id': [],
    'domain': [],
    'sequence_length': [],
    'sequence': [],
    'prediction': [],
    'confidence': [],
    'reasoning': []
}

sequences = {
    'coin': {
        5: '[True, True, False, False, True]',
        10: '[True, False, True, False, False, False, True, True, True, False]'
    },
    'basketball': {
        5: '[True, False, True, True, True]',
        10: '[True, False, True, True, True, True, True, True, False, False]'
    },
    'weather': {
        5: '[True, True, True, True, False]',
        10: '[True, True, False, True, True, False, False, False, True, True]'
    }
}

# Common reasoning patterns
coin_reasons = [
    "Seems random, close to 50-50",
    "After several tails, heads is due",
    "Pattern suggests alternating outcomes",
    "Random but slightly favoring heads",
    "Previous outcomes don't matter, 50-50"
]

basketball_reasons = [
    "They're on a hot streak",
    "Seems like a skilled player",
    "Missing might affect confidence",
    "Strong pattern of success",
    "Player seems to recover well after misses"
]

weather_reasons = [
    "Weather tends to persist",
    "Change seems likely after several days",
    "Patterns suggest stable conditions",
    "Season probably affects likelihood",
    "Weather patterns show persistence"
]

# Generate data for 10 participants
for pid in range(1, 11):
    for domain in sequences:
        for length in [5, 10]:
            example_data['participant_id'].append(pid)
            example_data['domain'].append(domain)
            example_data['sequence_length'].append(length)
            example_data['sequence'].append(sequences[domain][length])
            
            # Add plausible predictions with some noise
            base_pred = {
                'coin': 50,
                'basketball': 70,
                'weather': 65
            }[domain]
            prediction = base_pred + np.random.normal(0, 10)
            example_data['prediction'].append(max(0, min(100, prediction)))
            
            # Add plausible confidence ratings
            base_conf = {
                'coin': 60,
                'basketball': 75,
                'weather': 70
            }[domain]
            confidence = base_conf + np.random.normal(0, 15)
            example_data['confidence'].append(max(0, min(100, confidence)))
            
            # Add reasoning
            reasons = {
                'coin': coin_reasons,
                'basketball': basketball_reasons,
                'weather': weather_reasons
            }[domain]
            example_data['reasoning'].append(np.random.choice(reasons))

# Create DataFrame
human_df = pd.DataFrame(example_data)

# Let's analyze this data and compare it with our model predictions
def analyze_human_model_comparison(human_df, model_results):
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Predictions by domain - comparing human vs model
    plt.subplot(2, 2, 1)
    sns.boxplot(data=human_df, x='domain', y='prediction')
    plt.title('Human Predictions by Domain')
    
    # 2. Confidence by domain and sequence length
    plt.subplot(2, 2, 2)
    sns.boxplot(data=human_df, x='domain', y='confidence', hue='sequence_length')
    plt.title('Confidence by Domain and Sequence Length')
    
    # 3. Sequence length effect on predictions
    plt.subplot(2, 2, 3)
    sns.boxplot(data=human_df, x='domain', y='prediction', hue='sequence_length')
    plt.title('Predictions by Sequence Length')
    
    # 4. Distribution of predictions
    plt.subplot(2, 2, 4)
    for domain in human_df['domain'].unique():
        sns.kdeplot(data=human_df[human_df['domain'] == domain], x='prediction', label=domain)
    plt.title('Distribution of Predictions by Domain')
    
    plt.tight_layout()
    plt.savefig('results/human_analysis.png')
    
    # Print summary statistics
    print("\nHuman Prediction Summary by Domain:")
    print(human_df.groupby('domain')['prediction'].describe())
    
    print("\nConfidence Summary by Domain:")
    print(human_df.groupby('domain')['confidence'].describe())
    
    print("\nSequence Length Effects:")
    print(human_df.groupby(['domain', 'sequence_length'])['prediction'].mean())
    
    # Analyze reasoning patterns
    print("\nCommon Reasoning Patterns:")
    for domain in human_df['domain'].unique():
        print(f"\n{domain.upper()}:")
        reasons = human_df[human_df['domain'] == domain]['reasoning'].value_counts()
        for reason, count in reasons.items():
            print(f"- {reason}: {count} times")

# Run analysis
analyze_human_model_comparison(human_df, None)  # We can add model results comparison later

# Save the simulated data
human_df.to_csv('results/human_experiment_data.csv', index=False)