import pandas as pd
import numpy as np

# Part 1: Sequential Predictions
def create_sequential_predictions_df():
    sequence1 = ['H', 'T', 'T', 'H', 'T', 'H', 'H', 'H', 'T', 'T']
    sequence2 = ['T', 'H', 'T', 'T', 'H', 'T', 'H', 'H', 'T', 'H']
    
    # Put human data into the `data` dictionary
    # Says each participant's predictions for each domain (coin/basketball) in each sequence above
    data = {
        'sequence1': {
            'coin': {
                'Participant 1': ['T', 'T', 'H', 'H', 'T', 'T', 'H', 'H', 'H'],
                'Participant 2': ['H', 'H', 'H', 'H', 'T', 'H', 'H', 'H', 'H'],
                'Participant 3': ['H', 'H', 'H', 'H', 'H', 'T', 'T', 'H', 'H']
            },
            'basketball': {
                'Participant 1': ['H', 'T', 'T', 'T', 'T', 'H', 'H', 'H', 'T'],
                'Participant 2': ['T', 'T', 'T', 'H', 'T', 'T', 'H', 'H', 'T'],
                'Participant 3': ['T', 'T', 'T', 'T', 'H', 'T', 'H', 'T', 'T']
            }
        },
        'sequence2': {
            'coin': {
                'Participant 4': ['T', 'H', 'T', 'T', 'T', 'T', 'T', 'H', 'T'],
                'Participant 5': ['T', 'T', 'H', 'H', 'T', 'H', 'T', 'T', 'T'],
                'Participant 6': ['T', 'T', 'H', 'H', 'T', 'H', 'T', 'T', 'T']
            },
            'basketball': {
                'Participant 4': ['T', 'H', 'T', 'T', 'T', 'T', 'T', 'T', 'H'],
                'Participant 5': ['T', 'H', 'H', 'T', 'T', 'H', 'H', 'H', 'H'],
                'Participant 6': ['T', 'T', 'H', 'T', 'H', 'H', 'H', 'H', 'H']
            }
        }
    }

    
    # Create DataFrame for sequential predictions
    rows = []
    for seq_num, sequence in enumerate([sequence1, sequence2], 1):
        seq_data = data[f'sequence{seq_num}']
        for domain in ['coin', 'basketball']:
            for participant, predictions in seq_data[domain].items():
                # Start from turn 2 (index 1) since turn 1 was given (they only predict turns 2-10)
                for turn, (true_outcome, prediction) in enumerate(zip(sequence[1:], predictions), 1):
                    rows.append({
                        'sequence_number': seq_num,
                        'participant': participant,
                        'domain': domain,
                        'turn': turn,
                        'true_outcome': true_outcome,
                        'prediction': prediction,
                        'previous_outcome': sequence[turn-1]  # Add previous outcome for context
                    })
    
    df_sequential = pd.DataFrame(rows)

    # Map outcomes to binary values
    df_sequential['true_outcome'] = df_sequential['true_outcome'].map({'H': 1, 'T': 0})
    df_sequential['prediction'] = df_sequential['prediction'].map({'H': 1, 'T': 0})
    df_sequential['previous_outcome'] = df_sequential['previous_outcome'].map({'H': 1, 'T': 0})
    
    return df_sequential

# Part 2: Domain Classification
def create_domain_classification_df():
    alternating_sequences = ['TTFFTFTFFT', 'TFTFTFTFTF']
    streaky_sequences = ['TTTFFTTFFF', 'TTTTTFFFFF']
    
    # Put human data into the `data` dictionary
    # Says each participant's predictions for each domain (coin/basketball) in each sequence above
    data = {
        'alternating': {
            'Participant 1': ['Coin', 'Coin'],
            'Participant 2': ['Coin', 'Coin'],
            'Participant 3': ['Coin', 'Coin']
        },
        'streaky': {
            'Participant 4': ['Coin', 'Basketball'],
            'Participant 5': ['Basketball', 'Basketball'],
            'Participant 6': ['Basketball', 'Basketball']
        }
    }
    
    rows = []
    for sequence_type in ['alternating', 'streaky']:
        sequences = alternating_sequences if sequence_type == 'alternating' else streaky_sequences
        for participant, predictions in data[sequence_type].items():
            for seq_num, (sequence, prediction) in enumerate(zip(sequences, predictions)):
                rows.append({
                    'sequence_type': sequence_type,
                    'sequence': sequence,
                    'participant': participant,
                    'prediction': prediction
                })
    
    df_classification = pd.DataFrame(rows)
    return df_classification

# Part 3: Luck vs Skill Ratings (not used in final report)
# The original experiment was asking participants to rate the luck vs skill of the sequences,
# as in a 0 means they think the source was a game that's all luck, and 100 means they think
# the source was a game that's all skill.
def create_skill_ratings_df():
    alternating_sequences = ['TTFFTFTFFT', 'TFTFTFTFTF']
    streaky_sequences = ['TTTFFTTFFF', 'TTTTTFFFFF']
    
    data = {
        'alternating': {
            'Participant 1': [15, 10],
            'Participant 2': [25, 20],
            'Participant 3': [35, 20]
        },
        'streaky': {
            'Participant 4': [30, 70],
            'Participant 5': [40, 75],
            'Participant 6': [45, 85]
        }
    }
    
    rows = []
    for sequence_type in ['alternating', 'streaky']:
        sequences = alternating_sequences if sequence_type == 'alternating' else streaky_sequences
        for participant, ratings in data[sequence_type].items():
            for seq_num, (sequence, rating) in enumerate(zip(sequences, ratings)):
                rows.append({
                    'sequence_type': sequence_type,
                    'sequence': sequence,
                    'participant': participant,
                    'skill_rating': rating
                })
    
    df_ratings = pd.DataFrame(rows)
    return df_ratings

# Create and save all DataFrames
df_sequential = create_sequential_predictions_df()
df_classification = create_domain_classification_df()
df_ratings = create_skill_ratings_df()

# Save to CSV files
df_sequential.to_csv('results/sequential_predictions.csv', index=False)
df_classification.to_csv('results/domain_classification.csv', index=False)
df_ratings.to_csv('results/skill_ratings.csv', index=False)

# Print preview of the data
print("\nSequential Predictions Preview:")
print(df_sequential.head())
print("\nDomain Classification Preview:")
print(df_classification.head())
print("\nSkill Ratings Preview:")
print(df_ratings.head())