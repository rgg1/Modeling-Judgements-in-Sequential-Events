import pandas as pd

# Create a template DataFrame to record participant responses
def create_data_template():
    # Create lists of all conditions
    domains = ['coin', 'basketball', 'weather']
    lengths = [5, 10]
    
    # Create empty DataFrame with appropriate columns
    columns = [
        'participant_id',          # Unique ID for each participant
        'domain',                  # Which domain (coin/basketball/weather)
        'sequence_length',         # Length of sequence (5 or 10)
        'sequence',                # The actual sequence shown
        'prediction',              # Their probability prediction (0-100)
        'confidence',              # Their confidence rating (0-100)
        'reasoning',               # Their written explanation
        'time_taken'              # Optional: how long they took
    ]
    
    # Create empty DataFrame
    df = pd.DataFrame(columns=columns)
    
    # Save template
    df.to_csv('data_collection_template.csv', index=False)
    
    return df

# Create template
template = create_data_template()

# Show example of how data would be entered
example_data = {
    'participant_id': [1, 1, 1],
    'domain': ['coin', 'coin', 'basketball'],
    'sequence_length': [5, 10, 5],
    'sequence': [
        '[True, True, False, False, True]',
        '[True, False, True, False, False, False, True, True, True, False]',
        '[True, False, True, True, True]'
    ],
    'prediction': [65, 48, 75],
    'confidence': [80, 60, 85],
    'reasoning': [
        'After two tails, heads seems more likely',
        'Seems random, slightly less than 50-50',
        'They seem to be on a hot streak'
    ],
    'time_taken': [45, 60, 50]
}

# Show example
example_df = pd.DataFrame(example_data)
print("\nExample of how data would be entered:")
print(example_df)