# Create both protocol and sheet

test_sequences = {
    'coin': [
        [True, True, False, False, True],  # 5-length
        [True, False, True, False, False, False, True, True, True, False]  # 10-length
    ],
    'basketball': [
        [True, False, True, True, True],  # 5-length
        [True, False, True, True, True, True, True, True, False, False]  # 10-length
    ],
    'weather': [
        [True, True, True, True, False],  # 5-length 
        [True, True, False, True, True, False, False, False, True, True]  # 10-length
    ]
}

def create_experiment_protocol():
    protocol = """
EXPERIMENT PROTOCOL - FOR EXPERIMENTERS

Setup:
1. Print one experiment sheet per participant
2. Have a timer ready (phone is fine)
3. Have a pen ready for participant

Instructions to give participants:
1. "You'll see sequences from three domains: coin flips, basketball shots, and weather"
2. "For each sequence, predict the probability of the next outcome being True"
3. "Also rate your confidence and briefly explain your reasoning"
4. "Take your time, but try not to spend more than 1-2 minutes per sequence"

Procedure:
1. Record participant's start time
2. Let them complete each sequence in order
3. Make sure they fill in all three parts (prediction, confidence, reasoning)
4. If they ask questions:
   - About probability: "Use 0-100% scale"
   - About confidence: "How sure are you about your prediction?"
   - About sequence: "True means heads/made shot/sunny, False means tails/missed/rainy"
   
Data Collection:
1. Each participant should complete all sequences
2. Ensure all fields are filled
3. Record total time taken
4. Thank participant for their time

Important Notes:
- Don't tell participants about model predictions
- Don't share other participants' responses
- If they ask about "correct" answers, say we're interested in their intuitions
"""
    return protocol

def create_experiment_sheet():
    sheet = """
SEQUENTIAL PREDICTION STUDY
Participant ID: _____

Instructions:
For each sequence shown below:
1. Predict probability (0-100%) that next outcome will be True
2. Rate your confidence (0-100%) in your prediction
3. Briefly explain your reasoning

True = Heads (coin) / Made shot (basketball) / Sunny (weather)
False = Tails (coin) / Missed shot (basketball) / Rainy (weather)

"""
    # Add sequences and response spaces
    for domain in test_sequences:
        sheet += f"\n{domain.upper()} SEQUENCES\n"
        for seq in test_sequences[domain]:
            sheet += f"\nSequence: {seq}\n"
            sheet += "Your prediction (0-100%): _____\n"
            sheet += "Your confidence (0-100%): _____\n"
            sheet += "Your reasoning: ______________________________\n"
    
    return sheet

# Save both files
with open('experiment_protocol.txt', 'w') as f:
    f.write(create_experiment_protocol())

with open('experiment_sheet.txt', 'w') as f:
    f.write(create_experiment_sheet())