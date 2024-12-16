import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

def load_data():
    try:
        sequential_df = pd.read_csv('results/sequential_predictions.csv')
        classification_df = pd.read_csv('results/domain_classification.csv')
        with open('results/model_predictions.json', 'r') as f:
            model_data = json.load(f)
        return sequential_df, classification_df, model_data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e

def analyze_streak_influence(df, model_data):
    """
    Analyze how streak length affects predictions

    Args:
        df: DataFrame with human predictions
        model_data: Dictionary with model predictions

    Returns:
        streak_analysis: Dictionary with mean and SEM (standard error of the mean) for each streak length
        correlations: Dictionary with Pearson correlation and p-value for each domain
    """
    
    def get_streak_length(sequence, position):
        if position == 0:
            return 0
        current_val = sequence[position-1]
        streak = 1
        for i in range(position-2, -1, -1):
            if sequence[i] == current_val:
                streak += 1
            else:
                break
        return streak
    
    # Calculate streak lengths for human data
    df['streak_length'] = df.apply(lambda row: 
        get_streak_length(df[df['sequence_number'] == row['sequence_number']]['true_outcome'].tolist(), 
                         row['turn']), axis=1)
    
    # Process model predictions
    model_rows = []
    for seq_data in model_data['sequentialPredictions']:
        sequence = seq_data['sequence']
        for pred in seq_data['predictions']:
            if pred is not None:
                streak_length = get_streak_length(pred['prevSequence'], pred['position'])
                model_rows.append({
                    'position': pred['position'],
                    'streak_length': streak_length,
                    'coin_prediction': pred['coinPrediction'],
                    'basketball_prediction': pred['basketballPrediction']
                })
    model_df = pd.DataFrame(model_rows)
    
    # Calculate mean and standard error for each streak length
    streak_analysis = {
        'Human': {
            'Coin': df[df['domain'] == 'coin'].groupby('streak_length')['prediction'].agg(['mean', 'sem']),
            'Basketball': df[df['domain'] == 'basketball'].groupby('streak_length')['prediction'].agg(['mean', 'sem'])
        },
        'Model': {
            'Coin': model_df.groupby('streak_length')['coin_prediction'].agg(['mean', 'sem']),
            'Basketball': model_df.groupby('streak_length')['basketball_prediction'].agg(['mean', 'sem'])
        }
    }
    
    # Calculate correlations
    correlations = {
        'Human': {
            'Coin': stats.pearsonr(df[df['domain'] == 'coin']['streak_length'], 
                                 df[df['domain'] == 'coin']['prediction']),
            'Basketball': stats.pearsonr(df[df['domain'] == 'basketball']['streak_length'], 
                                       df[df['domain'] == 'basketball']['prediction'])
        },
        'Model': {
            'Coin': stats.pearsonr(model_df['streak_length'], 
                                 model_df['coin_prediction']),
            'Basketball': stats.pearsonr(model_df['streak_length'], 
                                       model_df['basketball_prediction'])
        }
    }
    
    return streak_analysis, correlations

def analyze_sequence_classification(classification_df, model_data):
    """
    Compare human and model sequence classification

    Args:
        classification_df: DataFrame with human classifications
        model_data: Dictionary with model predictions

    Returns:
        human_classifications: Proportion of human classifications as 'Basketball' for each sequence type
        model_classifications: Mean probability of basketball classification for each sequence type in the model
    """
    
    # Process human classifications
    human_classifications = classification_df.groupby(['sequence_type'])['prediction'].apply(
        lambda x: (x == 'Basketball').mean()
    )
    
    # Process model classifications
    model_classifications = {
        'alternating': np.mean([pred['basketballProb'] for pred in model_data['classificationPredictions']['alternating']]),
        'streaky': np.mean([pred['basketballProb'] for pred in model_data['classificationPredictions']['streaky']])
    }
    
    return human_classifications, model_classifications

def plot_streak_analysis(streak_analysis):
    plt.figure(figsize=(15, 8))
    
    # Create subplots for each domain
    domains = ['Coin', 'Basketball']
    for idx, domain in enumerate(domains, 1):
        plt.subplot(1, 2, idx)
        
        # Plot human data with confidence intervals
        human_data = streak_analysis['Human'][domain]
        plt.errorbar(human_data.index, human_data['mean'], 
                    yerr=human_data['sem'] * 1.96,  # 95% CI
                    fmt='o-', label='Human', capsize=5)
        
        # Plot model data with confidence intervals
        model_data = streak_analysis['Model'][domain]
        plt.errorbar(model_data.index, model_data['mean'],
                    yerr=model_data['sem'] * 1.96,  # 95% CI
                    fmt='s--', label='Model', capsize=5)
        
        plt.xlabel('Streak Length')
        plt.ylabel('Prediction Probability')
        plt.title(f'Effect of Streak Length on {domain} Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set y-axis limits
        plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('results/streak_analysis.png')

def analyze_transitions(df, model_data):
    """
    Analyze how predictions change based on previous outcomes

    Args:
        df: DataFrame with human predictions
        model_data: Dictionary with model predictions

    Returns:
        human_transitions: Mean and SEM for human transitions
        model_transitions: Mean and SEM for model transitions
    """
    # Human transitions
    human_transitions = {}
    for domain in ['coin', 'basketball']:
        domain_data = df[df['domain'] == domain]
        transitions = domain_data.groupby('previous_outcome')['prediction'].agg(['mean', 'sem'])
        human_transitions[domain] = transitions
    
    # Model transitions
    model_transitions = {
        'coin': {'0': [], '1': []},
        'basketball': {'0': [], '1': []}
    }
    
    for seq_data in model_data['sequentialPredictions']:
        for pred in seq_data['predictions']:
            if pred is not None and len(pred['prevSequence']) > 0:
                last_outcome = pred['prevSequence'][-1]
                key = '1' if last_outcome else '0'
                model_transitions['coin'][key].append(pred['coinPrediction'])
                model_transitions['basketball'][key].append(pred['basketballPrediction'])
    
    # Calculate means and SEMs for model transitions
    model_transitions_stats = {}
    for domain in ['coin', 'basketball']:
        stats_dict = {}
        for outcome in ['0', '1']:
            data = np.array(model_transitions[domain][outcome])
            stats_dict[outcome] = {
                'mean': np.mean(data),
                'sem': np.std(data, ddof=1) / np.sqrt(len(data)) if len(data) > 1 else np.nan
            }
        model_transitions_stats[domain] = stats_dict
    
    return human_transitions, model_transitions_stats

def print_transition_analysis(human_transitions, model_transitions):
    print("\nTransition Analysis:")
    
    # Print human transitions
    for domain in ['coin', 'basketball']:
        print(f"\nHuman {domain.capitalize()} transitions:")
        for prev_outcome in [0, 1]:
            mean = human_transitions[domain].loc[prev_outcome, 'mean']
            sem = human_transitions[domain].loc[prev_outcome, 'sem']
            outcome_type = "Make/Heads" if prev_outcome == 1 else "Miss/Tails"
            print(f"After {prev_outcome} ({outcome_type}): "
                  f"Predict 1 with probability {mean:.3f} (±{sem:.3f})")
    
    # Print model transitions
    for domain in ['coin', 'basketball']:
        print(f"\nModel {domain.capitalize()} transitions:")
        for prev_outcome in ['0', '1']:
            stats = model_transitions[domain][prev_outcome]
            outcome_type = "Make/Heads" if prev_outcome == '1' else "Miss/Tails"
            print(f"After {prev_outcome} ({outcome_type}): "
                  f"Predict 1 with probability {stats['mean']:.3f} (±{stats['sem']:.3f})")
            
def plot_transition_analysis(human_transitions, model_transitions):
    plt.figure(figsize=(12, 6))
    
    # Set up the bar positions
    bar_width = 0.15
    r1 = np.arange(2)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    
    # Create bars
    domains = ['coin', 'basketball']
    colors = {'human_coin': '#1f77b4', 'human_basketball': '#2ca02c',
              'model_coin': '#9467bd', 'model_basketball': '#d62728'}
    
    # Plot human data
    for i, domain in enumerate(domains):
        means = [human_transitions[domain].loc[0, 'mean'], 
                human_transitions[domain].loc[1, 'mean']]
        sems = [human_transitions[domain].loc[0, 'sem'], 
                human_transitions[domain].loc[1, 'sem']]
        pos = r1 if i == 0 else r2
        plt.bar(pos, means, bar_width, yerr=sems, capsize=5,
                label=f'Human {domain.capitalize()}',
                color=colors[f'human_{domain}'])
    
    # Plot model data
    for i, domain in enumerate(domains):
        means = [model_transitions[domain]['0']['mean'], 
                model_transitions[domain]['1']['mean']]
        sems = [model_transitions[domain]['0']['sem'], 
                model_transitions[domain]['1']['sem']]
        pos = r3 if i == 0 else r4
        plt.bar(pos, means, bar_width, yerr=sems, capsize=5,
                label=f'Model {domain.capitalize()}',
                color=colors[f'model_{domain}'])
    
    plt.ylabel('Probability of Predicting 1 (Make/Heads)')
    plt.xlabel('Previous Outcome')
    plt.title('Transition Probabilities by Domain and Source')
    plt.xticks([r + bar_width*1.5 for r in r1], ['After 0 (Miss/Tails)', 'After 1 (Make/Heads)'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/transition_analysis.png', bbox_inches='tight')
    plt.close()

def plot_sequence_classification(human_class, model_class):
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    categories = ['Alternating', 'Streaky']
    human_probs = [human_class['alternating'], human_class['streaky']]
    model_probs = [model_class['alternating'], model_class['streaky']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, human_probs, width, label='Human', color='#1f77b4')
    plt.bar(x + width/2, model_probs, width, label='Model', color='#2ca02c')
    
    plt.ylabel('Probability of Basketball Classification')
    plt.xlabel('Sequence Type')
    plt.title('Sequence Classification: Human vs Model')
    plt.xticks(x, categories)
    plt.legend()
    
    # Add value labels on bars
    for i, v in enumerate(human_probs):
        plt.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center')
    for i, v in enumerate(model_probs):
        plt.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center')
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('results/sequence_classification.png')
    plt.close()

def plot_streak_correlation(df, model_data, correlations):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Prepare data
    human_data = {
        'Coin': df[df['domain'] == 'coin'],
        'Basketball': df[df['domain'] == 'basketball']
    }
    
    model_rows = []
    for seq_data in model_data['sequentialPredictions']:
        for pred in seq_data['predictions']:
            if pred is not None:
                streak_length = len([x for x in pred['prevSequence'] if x == pred['prevSequence'][-1]]) if pred['prevSequence'] else 0
                model_rows.append({
                    'streak_length': streak_length,
                    'coin_prediction': pred['coinPrediction'],
                    'basketball_prediction': pred['basketballPrediction']
                })
    model_df = pd.DataFrame(model_rows)
    
    # Plot settings
    titles = [['Human Coin', 'Human Basketball'],
              ['Model Coin', 'Model Basketball']]
    
    for i, source in enumerate(['Human', 'Model']):
        for j, domain in enumerate(['Coin', 'Basketball']):
            ax = axes[i, j]
            
            if source == 'Human':
                data = human_data[domain]
                x = data['streak_length']
                y = data['prediction']
            else:
                x = model_df['streak_length']
                y = model_df[f'{domain.lower()}_prediction']
            
            x_jitter = x + np.random.normal(0, 0.05, len(x))
            ax.scatter(x_jitter, y, alpha=0.4)
            
            # Add trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(np.unique(x), p(np.unique(x)), "r--", alpha=0.8)
            
            # Add correlation info
            corr = correlations[source][domain]
            ax.text(0.05, 0.95, f'r={corr[0]:.3f}\np={corr[1]:.3f}',
                   transform=ax.transAxes, verticalalignment='top')
            
            ax.set_xlabel('Streak Length')
            ax.set_ylabel('Prediction Probability')
            ax.set_title(titles[i][j])
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/streak_correlation.png')
    plt.close()

def main():
    sequential_df, classification_df, model_data = load_data()
    
    # Analyze streak influence
    streak_analysis, correlations = analyze_streak_influence(sequential_df, model_data)
    
    # Print streak analysis results
    print("\nPrediction Probabilities by Streak Length:")
    print("\nHuman Data:")
    for domain in ['Coin', 'Basketball']:
        print(f"\n{domain}:")
        print(streak_analysis['Human'][domain])
    
    print("\nModel Predictions:")
    for domain in ['Coin', 'Basketball']:
        print(f"\n{domain}:")
        print(streak_analysis['Model'][domain])
    
    print("\nCorrelation between Streak Length and Predictions:")
    for source in ['Human', 'Model']:
        print(f"\n{source}:")
        for domain in ['Coin', 'Basketball']:
            corr = correlations[source][domain]
            print(f"{domain}: r={corr[0]:.3f}, p={corr[1]:.3f}")
    
    print("\nSequence Classification Results:")
    human_class, model_class = analyze_sequence_classification(classification_df, model_data)
    print("\nHuman Classifications (proportion categorized as basketball):")
    print(human_class)
    print("\nModel Classifications (probability of basketball):")
    print(f"Alternating sequences: {model_class['alternating']:.3f}")
    print(f"Streaky sequences: {model_class['streaky']:.3f}")
    
    # Analyze transitions
    human_transitions, model_transitions = analyze_transitions(sequential_df, model_data)
    print_transition_analysis(human_transitions, model_transitions)

    print("Generating visualizations...")
    plot_streak_analysis(streak_analysis)
    plot_transition_analysis(human_transitions, model_transitions)
    plot_sequence_classification(human_class, model_class)
    plot_streak_correlation(sequential_df, model_data, correlations)
    
    print("Visualizations saved in results directory.")

if __name__ == "__main__":
    main()
