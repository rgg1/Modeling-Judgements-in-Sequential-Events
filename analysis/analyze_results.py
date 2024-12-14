import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from scipy import stats

def analyze_streak_effects(df):
    def get_streak_length(sequence):
        count = 1
        for i in range(len(sequence)-2, -1, -1):
            if sequence[i] == sequence[-1]:
                count += 1
            else:
                break
        return count
    
    df['final_streak'] = df['sequence'].apply(get_streak_length)
    df['streak_type'] = df['sequence'].apply(lambda x: 'success' if x[-1] else 'failure')
    
    return df

def analyze_results(json_file_path):
    # Previous code remains the same until visualizations
    with open(json_file_path, 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    df['sequence_str'] = df['sequence'].apply(lambda x: ','.join(str(i) for i in x))
    df['success_rate'] = df['sequence'].apply(lambda x: sum(x)/len(x))
    df['final_outcome'] = df['sequence'].apply(lambda x: x[-1])
    
    # Add streak analysis
    df = analyze_streak_effects(df)
    
    # Create enhanced visualizations
    plt.figure(figsize=(15, 15))
    
    # 1. Box plot of predictions by domain
    plt.subplot(3, 2, 1)
    sns.boxplot(data=df, x='domain', y='prediction')
    plt.title('Predictions by Domain')
    
    # 2. Scatter plot of confidence vs prediction
    plt.subplot(3, 2, 2)
    sns.scatterplot(data=df, x='prediction', y='confidence', hue='domain')
    plt.title('Confidence vs Prediction')
    
    # 3. Predictions by streak length
    plt.subplot(3, 2, 3)
    sns.scatterplot(data=df, x='final_streak', y='prediction', hue='domain', style='streak_type')
    plt.title('Predictions by Streak Length')
    
    # 4. Success rate vs prediction
    plt.subplot(3, 2, 4)
    sns.scatterplot(data=df, x='success_rate', y='prediction', hue='domain')
    plt.title('Success Rate vs Prediction')
    
    # 5. Confidence by sequence length
    plt.subplot(3, 2, 5)
    sns.boxplot(data=df, x='domain', y='confidence', hue='length')
    plt.title('Confidence by Domain and Length')
    
    # 6. Prediction by streak type
    plt.subplot(3, 2, 6)
    sns.violinplot(data=df, x='domain', y='prediction', hue='streak_type')
    plt.title('Predictions by Domain and Streak Type')
    
    plt.tight_layout()
    plt.savefig('results/analysis_plots.png')
    
    # Enhanced statistical analysis
    print("\nSummary Statistics by Domain:")
    print(df.groupby('domain')[['prediction', 'confidence']].describe())
    
    print("\nCorrelations between success rate and predictions:")
    for domain in df['domain'].unique():
        domain_data = df[df['domain'] == domain]
        corr = domain_data['success_rate'].corr(domain_data['prediction'])
        print(f"{domain}: {corr:.3f}")
    
    print("\nStreak Analysis:")
    print(df.groupby(['domain', 'streak_type'])['prediction'].mean())
    
    print("\nConfidence by Sequence Length:")
    print(df.groupby(['domain', 'length'])['confidence'].mean())
    
    return df

# Run analysis
df = analyze_results('results/experiment_results.json')

def compare_model_human():
    # Load data
    human_df = pd.read_csv('results/human_experiment_data.csv')
    
    # Load JSON model results
    with open('results/experiment_results.json', 'r') as f:
        model_data = json.load(f)
    model_df = pd.DataFrame(model_data)
    
    # Convert predictions to same scale (assuming model predictions are 0-1 and human are 0-100)
    model_df['prediction'] = model_df['prediction'] * 100
    
    # Create comparison visualizations
    plt.figure(figsize=(15, 12))
    
    # 1. Model vs Human Predictions by Domain
    plt.subplot(2, 2, 1)
    
    # Calculate means for both human and model data
    human_means = human_df.groupby('domain')['prediction'].mean()
    model_means = model_df.groupby('domain')['prediction'].mean()
    
    # Create comparison data
    comparison_data = pd.DataFrame({
        'Domain': human_means.index.tolist() * 2,
        'Source': ['Human']*3 + ['Model']*3,
        'Prediction': list(human_means.values) + list(model_means.values)
    })
    
    sns.barplot(data=comparison_data, x='Domain', y='Prediction', hue='Source')
    plt.title('Model vs Human Predictions by Domain')
    
    # 2. Sequence Length Effects
    plt.subplot(2, 2, 2)
    human_length_effect = human_df.groupby(['domain', 'sequence_length'])['prediction'].mean().reset_index()
    model_length_effect = model_df.groupby(['domain', 'length'])['prediction'].mean().reset_index()
    
    for domain in human_df['domain'].unique():
        domain_human = human_length_effect[human_length_effect['domain'] == domain]
        domain_model = model_length_effect[model_length_effect['domain'] == domain]
        
        plt.plot(domain_human['sequence_length'], domain_human['prediction'], 
                'o-', label=f'Human-{domain}')
        plt.plot(domain_model['length'], domain_model['prediction'], 
                's--', label=f'Model-{domain}')
    
    plt.title('Sequence Length Effects')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Sequence Length')
    plt.ylabel('Prediction')
    
    # 3. Confidence Comparison
    plt.subplot(2, 2, 3)
    human_conf = human_df.groupby('domain')['confidence'].mean()
    model_conf = model_df.groupby('domain')['confidence'].mean()
    
    conf_comparison = pd.DataFrame({
        'Domain': human_conf.index.tolist() * 2,
        'Source': ['Human']*3 + ['Model']*3,
        'Confidence': list(human_conf.values) + list(model_conf.values * 100)  # Scale model confidence
    })
    
    sns.barplot(data=conf_comparison, x='Domain', y='Confidence', hue='Source')
    plt.title('Confidence Comparison')
    
    # 4. Prediction Distributions
    plt.subplot(2, 2, 4)
    for domain in ['coin', 'basketball', 'weather']:
        sns.kdeplot(data=human_df[human_df['domain'] == domain], 
                   x='prediction', label=f'Human-{domain}', linestyle='-')
        sns.kdeplot(data=model_df[model_df['domain'] == domain], 
                   x='prediction', label=f'Model-{domain}', linestyle='--')
    plt.title('Prediction Distributions')
    
    plt.tight_layout()
    plt.savefig('results/model_human_comparison.png')
    
    # Statistical Analysis
    print("\nStatistical Comparison of Model and Human Predictions:")
    for domain in ['coin', 'basketball', 'weather']:
        human_preds = human_df[human_df['domain'] == domain]['prediction']
        model_preds = model_df[model_df['domain'] == domain]['prediction']
        
        t_stat, p_val = stats.ttest_ind(human_preds, model_preds)
        print(f"\n{domain.capitalize()}:")
        print(f"t-statistic: {t_stat:.3f}")
        print(f"p-value: {p_val:.3f}")
        print(f"Mean difference: {human_preds.mean() - model_preds.mean():.3f}")
        print(f"Human mean: {human_preds.mean():.3f}")
        print(f"Model mean: {model_preds.mean():.3f}")
    
    return comparison_data

# Run comparison
comparison_data = compare_model_human()