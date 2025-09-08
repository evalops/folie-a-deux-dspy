#!/usr/bin/env python3
"""
Plot truth-agreement trade-offs from Folie à Deux experiments
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_tradeoff(
    csv_path="results/results_alpha_sweep.csv",
    save_path="results/figures/tradeoff_results.png",
):
    """Plot the truth-agreement Pareto frontier"""
    
    # Load results
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Truth vs Agreement trade-off
    ax1.plot(df['alpha'], df['truth_accuracy'], 'bo-', label='Truth Accuracy', linewidth=2, markersize=6)
    ax1.plot(df['alpha'], df['kappa_agreement'], 'rs-', label="Cohen's κ", linewidth=2, markersize=6)
    ax1.plot(df['alpha'], df['raw_agreement'], 'g^--', label='Raw Agreement', linewidth=1, markersize=5, alpha=0.7)
    
    ax1.set_xlabel('Alpha (Truth Anchoring Weight)', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Truth-Agreement Trade-off', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.3, 1.0)
    
    # Plot 2: Conditional accuracy
    ax2.plot(df['alpha'], df['p_correct_given_agree'], 'mo-', label='P(correct|agree)', linewidth=2, markersize=6)
    ax2.plot(df['alpha'], df['p_correct_given_disagree'], 'co-', label='P(correct|disagree)', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Alpha (Truth Anchoring Weight)', fontsize=12)
    ax2.set_ylabel('Conditional Accuracy', fontsize=12)
    ax2.set_title('Consensus Quality vs. Disagreement Signal', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.6, 1.0)
    
    plt.tight_layout()
    
    # Save figure
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    
    # Show key insights
    print("\nKey Insights:")
    print("=" * 50)
    
    # Find sweet spot where P(correct|agree) is high but kappa is still meaningful  
    sweet_spot_idx = df.loc[df['p_correct_given_agree'] >= 0.9].index[0]
    sweet_spot_alpha = df.loc[sweet_spot_idx, 'alpha']
    
    print(f"Sweet spot at α = {sweet_spot_alpha}:")
    print(f"  Truth Accuracy: {df.loc[sweet_spot_idx, 'truth_accuracy']:.3f}")
    print(f"  Cohen's κ: {df.loc[sweet_spot_idx, 'kappa_agreement']:.3f}")
    print(f"  P(correct|agree): {df.loc[sweet_spot_idx, 'p_correct_given_agree']:.3f}")
    
    # Show the groupthink problem at low alpha
    groupthink_idx = df.loc[df['alpha'] == 0.0].index[0]
    print(f"\nGroupthink at α = 0.0:")
    print(f"  High raw agreement: {df.loc[groupthink_idx, 'raw_agreement']:.3f}")
    print(f"  Low κ (chance-level): {df.loc[groupthink_idx, 'kappa_agreement']:.3f}")
    print(f"  Poor consensus quality: {df.loc[groupthink_idx, 'p_correct_given_agree']:.3f}")

if __name__ == "__main__":
    plot_tradeoff()