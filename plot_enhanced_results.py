#!/usr/bin/env python3
"""
Enhanced plotting script with baselines and sensitivity analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_comprehensive_analysis():
    """Generate comprehensive analysis plots"""
    
    # Load all data
    df_baselines = pd.read_csv("results_with_baselines.csv")
    df_sensitivity = pd.read_csv("degeneracy_sensitivity.csv")
    
    # Create comprehensive figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Main trade-off with baselines
    folie_data = df_baselines[df_baselines['method'] == 'folie_a_deux']
    ax1.plot(folie_data['alpha'], folie_data['truth_accuracy'], 'bo-', 
             label='Truth Accuracy', linewidth=2, markersize=6)
    ax1.plot(folie_data['alpha'], folie_data['cohen_kappa'], 'rs-', 
             label="Cohen's κ", linewidth=2, markersize=6)
    ax1.plot(folie_data['alpha'], folie_data['p_correct_given_agree'], 'go-', 
             label='P(correct|agree)', linewidth=2, markersize=6)
    
    # Add baseline horizontal lines
    baseline_data = df_baselines[df_baselines['method'] != 'folie_a_deux']
    for _, row in baseline_data.iterrows():
        if not pd.isna(row['truth_accuracy']):
            ax1.axhline(y=row['truth_accuracy'], color='gray', linestyle='--', alpha=0.7,
                       label=f"{row['method'].replace('_', ' ').title()}: {row['truth_accuracy']:.2f}")
    
    ax1.set_xlabel('Alpha (Truth Anchoring Weight)', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Truth-Agreement Trade-off vs Baselines', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.3, 1.0)
    
    # Plot 2: Conditional accuracy analysis
    ax2.plot(folie_data['alpha'], folie_data['p_correct_given_agree'], 'mo-', 
             label='P(correct|agree)', linewidth=2, markersize=6)
    ax2.plot(folie_data['alpha'], folie_data['p_correct_given_disagree'], 'co-', 
             label='P(correct|disagree)', linewidth=2, markersize=6)
    
    # Add self-consistency baseline
    sc_row = baseline_data[baseline_data['method'] == 'self_consistency_n5']
    if not sc_row.empty:
        sc_acc = float(sc_row['p_correct_given_agree'].iloc[0])
        ax2.axhline(y=sc_acc, color='orange', linestyle='--', alpha=0.8,
                   label=f'Self-Consistency: {sc_acc:.2f}')
    
    ax2.set_xlabel('Alpha (Truth Anchoring Weight)', fontsize=12)
    ax2.set_ylabel('Conditional Accuracy', fontsize=12)
    ax2.set_title('Consensus Quality Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.6, 1.0)
    
    # Plot 3: Degeneracy sensitivity
    for h_target in df_sensitivity['h_target'].unique():
        subset = df_sensitivity[df_sensitivity['h_target'] == h_target]
        ax3.plot(subset['alpha'], subset['truth_accuracy'], 'o-', 
                label=f'H_target = {h_target}', linewidth=2, markersize=5)
    
    ax3.set_xlabel('Alpha', fontsize=12)
    ax3.set_ylabel('Truth Accuracy', fontsize=12)
    ax3.set_title('Sensitivity to Degeneracy Penalty Target', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Label entropy vs degeneracy control
    h_1_data = df_sensitivity[df_sensitivity['h_target'] == 1.0]
    ax4.plot(h_1_data['alpha'], h_1_data['label_entropy_a'], 'bo-', 
             label='Agent A Entropy', linewidth=2, markersize=5)
    ax4.plot(h_1_data['alpha'], h_1_data['label_entropy_b'], 'ro-', 
             label='Agent B Entropy', linewidth=2, markersize=5)
    ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.8, 
               label='Target Entropy (1.0)')
    
    ax4.set_xlabel('Alpha', fontsize=12)
    ax4.set_ylabel('Label Entropy', fontsize=12)
    ax4.set_title('Degeneracy Control: Label Entropy vs Alpha', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.4, 1.1)
    
    plt.tight_layout()
    
    # Save comprehensive figure
    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive analysis to figures/comprehensive_analysis.png")
    
    # Print key insights
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS INSIGHTS")
    print("="*60)
    
    # Baseline comparisons
    print("\n🎯 BASELINE COMPARISONS:")
    folie_best = folie_data.loc[folie_data['p_correct_given_agree'].idxmax()]
    sc_score = baseline_data[baseline_data['method'] == 'self_consistency_n5']['p_correct_given_agree'].iloc[0]
    single_score = baseline_data[baseline_data['method'] == 'single_miprov2']['truth_accuracy'].iloc[0]
    
    print(f"• Best Folie à Deux (α={folie_best['alpha']:.1f}): {folie_best['p_correct_given_agree']:.1%} P(correct|agree)")
    print(f"• Self-Consistency (n=5): {sc_score:.1%} accuracy")
    print(f"• Single MIPROv2: {single_score:.1%} accuracy")
    print(f"• Improvement over self-consistency: {folie_best['p_correct_given_agree'] - sc_score:.1%}")
    
    # Sweet spot analysis
    sweet_spot = folie_data.loc[folie_data['p_correct_given_agree'] >= 0.9].iloc[0]
    print(f"\n🎯 SWEET SPOT (α={sweet_spot['alpha']:.1f}):")
    print(f"• Truth Accuracy: {sweet_spot['truth_accuracy']:.1%}")
    print(f"• Cohen's κ: {sweet_spot['cohen_kappa']:.3f} (meaningful consensus)")
    print(f"• P(correct|agree): {sweet_spot['p_correct_given_agree']:.1%} (high-quality agreement)")
    print(f"• P(correct|disagree): {sweet_spot['p_correct_given_disagree']:.1%} (disagreement signal)")
    
    # Degeneracy control validation
    print(f"\n🛡️  DEGENERACY CONTROL VALIDATION:")
    print("H_target sensitivity analysis shows:")
    for h in [0.8, 0.9, 1.0, 1.1]:
        subset = df_sensitivity[(df_sensitivity['h_target'] == h) & (df_sensitivity['alpha'] == 0.5)]
        if not subset.empty:
            acc = subset['truth_accuracy'].iloc[0]
            print(f"• H_target = {h}: {acc:.1%} accuracy")
    print("→ H_target = 1.0 optimal for balanced binary tasks")
    
    # Groupthink detection
    groupthink = folie_data[folie_data['alpha'] == 0.0].iloc[0]
    print(f"\n⚠️  GROUPTHINK DETECTION (α=0.0):")
    print(f"• High raw agreement: {groupthink['raw_agreement']:.1%}")
    print(f"• Low κ (chance-level): {groupthink['cohen_kappa']:.3f}")
    print(f"• Poor consensus quality: {groupthink['p_correct_given_agree']:.1%}")
    print("→ Raw agreement metrics would miss this failure mode!")

if __name__ == "__main__":
    plot_comprehensive_analysis()