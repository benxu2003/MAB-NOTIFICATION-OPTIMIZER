import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.environment import NotificationEnv, RealDataStreamer
from src.bandit import ThompsonSamplingBandit

def run_synthetic_simulation(steps=10000):
    print("🚀 Starting Synthetic Notification Optimization Simulation...")
    notification_configs = {
        "V1_Generic_Promo": 0.05,
        "V2_Urgency_FOMO": 0.08,
        "V3_Personalized_Offer": 0.15, 
        "V4_Emoji_Heavy": 0.04
    }
    
    env = NotificationEnv(notification_configs)
    bandit = ThompsonSamplingBandit(list(notification_configs.keys()))
    
    rewards_log = []
    cumulative_rewards = 0
    selection_counts = {v: 0 for v in notification_configs.keys()}
    ctr_log = []
    
    for step in range(1, steps + 1):
        chosen_variant = bandit.select_variant()
        selection_counts[chosen_variant] += 1
        reward = env.push_notification(chosen_variant)
        bandit.update(chosen_variant, reward)
        
        cumulative_rewards += reward
        rewards_log.append(cumulative_rewards)
        ctr_log.append(cumulative_rewards / step)
        
        if step % 2000 == 0:
            print(f"Step {step}/{steps} | Cumulative CTR: {cumulative_rewards/step:.4f} | Leader: {chosen_variant}")

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Cumulative rewards
    axes[0, 0].plot(rewards_log, label="Thompson Sampling", color="teal", linewidth=2)
    axes[0, 0].plot([i * 0.15 for i in range(steps)], label="Theoretical Optimal", linestyle="--", color="orange", alpha=0.7)
    axes[0, 0].set_xlabel("Notifications Pushed")
    axes[0, 0].set_ylabel("Cumulative Clicks")
    axes[0, 0].set_title("Convergence Process - Cumulative Rewards")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Running CTR
    axes[0, 1].plot(ctr_log, label="Thompson Sampling CTR", color="teal", linewidth=2)
    axes[0, 1].axhline(y=0.15, label="Optimal CTR (V3)", linestyle="--", color="orange", alpha=0.7)
    axes[0, 1].set_xlabel("Notifications Pushed")
    axes[0, 1].set_ylabel("Click-Through Rate (CTR)")
    axes[0, 1].set_title("Running CTR Over Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Final traffic allocation
    axes[1, 0].bar(selection_counts.keys(), selection_counts.values(), color="teal", alpha=0.7)
    axes[1, 0].set_xticklabels(selection_counts.keys(), rotation=15)
    axes[1, 0].set_ylabel("Impressions Allocated")
    axes[1, 0].set_title("Final Traffic Allocation by Variant")
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: True vs Learned CTRs
    true_ctrs = list(notification_configs.values())
    learned_ctrs = [bandit.alphas[k] / (bandit.alphas[k] + bandit.betas[k]) for k in notification_configs.keys()]
    x = np.arange(len(notification_configs))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, true_ctrs, width, label='True CTR', alpha=0.7, color='lightcoral')
    axes[1, 1].bar(x + width/2, learned_ctrs, width, label='Learned CTR', alpha=0.7, color='skyblue')
    axes[1, 1].set_xlabel('Notification Variants')
    axes[1, 1].set_ylabel('Conversion Rate')
    axes[1, 1].set_title('True vs Learned Conversion Rates')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(notification_configs.keys(), rotation=15)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"results/synthetic_simulation_{timestamp}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved as {plot_filename}")
    
    # Print final statistics
    print("\n📈 Final Statistics:")
    print(f"Total clicks achieved: {cumulative_rewards}")
    print(f"Final CTR: {cumulative_rewards/steps:.4f}")
    print(f"Regret compared to optimal: {(steps * 0.15 - cumulative_rewards)/steps:.4f}")
    print("Traffic allocation:")
    for variant, count in selection_counts.items():
        print(f"  {variant}: {count} ({count/steps*100:.2f}%)")
    
    return cumulative_rewards, selection_counts, bandit

def run_real_data_evaluation(data_file, variant_col='test_group', converted_col='converted'):
    print("🔄 Starting Real Data Evaluation...")
    
    try:
        streamer = RealDataStreamer(data_file, variant_col, converted_col)
        variants = streamer.variants
        print(f"Found variants: {variants}")
        
        # Initialize bandit with real data variants
        bandit = ThompsonSamplingBandit(variants)
        
        rewards_log = []
        cumulative_rewards = 0
        selection_counts = {v: 0 for v in variants}
        policy_selections = {v: 0 for v in variants}  # Track how many times each variant was selected by the policy
        
        step = 0
        while streamer.has_next():
            assigned_variant, actual_conversion = streamer.get_next_interaction()
            step += 1
            
            # Get bandit's recommendation
            recommended_variant = bandit.select_variant()
            policy_selections[recommended_variant] += 1
            
            # Apply off-policy evaluation: if our policy chose the same variant as was actually shown
            if recommended_variant == assigned_variant:
                # Update with actual outcome
                bandit.update(recommended_variant, actual_conversion)
                cumulative_rewards += actual_conversion
            else:
                # We can't update since we didn't choose this variant (off-policy)
                pass
                
            rewards_log.append(cumulative_rewards)
            
            if step % 5000 == 0:
                print(f"Processed {step}/{streamer.total_rows} interactions | Current cumulative reward: {cumulative_rewards}")
                
        print(f"✅ Completed processing {step} interactions from real data")
        
        # Calculate metrics
        if step > 0:
            print(f"Final cumulative reward: {cumulative_rewards}")
            print(f"Observed CTR: {cumulative_rewards/step:.4f}")
        
        return cumulative_rewards, policy_selections, bandit
        
    except FileNotFoundError:
        print(f"❌ Data file {data_file} not found. Please ensure the file exists.")
        return None, None, None

def compare_bandit_algorithms(notification_configs, steps=5000):
    print("⚖️ Comparing different bandit algorithms...")
    
    from src.bandit import ThompsonSamplingBandit, EpsilonGreedyBandit, UCB1Bandit
    
    # Thompson Sampling
    env1 = NotificationEnv(notification_configs)
    thompson_bandit = ThompsonSamplingBandit(list(notification_configs.keys()))
    thompson_rewards = []
    thompson_cumulative = 0
    
    # Epsilon-Greedy
    env2 = NotificationEnv(notification_configs)
    epsilon_bandit = EpsilonGreedyBandit(list(notification_configs.keys()), epsilon=0.1)
    epsilon_rewards = []
    epsilon_cumulative = 0
    
    # UCB1
    env3 = NotificationEnv(notification_configs)
    ucb_bandit = UCB1Bandit(list(notification_configs.keys()))
    ucb_rewards = []
    ucb_cumulative = 0
    
    for step in range(1, steps + 1):
        # Thompson Sampling
        thompson_choice = thompson_bandit.select_variant()
        thompson_reward = env1.push_notification(thompson_choice)
        thompson_bandit.update(thompson_choice, thompson_reward)
        thompson_cumulative += thompson_reward
        thompson_rewards.append(thompson_cumulative)
        
        # Epsilon-Greedy
        epsilon_choice = epsilon_bandit.select_variant()
        epsilon_reward = env2.push_notification(epsilon_choice)
        epsilon_bandit.update(epsilon_choice, epsilon_reward)
        epsilon_cumulative += epsilon_reward
        epsilon_rewards.append(epsilon_cumulative)
        
        # UCB1
        ucb_choice = ucb_bandit.select_variant()
        ucb_reward = env3.push_notification(ucb_choice)
        ucb_bandit.update(ucb_choice, ucb_reward)
        ucb_cumulative += ucb_reward
        ucb_rewards.append(ucb_cumulative)
        
        if step % 1000 == 0:
            print(f"Step {step}/{steps} | Thompson: {thompson_cumulative}, Epsilon-Greedy: {epsilon_cumulative}, UCB1: {ucb_cumulative}")

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(thompson_rewards, label="Thompson Sampling", linewidth=2)
    plt.plot(epsilon_rewards, label="Epsilon-Greedy (ε=0.1)", linewidth=2)
    plt.plot(ucb_rewards, label="UCB1", linewidth=2)
    plt.plot([i * 0.15 for i in range(steps)], label="Theoretical Optimal", linestyle="--", color="black", alpha=0.7)
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward")
    plt.title("Bandit Algorithm Comparison - Mobile Notification Optimization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"results/algorithm_comparison_{timestamp}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Algorithm comparison plot saved as {plot_filename}")
    
    print(f"\n🏆 Final Results (after {steps} steps):\nThompson Sampling: {thompson_cumulative}\nEpsilon-Greedy: {epsilon_cumulative}\nUCB1: {ucb_cumulative}")

def main():
    print("📱 Multi-Armed Bandit for Mobile Notification Optimization") 
    print("========================================================\n")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run synthetic simulation
    print("1️⃣ Running Synthetic Simulation...")
    synthetic_rewards, selection_counts, bandit = run_synthetic_simulation(steps=10000)
    
    # Compare algorithms
    print("\n2️⃣ Comparing Different Bandit Algorithms...")
    notification_configs = {
        "V1_Generic_Promo": 0.05,
        "V2_Urgency_FOMO": 0.08,
        "V3_Personalized_Offer": 0.15, 
        "V4_Emoji_Heavy": 0.04
    }
    compare_bandit_algorithms(notification_configs, steps=5000)
    
    # Try real data evaluation if available
    print("\n3️⃣ Attempting Real Data Evaluation...")
    data_file = "data/marketing_data.csv"
    if os.path.exists(data_file):
        real_rewards, policy_selections, real_bandit = run_real_data_evaluation(data_file)
        if real_rewards is not None:
            print(f"Real data evaluation completed with {real_rewards} cumulative rewards")
    else:
        print(f"⚠️ Real data file '{data_file}' not found. Skipping real data evaluation.")
        print("To run real data evaluation, place your CSV file at 'data/marketing_data.csv'")
        print("with columns 'test_group' and 'converted'.")
    
    print("\n🎉 All simulations completed!")
    print("Check the 'results/' folder for visualizations and reports.")

if __name__ == "__main__":
    main()