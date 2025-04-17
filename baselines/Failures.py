import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

# List of planners
# planners = ["PID", "EH", "MPPI", "WMVCT", "MPPI-6", "TAL", "TNT"]
planners = ["RL", "MCL", "ACL"]
base_dir = "./res/data_all/"

# Constants
MAX_STEPS = 12000  # 60 seconds / 5e-3 step size = 12000 steps
ROLLOVER_THRESHOLD = 30  # degrees

# Dictionary to store failure statistics
failure_stats = {}

# Process each planner
for planner in planners:
    planner_dir = os.path.join(base_dir, planner)
    
    if not os.path.exists(planner_dir):
        print(f"Warning: Directory not found for planner {planner}. Skipping.")
        continue
    
    # Initialize counters
    total_runs = 0
    success_count = 0
    timeout_count = 0
    rollover_count = 0
    stuck_count = 0
    
    # Process each world
    for world_id in range(1, 101):
        world_file = os.path.join(planner_dir, f"world{world_id}_data.json")
        
        if not os.path.exists(world_file):
            print(f"Warning: Data file not found for {planner}, world {world_id}. Skipping.")
            continue
        
        try:
            with open(world_file, 'r') as f:
                world_data = json.load(f)
            
            # Process each position in this world
            for pos_data in world_data:
                total_runs += 1
                
                if pos_data['success']:
                    success_count += 1
                else:
                    # Check failure type
                    roll_data = pos_data.get('roll_data', [])
                    pitch_data = pos_data.get('pitch_data', [])
                    roll_data_abs = [abs(val) if isinstance(val, (int, float)) else 0 for val in roll_data]
                    pitch_data_abs = [abs(val) if isinstance(val, (int, float)) else 0 for val in pitch_data]
                    
                    if len(roll_data) == MAX_STEPS or len(pitch_data) == MAX_STEPS:
                        timeout_count += 1
                    elif (roll_data_abs and max(roll_data_abs) > ROLLOVER_THRESHOLD) or \
                         (pitch_data_abs and max(pitch_data_abs) > ROLLOVER_THRESHOLD):
                        rollover_count += 1
                    else:
                        stuck_count += 1
                        
        except Exception as e:
            print(f"Error processing {world_file}: {e}")
    
    # Calculate percentages
    failure_count = rollover_count + stuck_count + timeout_count
    
    if total_runs == 0:
        print(f"Warning: No valid runs found for planner {planner}")
        continue
        
    success_percentage = (success_count / total_runs) * 100
    
    # Calculate percentages of each failure type relative to total runs
    rollover_percentage = (rollover_count / total_runs) * 100
    stuck_percentage = (stuck_count / total_runs) * 100
    timeout_percentage = (timeout_count / total_runs) * 100
    
    # Calculate percentages of each failure type relative to failure cases
    if failure_count > 0:
        rollover_percentage_of_failures = (rollover_count / failure_count) * 100
        stuck_percentage_of_failures = (stuck_count / failure_count) * 100
        timeout_percentage_of_failures = (timeout_count / failure_count) * 100
    else:
        rollover_percentage_of_failures = 0
        stuck_percentage_of_failures = 0
        timeout_percentage_of_failures = 0
    
    # Store statistics
    failure_stats[planner] = {
        "total_runs": total_runs,
        "success_count": success_count,
        "success_percentage": success_percentage,
        "failure_count": failure_count,
        "rollover_count": rollover_count,
        "stuck_count": stuck_count,
        "timeout_count": timeout_count,
        "rollover_percentage": rollover_percentage,
        "stuck_percentage": stuck_percentage,
        "timeout_percentage": timeout_percentage,
        "rollover_percentage_of_failures": rollover_percentage_of_failures,
        "stuck_percentage_of_failures": stuck_percentage_of_failures,
        "timeout_percentage_of_failures": timeout_percentage_of_failures
    }

# Create a DataFrame for better visualization
results_df = pd.DataFrame({
    "Planner": [],
    "Success Rate (%)": [],
    "Rollover (%)": [],
    "Stuck (%)": [],
    "Timeout (%)": [],
    "Rollover (% of failures)": [],
    "Stuck (% of failures)": [],
    "Timeout (% of failures)": []
})

for planner, stats in failure_stats.items():
    results_df = pd.concat([results_df, pd.DataFrame({
        "Planner": [planner],
        "Success Rate (%)": [stats["success_percentage"]],
        "Rollover (%)": [stats["rollover_percentage"]],
        "Stuck (%)": [stats["stuck_percentage"]],
        "Timeout (%)": [stats["timeout_percentage"]],
        "Rollover (% of failures)": [stats["rollover_percentage_of_failures"]],
        "Stuck (% of failures)": [stats["stuck_percentage_of_failures"]],
        "Timeout (% of failures)": [stats["timeout_percentage_of_failures"]]
    })], ignore_index=True)

# Sort by success rate
results_df = results_df.sort_values(by="Success Rate (%)", ascending=False)

# Display results
print("\nPlanner Failure Analysis:")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Save results to CSV
results_df.to_csv(f"./res/data_all/" + "planner_failure_analysis_learning.csv", index=False)
print("\nResults saved to planner_failure_analysis_learning.csv")

# Create visualizations

# Set up the figure and axes
plt.figure(figsize=(14, 10))

# 1. Success Rate bar chart
plt.subplot(2, 2, 1)
bars = plt.bar(results_df["Planner"], results_df["Success Rate (%)"], color='green')
plt.title('Success Rate by Planner')
plt.xlabel('Planner')
plt.ylabel('Success Rate (%)')
plt.xticks(rotation=45)
plt.ylim(0, 100)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom')

# 2. Failure Distribution (% of total runs)
plt.subplot(2, 2, 2)
bottom = np.zeros(len(results_df))

for failure_type, color in [("Rollover (%)", 'red'), ("Stuck (%)", 'orange'), ("Timeout (%)", 'blue')]:
    plt.bar(results_df["Planner"], results_df[failure_type], bottom=bottom, label=failure_type, color=color)
    bottom += results_df[failure_type]

plt.title('Failure Distribution (% of Total Runs)')
plt.xlabel('Planner')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend()
plt.ylim(0, 100)

# 3. Failure Distribution (% of failures)
plt.subplot(2, 2, 3)
bottom = np.zeros(len(results_df))

for failure_type, color in [("Rollover (% of failures)", 'red'), 
                           ("Stuck (% of failures)", 'orange'), 
                           ("Timeout (% of failures)", 'blue')]:
    plt.bar(results_df["Planner"], results_df[failure_type], bottom=bottom, label=failure_type.replace(" (% of failures)", ""), color=color)
    bottom += results_df[failure_type]

plt.title('Failure Distribution (% of Failures)')
plt.xlabel('Planner')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend()
plt.ylim(0, 100)

# 4. Failure counts stacked bar chart
plt.subplot(2, 2, 4)

# Create a dataframe with raw counts for plotting
counts_df = pd.DataFrame({
    "Planner": [],
    "Success": [],
    "Rollover": [],
    "Stuck": [],
    "Timeout": []
})

for planner, stats in failure_stats.items():
    counts_df = pd.concat([counts_df, pd.DataFrame({
        "Planner": [planner],
        "Success": [stats["success_count"]],
        "Rollover": [stats["rollover_count"]],
        "Stuck": [stats["stuck_count"]],
        "Timeout": [stats["timeout_count"]]
    })], ignore_index=True)

# Sort by the same order as the main results dataframe
counts_df = counts_df.set_index("Planner").loc[results_df["Planner"]].reset_index()

# Additionally, create a detailed table with raw counts
detailed_df = pd.DataFrame({
    "Planner": [],
    "Total Runs": [],
    "Success Count": [],
    "Success Rate (%)": [],
    "Rollover Count": [],
    "Stuck Count": [],
    "Timeout Count": []
})

for planner, stats in failure_stats.items():
    detailed_df = pd.concat([detailed_df, pd.DataFrame({
        "Planner": [planner],
        "Total Runs": [stats["total_runs"]],
        "Success Count": [stats["success_count"]],
        "Success Rate (%)": [stats["success_percentage"]],
        "Rollover Count": [stats["rollover_count"]],
        "Stuck Count": [stats["stuck_count"]],
        "Timeout Count": [stats["timeout_count"]]
    })], ignore_index=True)

# Sort by success rate
detailed_df = detailed_df.sort_values(by="Success Rate (%)", ascending=False)

# Save to CSV
detailed_df.to_csv(f"./res/data_all/" + "planner_detailed_stats_learning.csv", index=False)
print("Detailed statistics saved to planner_detailed_stats_learning.csv")