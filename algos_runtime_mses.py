import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Directory containing JSON files
directory = './json_files'  # Replace with your directory path

# Initialize dictionaries to store cumulative values and counts
runtime_totals = {}
mse_totals = {}
counts = {}

# Read and process all JSON files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            data = json.load(file)

            for method, runtime in data['runtimes'].items():
                runtime_totals[method] = runtime_totals.get(method, 0) + runtime
                counts[method] = counts.get(method, 0) + 1

            for method, mse in data['mses'].items():
                mse_totals[method] = mse_totals.get(method, 0) + mse

# Compute averages
runtime_averages = {method: runtime_totals[method] / counts[method] for method in runtime_totals}
mse_averages = {method: mse_totals[method] / counts[method] for method in mse_totals}

# Plot runtime averages
methods = list(runtime_averages.keys())
runtime_values = list(runtime_averages.values())
mse_values = [mse_averages[method] for method in methods]

x = np.arange(len(methods))  # x-axis positions

fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle(f"Comparison of Runtimes and Mean Squared Errors (MSEs) of different algorithms", fontsize=16, weight='bold')

# Runtime plot
ax[0].bar(x, runtime_values, color='skyblue')
for i, v in enumerate(runtime_values):
    ax[0].text(i, v + max(runtime_values) * 0.02, f'{v:.2f}', ha='center', fontsize=10)
ax[0].set_ylabel('Average Runtime (seconds)')
ax[0].set_title('Average Runtime')

# MSE plot
ax[1].bar(x, mse_values, color='lightcoral')
for i, v in enumerate(mse_values):
    ax[1].text(i, v + max(mse_values) * 0.02, f'{v:.2f}', ha='center', fontsize=10)
ax[1].set_ylabel('Average MSE')
ax[1].set_title('Average MSE')

# Common x-axis
plt.xticks(x, methods, rotation=45, ha='right', fontsize=10)
plt.tight_layout()

# Show the plot
# plt.show()
plt.savefig("algos_runtime_mses.jpg", dpi=300, bbox_inches='tight')
