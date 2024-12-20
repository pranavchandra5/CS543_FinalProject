import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set the directory path where the JSON files are located
directory_path = './output'  # Change this to your target directory

# Initialize a dictionary to hold pixel counts and MSEs for each algorithm
algorithm_data = {}

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.json'):  # Only process JSON files
        file_path = os.path.join(directory_path, filename)
        
        # Read and parse the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract the resolution and MSE data
        resolution = data['resolution']
        mse_values = data['mses']
        
        # Calculate the pixel count (width * height)
        pixel_count = resolution[0] * resolution[1]
        
        # Loop through the MSE values and store the data for each algorithm
        for algorithm, mse in mse_values.items():
            if algorithm not in algorithm_data:
                algorithm_data[algorithm] = {'pixel_counts': [], 'mses': []}
            algorithm_data[algorithm]['pixel_counts'].append(pixel_count)
            algorithm_data[algorithm]['mses'].append(mse)

# Plotting
plt.figure(figsize=(10, 6))

# Define a color map for different algorithms
colors = plt.cm.tab10(np.linspace(0, 1, len(algorithm_data)))

# Plot each algorithm's MSE vs. Pixel Count with different colors
for (algorithm, data), color in zip(algorithm_data.items(), colors):
    plt.scatter(data['pixel_counts'], data['mses'], label=algorithm, color=color)

# Add labels, title, and legend
plt.xlabel('Pixel Count')
plt.ylabel('MSE')
plt.title('Pixel Count vs MSE for Different Algorithms')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig("output/resolutions_mses.jpg", dpi=300, bbox_inches='tight')
# plt.show()

