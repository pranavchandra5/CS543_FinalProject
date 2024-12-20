import os
import json
import matplotlib.pyplot as plt

folder_path = './json_files'

pixel_counts = {}
mse_values = {}

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        with open(os.path.join(folder_path, filename), 'r') as file:
            data = json.load(file)            
            resolution = data['resolution']
            pixel_count = resolution[0] * resolution[1]
            mse_data = data['mses']
            average_mse = sum(mse_data.values()) / len(mse_data)
            if pixel_count not in pixel_counts:
                pixel_counts[pixel_count] = []
                mse_values[pixel_count] = []
            pixel_counts[pixel_count].append(pixel_count)
            mse_values[pixel_count].append(average_mse)

avg_mse_by_pixel_count = {pixel_count: sum(mse_values[pixel_count]) / len(mse_values[pixel_count]) for pixel_count in mse_values}

sorted_pixel_counts = sorted(avg_mse_by_pixel_count.keys())
sorted_avg_mse = [avg_mse_by_pixel_count[count] for count in sorted_pixel_counts]

plt.figure(figsize=(10, 6))
plt.plot(sorted_pixel_counts, sorted_avg_mse, marker='o', linestyle='-', color='b')
plt.xlabel('Resolution (Pixel Count)')
plt.ylabel('Average MSE')
plt.title('Average MSE vs Resolution')
plt.grid(True)
# plt.show()
plt.savefig("resolutions_mses.jpg", dpi=300, bbox_inches='tight')
