import os
import json
import matplotlib.pyplot as plt

directory_path = './output'
pixel_counts = []
mses = []

for filename in os.listdir(directory_path):
    if filename.endswith('.json'):
        file_path = os.path.join(directory_path, filename)        
        with open(file_path, 'r') as f:
            data = json.load(f)        
        resolution = data['resolution']
        mse_values = data['mses']        
        pixel_count = resolution[0] * resolution[1]        
        mse = mse_values.get('Graph_TSP_MST')
        pixel_counts.append(pixel_count)
        mses.append(mse)

# Plot the pixel count vs MSE
plt.scatter(pixel_counts, mses)
plt.xlabel('Resolution (Pixel Count)')
plt.ylabel('MSE')
plt.title('Resolution vs MSE')
plt.grid(True)
plt.savefig("output/resolutions_mses.jpg", dpi=300, bbox_inches='tight')
# plt.show()

