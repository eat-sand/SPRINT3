import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import defaultdict
import scipy.signal as sp

# Open read and extract the data of the txt files
def process_files_in_folder(folder_path):
    
    pixel_coords_list = []
    time_hit_list = []
    energy_list = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)

            # Check if file is empty
            if os.stat(file_path).st_size == 0:
                continue

            try:
                data = np.loadtxt(file_path, comments='#')

                # Ensure it's a 2D array and has at least 4 columns
                if data.ndim == 1:
                    data = data.reshape(1, -1) 
                if data.shape[1] < 4:
                    continue

                pixel_coords_list.append(data[:, 0])  # Linear Pixel Coordinates
                time_hit_list.append(data[:, 1])  # Time of hit in 40 MHz
                energy_list.append(data[:, 3])  # Energy in keV

            except Exception as e:
                continue
    
    return pixel_coords_list, time_hit_list, energy_list

folder_path = r"" # INSERT FILE PATH HERE
pixel_coords, time_hit, energy = process_files_in_folder(folder_path)

# Convert to flat np arrays
pixel_coords = np.concatenate(pixel_coords)
energy = np.concatenate(energy)
time_hit = np.concatenate(time_hit)

# Compute x, y pixel positions
x = pixel_coords % 256
y = pixel_coords // 256

# Clustering by time
time_cluster = []

# List to keep track of values that have already been clustered
visited = [False] * len(time_hit)
cluster_index = [0.] * len(time_hit) 

current_cluster = 1
for i in range(len(time_hit)):
    # if visited[i]:
    if cluster_index[i]:
        continue  # Skip if already clustered

    tcluster = [time_hit[i]] 
    minimum = time_hit[i]
    maximum = time_hit[i]
    visited[i] = True # Checked
    cluster_index[i] = current_cluster

    for j in range(i + 1, len(time_hit)):
        if visited[j]:
            continue  # Skip if already clustered

        if abs(minimum - time_hit[j]) < 5 or abs(maximum - time_hit[j]) < 5:
            tcluster.append(time_hit[j])
            visited[j] = True  # Checked
            cluster_index[j] = current_cluster

            # Update the max, min values
            minimum = min(minimum, time_hit[j])
            maximum = max(maximum, time_hit[j])

        else:
            break

    time_cluster.append(tcluster)
    current_cluster += 1

cluster_index = np.array(cluster_index)

# Energy Clustering
def cluster_energy (energy, index):
    
    cluster = defaultdict(list)
   
    for e, c in zip(energy, index): 
        cluster[c].append(e)
    
    return cluster

e_cluster = cluster_energy(energy, cluster_index)

# Calculate the sum of each key
total_energy = []
total = 1
while total <= len(e_cluster):
    total_energy.append(sum(e_cluster[total]))
    total += 1

# Sort Out X-Rays based on cluster size
x_rays = []
not_xrays = []

i_cluster = 1

while i_cluster < len(e_cluster):

    if len(e_cluster[i_cluster]) < 5:
        x_rays.append(sum(e_cluster[i_cluster]))

    else: 
        not_xrays.append(sum(e_cluster[i_cluster]))
    
    i_cluster += 1

# Sorting x-rays with the Chandra x-ray ACIS grading system
grid_values = { 
    (-1, 1): 32, (0,1): 64, (1, 1): 128,
    ( -1,0): 8,  ( 0, 0): 0,  ( 1,0): 16,
    ( -1, -1): 1,  ( 0, -1): 2,  ( 1, -1): 4
} 

acis_grades = {}

for cluster_num in cluster_index:
    # Mask to get the indices of the current cluster
    cluster_mask = cluster_index == cluster_num

    x_vals = x[cluster_mask]
    y_vals = y[cluster_mask]
    e_vals = energy[cluster_mask]

    # Find pixel with the highest energy
    max_idx = np.argmax(e_vals)
    x0, y0 = x_vals[max_idx], y_vals[max_idx]

    # Calculate grade for cluster
    grade = 0
    valid_cluster = True
    for xi, yi in zip(x_vals, y_vals):
        dx = xi - x0
        dy = yi - y0
        if (dx, dy) in grid_values:
            grade += grid_values[(dx, dy)]
        else:
            valid_cluster = False
            break

    # Only store if all pixels on 3x3 grid
    if valid_cluster:
        acis_grades[cluster_num] = grade

# Convert ACIS grades to ASCA grades
acis_to_asca = {
    0: 0,
    64: 2, 65: 2, 68: 2, 69: 2, 2: 2, 34: 2, 130: 2, 162: 2,
    8: 3, 12: 3, 136: 3, 140: 3,
    16: 4, 17: 4, 48: 4, 49: 4,
    72: 6, 76: 6, 104: 6, 108: 6, 10: 6, 11: 6, 138: 6, 139: 6,
    18: 6, 22: 6, 50: 6, 54: 6, 80: 6, 81: 6, 208: 6, 209: 6,
    1: 1, 4: 1, 32: 1, 128: 1, 5: 1, 33: 1, 132: 1, 160: 1,
    36: 1, 129: 1, 37: 1, 133: 1, 161: 1, 164: 1, 165: 1,
    3: 5, 6: 5, 9: 5, 20: 5, 40: 5, 96: 5, 144: 5, 192: 5,
    13: 5, 21: 5, 35: 5, 38: 5, 44: 5, 52: 5, 97: 5, 100: 5,
    131: 5, 134: 5, 137: 5, 145: 5, 168: 5, 176: 5, 193: 5,
    196: 5, 53: 5, 101: 5, 141: 5, 163: 5, 166: 5, 172: 5,
    177: 5, 197: 5,
}

asca = {}
standard_asca = {}
other_asca = {}

# Convert
for cluster_num, grade in acis_grades.items():
    asca_value = acis_to_asca.get(grade, 7)
    asca[cluster_num] = asca_value

standard_asca = {k: v for k, v in asca.items() if v in [0, 2, 3, 4, 6]}
other_asca = {k: v for k, v in asca.items() if v in [1,5,7]}

