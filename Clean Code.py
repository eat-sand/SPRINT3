import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import defaultdict
import scipy.signal as sp

# Load data function
def process_files_in_folder(folder_path):
    pixel_coords_list = []
    time_of_arrival_list = []
    energy_list = []
    unix_list = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)

            if os.stat(file_path).st_size == 0:
                print(f"Skipping empty file: {file_name}")
                continue

            try:
                data = np.loadtxt(file_path, comments='#')

                if data.ndim == 1:
                    data = data.reshape(1, -1)

                if data.shape[1] < 4:
                    print(f"Skipping {file_name}: Less than 4 columns found.")
                    continue

                unix_time = None
                with open(file_path, 'r') as file:
                    for line in file:
                        if line.strip().startswith('# Start of measurement - unix time: '):
                            unix_str = line.strip().split(': ')[-1]
                            try:
                                unix_time = float(unix_str)
                            except ValueError:
                                print(f"Could not convert timestamp to float in {file_name}")
                                unix_time = None
                            break

                pixel_coords_list.append(data[:, 0])
                time_of_arrival_list.append(data[:, 1])
                energy_list.append(data[:, 3])
                if unix_time is not None:
                    unix_list.extend([unix_time] * data.shape[0])
                else:
                    unix_list.extend([None] * data.shape[0])

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    return pixel_coords_list, time_of_arrival_list, energy_list, unix_list

folder_path = r"" # INSERT FILE PATH HERE
linear_pixel_coords, time_of_arrival, energy, unix_time = process_files_in_folder(folder_path)

# Convert to flat np arrays
linear_pixel_coords = np.concatenate(linear_pixel_coords)
energy = np.concatenate(energy)
time_of_arrival = np.concatenate(time_of_arrival)

# Compute x, y pixel positions
x = linear_pixel_coords % 256
y = linear_pixel_coords // 256

# Clustering by time
time_cluster = []

# List to keep track of values that have already been clustered
timclustered_energy = []

visited = [False] * len(time_of_arrival)
cluster_index = [0.] * len(time_of_arrival) 

current_cluster = 1

for i in range(len(time_of_arrival)):
    # if visited[i]:
    if cluster_index[i]:
        continue

    tcluster = [time_of_arrival[i]] 
    minimum = time_of_arrival[i]
    maximum = time_of_arrival[i]
    visited[i] = True # Checked
    cluster_index[i] = current_cluster

    for j in range(i + 1, len(time_of_arrival)):
        if visited[j]:
            continue

        if abs(minimum - time_of_arrival[j]) < 5 or abs(maximum - time_of_arrival[j]) < 5:
            tcluster.append(time_of_arrival[j])
            visited[j] = True  # Checked
            cluster_index[j] = current_cluster

            minimum = min(minimum, time_of_arrival[j])
            maximum = max(maximum, time_of_arrival[j])

        else:
            break

    timclustered_energy.append(tcluster)
    current_cluster += 1

cluster_index = np.array(cluster_index)

# Convert clock ticks to seconds
converted_toa = time_of_arrival / 40e6

# Cluster the UNIX, TOA and energy of each particle
def cluster_function (values, index):
    
    cluster = defaultdict(list)
   
    for v, c in zip(values, index): 
        cluster[c].append(v)
    
    return cluster

clustered_energy = cluster_function(energy, cluster_index)
clustered_unix = cluster_function(unix_time, cluster_index)
clustered_toa = cluster_function(converted_toa, cluster_index)


# Time of arrival and total energy of each cluster
time = []
energy_summed = []

for key in clustered_toa:
        ctoa = min(clustered_toa[key])
        cunix = (clustered_unix[key])
        cunix = cunix[0]
        ctu = ctoa + cunix

        time.append(ctu)

total = 1
while total <= len(clustered_energy):
    energy_summed.append(sum(clustered_energy[total]))
    total += 1

# Attatch the energy and time of each cluster to each cluster
particles = {}

for idx, (e_sum, t) in enumerate(zip(energy_summed, time), start=1):
    particles[idx] = {
        'energy': e_sum,
        'time arrived': t
    }

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

# Flux calculation
graph_times = [v['time arrived'] for v in particles.values()]
graph_times = sorted(graph_times)

# Calculate the Counts per Second
dtime = 50
cps = []
adjusted_times = []

i = 0
while i < len(graph_times):
    group_keys = []
    group_times = [graph_times[i]]
    j = i + 1

    while j < len(graph_times) and (graph_times[j] - graph_times[i]) < dtime:
        group_keys.append(1)
        group_times.append(graph_times[j])
        j += 1

    cps_value = sum(group_keys) / dtime 
    avg_time = sum(group_times) / len(group_times)

    cps.append(cps_value)
    adjusted_times.append(avg_time)

    i = j

cps = np.array(cps)