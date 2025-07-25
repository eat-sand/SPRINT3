import os
import numpy as np
import re
import matplotlib.pyplot as plt
import collections
from collections import defaultdict
import scipy.signal as sp
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Plotting a cluster on the 3x3 grid with energy and pixel coordinates
def plot_cluster(x_vals, y_vals, e_vals, cluster_num):
    max_idx = np.argmax(e_vals)
    x0, y0 = x_vals[max_idx], y_vals[max_idx]

    dx = x_vals - x0
    dy = y_vals - y0

    plt.figure(figsize=(5, 5))
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(-2, 3, 1))
    plt.yticks(np.arange(-2, 3, 1))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(f"X-ray Cluster {cluster_num}")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot each point
    for i in range(len(dx)):
        color = 'red' if i == max_idx else 'blue'
        plt.plot(dx[i], dy[i], 'o', color=color)
        plt.text(
            dx[i] + 0.1,
            dy[i] + 0.1,
            f"E={e_vals[i]:.1f} keV\n({x_vals[i]},{y_vals[i]})",
            fontsize=7,
            ha='left'
        )

    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

def grades_hist(histenergy):
    bin_edges = np.histogram_bin_edges(histenergy, bins='fd')
    plt.figure(figsize=(15, 7))
    plt.hist(histenergy, bins=bin_edges, color='pink', density=True)
    plt.xlabel("Energy (keV)")
    plt.ylabel("Count")
    plt.xlim(left=0)
    plt.grid()
    plt.show()

# Open read and extract the data of the txt files

def process_files_in_folder(folder_path):
    pixel_coords_list = []
    time_of_arrival_list = []
    energy_list = []
    unix_list = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)

            # Check if file is empty
            if os.stat(file_path).st_size == 0:
                print(f"Skipping empty file: {file_name}")
                continue

            try:
                data = np.loadtxt(file_path, comments='#')

                # Reshape if it's a single row
                if data.ndim == 1:
                    data = data.reshape(1, -1)

                # Skip files with fewer than 4 columns
                if data.shape[1] < 4:
                    print(f"Skipping {file_name}: Less than 4 columns found.")
                    continue

                # Extract UNIX timestamp
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

                # Append data
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

# Load data
folder_path = r"C:\Users\debbi\Desktop\Hardpix Files\22-05-2025"
linear_pixel_coords, time_of_arrival, energy, unix_time = process_files_in_folder(folder_path)

# Convert object arrays to flat NumPy arrays
linear_pixel_coords = np.concatenate(linear_pixel_coords)
energy = np.concatenate(energy)
time_of_arrival = np.concatenate(time_of_arrival)

# Compute x, y pixel positions
x = linear_pixel_coords % 256
y = linear_pixel_coords // 256

# Graph Energy
plt.figure(figsize=(15, 15))
plt.scatter(x, y, s=10, c=np.log(energy))
cbar = plt.colorbar()
cbar.set_label('log(Energy in keV)', size=10)
plt.grid()
plt.show()

# Graph Time
plt.figure(figsize=(15, 15))
plt.scatter(x, y, s=10, c = time_of_arrival)
cbar = plt.colorbar()
cbar.set_label('Hit Time in 40MHz')
plt.grid()
plt.show()

# Clustering by time
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
    visited[i] = True
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

print(len(cluster_index))

# Graph Time Clusters
plt.figure(figsize=(15, 15))
plt.scatter(x, y, s=10, c = cluster_index, cmap='jet')
cbar = plt.colorbar()
cbar.set_label('Cluster')
plt.grid()
plt.show()

# Convert clock ticks to seconds
converted_toa = time_of_arrival / 40e6

def cluster_function (values, index):
    
    cluster = defaultdict(list)
   
    for v, c in zip(values, index): 
        cluster[c].append(v)
    
    return cluster

clustered_energy = cluster_function(energy, cluster_index)
clustered_unix = cluster_function(unix_time, cluster_index)
clustered_toa = cluster_function(converted_toa, cluster_index)

time = []
energy_summed = []

for key in clustered_toa:
        ctoa = max(clustered_toa[key])
        cunix = (clustered_unix[key])
        cunix = cunix[0]
        ctu = ctoa + cunix

        time.append(ctu)

total = 1
while total <= len(clustered_energy):
    energy_summed.append(sum(clustered_energy[total]))
    total += 1

particles = {}

for idx, (e_sum, t) in enumerate(zip(energy_summed, time), start=1):
    particles[idx] = {
        'energy': e_sum,
        'time arrived': t
    }

# Manual Histogram

# Set Bins
bin_edges = np.arange((min(energy_summed) - 1), (max(energy_summed) + 1) , 1)
counts, edges = np.histogram(energy_summed, bins=bin_edges)
bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

# Graph
plt.figure(figsize=(15, 7))
plt.bar(bin_centers, counts, width=1, color="indigo", align="center")
plt.grid()
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.xlim(left=0)
plt.show()

bin_edges = np.histogram_bin_edges(energy_summed, bins='fd')

plt.figure(figsize=(15, 7))
plt.hist(energy_summed, bins=bin_edges, color='indigo', density=True)
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.show()

# Sort the X-rays and electrons out
x_rays = []
not_xrays = []

i_cluster = 1

while i_cluster < len(clustered_energy):

    if len(clustered_energy[i_cluster]) < 5:
        x_rays.append(sum(clustered_energy[i_cluster]))
    else: 
        not_xrays.append(sum(clustered_energy[i_cluster]))
    
    i_cluster += 1

# Graph X-rays

# Set Bins
bin_edges = np.arange((min(x_rays) - 1), (max(x_rays) + 1) , 1)
counts, edges = np.histogram(x_rays, bins=bin_edges)
bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

# Graph
plt.figure(figsize=(15, 7))
plt.bar(bin_centers, counts, width=1, color="seagreen", align="center")
plt.grid()
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.show()

bin_edges = np.histogram_bin_edges(x_rays, bins='fd')
plt.figure(figsize=(15, 7))
plt.hist(energy_summed, bins=bin_edges, color='seagreen', density=True)
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.show()

# Graph No X-rays

# Set Bins
bin_edges = np.arange((min(not_xrays) - 1), (max(not_xrays) + 1) , 1)
counts, edges = np.histogram(not_xrays, bins=bin_edges)
bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

# Graph
plt.figure(figsize=(15, 7))
plt.bar(bin_centers, counts, width=1, color="darkblue", align="center")
plt.grid()
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.show()

bin_edges = np.histogram_bin_edges(not_xrays, bins='fd')
plt.figure(figsize=(15, 7))
plt.hist(not_xrays, bins=bin_edges, color='darkblue', density=True)
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.show()

# Takes 3 hours

# Define the 3x3 grid with the relevant numbers
grid_values = {
    (-1, 1): 32, (0, 1): 64, (1, 1): 128,
    (-1, 0): 8,  (0, 0): 0,  (1, 0): 16,
    (-1, -1): 1, (0, -1): 2, (1, -1): 4
}

# Dictionary to hold the final grade for each particle cluster
particle_grades = {}

for cluster_num in cluster_index:
    cluster_mask = cluster_index == cluster_num

    x_vals = x[cluster_mask]
    y_vals = y[cluster_mask]
    e_vals = energy[cluster_mask]

    max_idx = np.argmax(e_vals)
    x0, y0 = x_vals[max_idx], y_vals[max_idx]

    # Calculate grade for the cluster
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

    if valid_cluster:
        particle_grades[cluster_num] = grade
        print(cluster_num)

# Map ACIS grades to ASCA grades
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

for cluster_num, grade in particle_grades.items():
    asca_value = acis_to_asca.get(grade, 7)
    asca[cluster_num] = asca_value

standard_asca = {k: v for k, v in asca.items() if v in [0, 2, 3, 4, 6]}
other_asca = {k: v for k, v in asca.items() if v in [1,5,7]}

# Histogram of all ACIS Grades
grade_values = list(particle_grades.values())

plt.figure(figsize=(15, 7))
plt.hist(grade_values, bins=range(max(grade_values) + 2), color='peru', align='left')
plt.xlabel("Event Grades")
plt.ylabel("Count")
plt.grid()
plt.show()

# ACSA Overall Count
asca_histogram = list(asca.values())
bin_edges = np.arange(-0.5, 7.5 + 1e-5, 1)
counts, edges = np.histogram(asca_histogram, bins=bin_edges)
bin_centers = np.arange(0, 8)

plt.figure(figsize=(10, 7))
plt.bar(bin_centers, counts, width=0.7, color="rosybrown", align="center")
plt.xlabel("Event Grades")
plt.ylabel("Count")
plt.xticks(np.arange(0, 8))
plt.grid(axis='y')
plt.show()

# ACSA Broken Up Count
fasca = list(standard_asca.values())
bin_edges = np.arange(-0.5, 7.5 + 1e-5, 1)
counts, edges = np.histogram(fasca, bins=bin_edges)
bin_centers = np.arange(0, 8)

plt.figure(figsize=(10, 7))
plt.bar(bin_centers, counts, width=0.7, color="darkturquoise", align="center")
plt.xlabel("Event Grades")
plt.ylabel("Count")
plt.xticks(np.arange(0, 8))
plt.grid(axis='y')
plt.show()

oasca = list(other_asca.values())
bin_edges = np.arange(-0.5, 7.5 + 1e-5, 1)
counts, edges = np.histogram(oasca, bins=bin_edges)
bin_centers = np.arange(0, 8)

plt.figure(figsize=(10, 7))
plt.bar(bin_centers, counts, width=0.7, color="darkturquoise", align="center")
plt.xlabel("Event Grades")
plt.ylabel("Count")
plt.xticks(np.arange(0, 8))
plt.grid(axis='y')
plt.show()

# Graph Standard ACSA
s_asca_e = []

for key in standard_asca:
    if key in clustered_energy:
        s_asca_e.append(sum(clustered_energy[key]))

bin_edges = np.arange((min(s_asca_e) - 1), (max(s_asca_e) + 1) , 1)
counts, edges = np.histogram(s_asca_e, bins=bin_edges)
bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

# Graph
plt.figure(figsize=(15, 7))
plt.bar(bin_centers, counts, width=1, color="olivedrab", align="center")
plt.grid()
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.xlim(left=0)
plt.show()

bin_edges = np.histogram_bin_edges(s_asca_e, bins='fd')
plt.figure(figsize=(15, 7))
plt.hist(s_asca_e, bins=bin_edges, color='olivedrab', density=True)
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.show()

# Graph Other ACSA
o_asca_e = []

for key in other_asca:
    if key in clustered_energy:
        o_asca_e.append(sum(clustered_energy[key]))

bin_edges = np.arange((min(o_asca_e) - 1), (max(o_asca_e) + 1) , 1)
counts, edges = np.histogram(o_asca_e, bins=bin_edges)
bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

plt.figure(figsize=(15, 7))
plt.bar(bin_centers, counts, width=1, color="dodgerblue", align="center")
plt.grid()
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.show()

bin_edges = np.histogram_bin_edges(o_asca_e, bins='fd')
plt.figure(figsize=(15, 7))
plt.hist(o_asca_e, bins=bin_edges, color='dodgerblue', density=True)
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.show()

# Comparison of xrays by size vs grading
bin_edges1 = np.histogram_bin_edges(s_asca_e, bins='fd')
bin_edges2 = np.histogram_bin_edges(x_rays, bins='fd')

plt.figure(figsize=(15, 7))
plt.hist(s_asca_e, bins=bin_edges1, density=True, color='magenta', alpha=0.5, label='Standard ASCA')
plt.hist(x_rays, bins=bin_edges2, density=True, color='olivedrab', alpha=0.5, label='Particles less than 5 pixels')
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.legend()
plt.show()

# Comparison of not xrays by size vs grading
bin_edges1 = np.histogram_bin_edges(o_asca_e, bins='fd')
bin_edges2 = np.histogram_bin_edges(not_xrays, bins='fd')

plt.figure(figsize=(15, 7))
plt.hist(o_asca_e, bins=bin_edges1, density=True, color='magenta', alpha=0.5, label='Filtered ASCA')
plt.hist(not_xrays, bins=bin_edges2, density=True, color='dodgerblue', alpha=0.5, label='Particles more than 5 pixels')
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.legend()
plt.show()

# Sort clusters into each ASCA grade
zero = {}
one = {}
two = {}
three = {}
four = {}
five = {}
six = {}
seven = {}

# Sort energies
zero_energy = []
one_energy = []
two_energy = []
three_energy = []
four_energy = []
five_energy = []
six_energy = []
seven_energy  = []

for key in standard_asca:
    if key in clustered_energy:
        s_asca_e.append(sum(clustered_energy[key]))

for cluster_num, grade in asca.items():
    if grade == 0:
        zero[cluster_num] = grade
        zero_energy.append(sum(clustered_energy[cluster_num]))
    elif grade == 1:
        one[cluster_num] = grade
        one_energy.append(sum(clustered_energy[cluster_num]))
    elif grade == 2:
        two[cluster_num] = grade
        two_energy.append(sum(clustered_energy[cluster_num]))
    elif grade == 3:
        three[cluster_num] = grade
        three_energy.append(sum(clustered_energy[cluster_num]))
    elif grade == 4:
        four[cluster_num] = grade
        four_energy.append(sum(clustered_energy[cluster_num]))
    elif grade == 5:
        five[cluster_num] = grade
        five_energy.append(sum(clustered_energy[cluster_num]))
    elif grade == 6:
        six[cluster_num] = grade
        six_energy.append(sum(clustered_energy[cluster_num]))
    elif grade == 7:
        seven[cluster_num] = grade
        seven_energy.append(sum(clustered_energy[cluster_num]))

# Histogram of seperated grades
energy_arrays = [
    (zero_energy, "Grade 0", "red"),
    (one_energy, "Grade 1", "orange"),
    (two_energy, "Grade 2", "yellow"),
    (three_energy, "Grade 3", "green"),
    (four_energy, "Grade 4", "blue"),
    (five_energy, "Grade 5", "purple"),
    (six_energy, "Grade 6", "brown"),
    (seven_energy, "Grade 7", "black"),
]

plt.figure(figsize=(15, 7))

for sets, label, color in energy_arrays:
    bin_edges = np.histogram_bin_edges(sets, bins='fd')
    plt.hist(sets, bins=bin_edges, density=True, alpha=0.5, label=label, color=color)

plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.legend()
plt.show()

group1 = [
    (zero_energy, "Grade 0", "red"),
    (two_energy, "Grade 2", "yellow"),
    (three_energy, "Grade 3", "green"),
    (four_energy, "Grade 4", "blue"),
    (six_energy, "Grade 6", "brown"),
]

plt.figure(figsize=(15, 7))

for sets, label, color in group1:
    bin_edges = np.histogram_bin_edges(sets, bins='fd')
    plt.hist(sets, bins=bin_edges, density=True, alpha=0.5, label=label, color=color)

plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.legend()
plt.title("Grades 0, 2, 3, 4, 6")
plt.show()

group2 = [
    (one_energy, "Grade 1", "orange"),
    (five_energy, "Grade 5", "purple"),
    (seven_energy, "Grade 7", "black"),
]

plt.figure(figsize=(15, 7))

for sets, label, color in group2:
    bin_edges = np.histogram_bin_edges(sets, bins='fd')
    plt.hist(sets, bins=bin_edges, density=True, alpha=0.5, label=label, color=color)

plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.legend()
plt.title("Grades 1, 5, 7")
plt.show()

# Graphs of the partilces in each grade
cluster_sources = [
    list(zero.keys())[:5],
    list(one.keys())[:5],
    list(two.keys())[:5],
    list(three.keys())[:5],
    list(four.keys())[:5],
    list(five.keys())[:5],
    list(six.keys())[:5],
    list(seven.keys())[:5],
]

for clusters in cluster_sources:
    for cluster_num in clusters:
        cluster_mask = cluster_index == cluster_num
        x_vals = x[cluster_mask]
        y_vals = y[cluster_mask]
        energy = np.array(energy)
        e_vals = energy[cluster_mask]
        plot_cluster(x_vals, y_vals, e_vals, cluster_num)

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
# flux = cps/6.23 # CONSTANT IS SUBJECT TO CHANGE

# Graph
utc_times = [datetime.utcfromtimestamp(t) for t in adjusted_times]

plt.figure(figsize=(15, 7))
plt.plot(utc_times, cps)
plt.xlabel("Time (UTC)")
plt.ylabel('Counts per Second')
plt.grid()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gcf().autofmt_xdate()
plt.legend()
plt.show()

asca_standard_particles =  {
    k: particles[k] for k in standard_asca if k in particles
}

standard_asca_times = [v['time arrived'] for v in asca_standard_particles.values()]
standard_asca_times = sorted(standard_asca_times)

dtime = 50
asca_cps = []
asca_adjusted_times = []

i = 0
while i < len(standard_asca_times):
    group_keys = []
    group_times = [standard_asca_times[i]]
    j = i + 1

    while j < len(standard_asca_times) and (standard_asca_times[j] - standard_asca_times[i]) < dtime:
        group_keys.append(1)
        group_times.append(standard_asca_times[j])
        j += 1

    cps_value = sum(group_keys) / dtime 
    avg_time = sum(group_times) / len(group_times)

    asca_cps.append(cps_value)
    asca_adjusted_times.append(avg_time)

    i = j

asca_cps = np.array(asca_cps)
# ascaflux = asca_cps/6.23 # CONSTANT IS SUBJECT TO CHANGE

# Graph
asca_utc_times = [datetime.utcfromtimestamp(t) for t in asca_adjusted_times]

plt.figure(figsize=(15, 7))
plt.plot(asca_utc_times, asca_cps)
plt.xlabel("Time (UTC)")
plt.ylabel('Counts per Second')
plt.grid()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gcf().autofmt_xdate()
plt.legend()
plt.show()

# Overlapped
plt.figure(figsize=(15, 7))
plt.plot(utc_times, cps, label='All Particles', color = 'orange')
plt.plot(asca_utc_times, asca_cps, label="Standard ASCA", color = 'purple', linestyle = 'dotted')
plt.xlabel("Time (UTC)")
plt.ylabel('Counts per Second')
plt.grid()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gcf().autofmt_xdate()
plt.legend()
plt.show()