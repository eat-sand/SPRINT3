import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import pytz
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Plotting particles function
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

    plt.title(f"Cluster {cluster_num}")
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

# Function for loading data files
def process_dict_files(folder_path, unix):
    cluster_data = {} # {cluster_id: {'coords': array, 'E_track': array, 'time': float}}
    keylist = []

    cluster_counter = 1
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

    for file_index, file_name in enumerate(file_list):
        file_path = os.path.join(folder_path, file_name)

        if os.stat(file_path).st_size == 0:
            continue

        try:
            data = np.load(file_path, allow_pickle=True).item()
            keylist.append(len(data))

            for cluster_id in data:
                coords = np.array(data[cluster_id]['coords'])
                energies = np.array(data[cluster_id]['E_track'])
                toa = unix[file_index]

                if coords.shape[0] != energies.shape[0]:
                    continue

                cluster_data[cluster_counter] = {
                    'x': coords[:, 0],
                    'y': coords[:, 1],
                    'energy': energies,
                    'time': toa
                }

                cluster_counter += 1

        except Exception as e:
            print(f"Skipped file {file_name} due to error: {e}")
            continue

    return cluster_data, keylist

# Load metadata and convert to UNIX time
metadata = np.load(r'C:\Users\debbi\Desktop\ABsatBalloon\MediPixMetadata.npy', allow_pickle=True).item()
epoch = metadata['epoch']

local_tz = pytz.timezone("America/Edmonton")
utc_epoch = [local_tz.localize(t).astimezone(pytz.utc) for t in epoch]
unix_epoch = [dt.timestamp() for dt in utc_epoch]

# Load cluster data
folder_path = r"C:\Users\debbi\Desktop\ABsatBalloon\ClassifiedForGrating"
cluster_data, keylist = process_dict_files(folder_path, unix_epoch)

# ACIS and ASCA grading outlines
grid_values = {
    (-1, 1): 32, (0, 1): 64, (1, 1): 128,
    (-1, 0): 8,  (0, 0): 0,  (1, 0): 16,
    (-1, -1): 1, (0, -1): 2, (1, -1): 4
}

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

acis_grades = {}
asca_grades = {}

# Grade clusters
for cluster_id, cluster in cluster_data.items():
    x_vals = cluster['x']
    y_vals = cluster['y']
    e_vals = cluster['energy']

    if len(x_vals) == 0:
        continue

    max_idx = np.argmax(e_vals)
    x0, y0 = x_vals[max_idx], y_vals[max_idx]

    grade = 0
    valid = True
    
    for xi, yi in zip(x_vals, y_vals):
        dx = xi - x0
        dy = yi - y0
        if (dx, dy) in grid_values:
            grade += grid_values[(dx, dy)]
        else:
            valid = False
            break

    if valid:
        acis_grades[cluster_id] = grade
        asca_grades[cluster_id] = acis_to_asca.get(grade, 7)

# ASCA format-preserved cluster groupings, with grade
standard_asca = {
    k: {
        'coords': np.column_stack((cluster_data[k]['x'], cluster_data[k]['y'])),
        'energy': cluster_data[k]['energy'],
        'time': cluster_data[k]['time'],
        'asca_grade': asca_grades[k]
    }
    for k in asca_grades if asca_grades[k] in [0, 2, 3, 4, 6]
}

other_asca = {
    k: {
        'coords': np.column_stack((cluster_data[k]['x'], cluster_data[k]['y'])),
        'energy': cluster_data[k]['energy'],
        'time': cluster_data[k]['time'],
        'asca_grade': asca_grades[k]
    }
    for k in asca_grades if asca_grades[k] in [1, 5, 7]
}

standard_asca1 = {k: v for k, v in asca_grades.items() if v in [0, 2, 3, 4, 6]}
other_asca1 = {k: v for k, v in asca_grades.items() if v in [1, 5, 7]}

total_energy = {k: sum(cluster_data[k]['energy']) for k in cluster_data}
xrays = {k: v for k, v in total_energy.items() if len(cluster_data[k]['energy']) < 5}
not_xrays = {k: v for k, v in total_energy.items() if len(cluster_data[k]['energy']) >= 5}

for cluster_num, grade in acis_grades.items():
    print(f"Cluster {cluster_num}: Grade = {grade}")

# Plot first 5 clusters from particle_grades
for cluster_num in list(acis_grades.keys())[:5]:
    cluster = cluster_data[cluster_num]
    x_vals = cluster['x']
    y_vals = cluster['y']
    e_vals = cluster['energy']
    plot_cluster(x_vals, y_vals, e_vals, cluster_num)

# Whole Spectrum Histogram
total_energy_values = list(total_energy.values())

bin_edges = np.arange((min(total_energy_values) - 1), (max(total_energy_values) + 1) , 1)
counts, edges = np.histogram(total_energy_values, bins=bin_edges)
bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

plt.figure(figsize=(15, 7))
plt.bar(bin_centers, counts, width=1, color="indigo", align="center")
plt.grid()
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.show()

# Normalized Histogram
bin_edges = np.histogram_bin_edges(total_energy_values, bins='fd')
plt.figure(figsize=(15, 7))
plt.hist(total_energy_values, bins=bin_edges, color='indigo', density=True)
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.show()

# Histogram of particles less than 5
xrays_values = list(xrays.values())

bin_edges = np.arange((min(xrays_values) - 1), (max(xrays_values) + 1) , 1)
counts, edges = np.histogram(xrays_values, bins=bin_edges)
bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

plt.figure(figsize=(15, 7))
plt.bar(bin_centers, counts, width=1, color="seagreen", align="center")
plt.grid()
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.show()

# Normalized Histogram
bin_edges = np.histogram_bin_edges(xrays_values, bins='fd')
plt.figure(figsize=(15, 7))
plt.hist(xrays_values, bins=bin_edges, color='seagreen', density=True)
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.show()

# Graph Not X-rays
not_xrays_values = list(not_xrays.values())

bin_edges = np.arange((min(not_xrays_values) - 1), (max(not_xrays_values) + 1) , 1)
counts, edges = np.histogram(not_xrays_values, bins=bin_edges)
bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

plt.figure(figsize=(15, 7))
plt.bar(bin_centers, counts, width=1, color="darkblue", align="center")
plt.grid()
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.show()

# Normalized Histogram
bin_edges = np.histogram_bin_edges(not_xrays_values, bins='fd')
plt.figure(figsize=(15, 7))
plt.hist(not_xrays_values, bins=bin_edges, color='darkblue', density=True)
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.show()

# Histogram of all ACIS Grades
grade_values = list(acis_grades.values())

plt.figure(figsize=(15, 7))
plt.hist(grade_values, bins=range(max(grade_values) + 2), color='peru', align='left')
plt.xlabel("Event Grades")
plt.ylabel("Count")
plt.grid()
plt.show()

# ACSA Overall Count
asca_histogram = list(asca_grades.values())

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

# Standard
sasca = [v['asca_grade'] for v in standard_asca.values()]
bin_edges = np.arange(-0.5, 7.5 + 1e-5, 1)
counts, edges = np.histogram(sasca, bins=bin_edges)
bin_centers = np.arange(0, 8)

plt.figure(figsize=(10, 7))
plt.bar(bin_centers, counts, width=0.7, color="darkturquoise", align="center")
plt.xlabel("Event Grades")
plt.ylabel("Count")
plt.xticks(np.arange(0, 8))
plt.grid(axis='y')
plt.show()

# Other
oasca = [v['asca_grade'] for v in other_asca.values()]
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

for key in standard_asca1:
    if key in total_energy:
        s_asca_e.append(total_energy[key])


bin_edges = np.arange((min(s_asca_e) - 1), (max(s_asca_e) + 1) , 1)
counts, edges = np.histogram(s_asca_e, bins=bin_edges)
bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

plt.figure(figsize=(15, 7))
plt.bar(bin_centers, counts, width=1, color="olivedrab", align="center")
plt.grid()
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.show()

bin_edges = np.histogram_bin_edges(s_asca_e, bins='fd')
plt.figure(figsize=(15, 7))
plt.hist(s_asca_e, bins=bin_edges, color='olivedrab', density=True)
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.show()

# Energy histogram of the Other ACSA
o_asca_e = []

for key in other_asca1:
    if key in total_energy:
        o_asca_e.append(total_energy[key])

for key in other_asca:
    if key in total_energy:
        o_asca_e.append(total_energy[key])

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

# Comparision of xrays by size and grading
bin_edges1 = np.histogram_bin_edges(s_asca_e, bins='fd')
bin_edges2 = np.histogram_bin_edges(xrays_values, bins='fd')

plt.figure(figsize=(15, 7))
plt.hist(s_asca_e, bins=bin_edges1, density=True, color='magenta', alpha=0.5, label='Standard ASCA')
plt.hist(xrays_values, bins=bin_edges2, density=True, color='olivedrab', alpha=0.5, label='Particles less than 5 pixels')
plt.xlabel("Energy (keV)")
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.legend()
plt.show()

# Comparision of not xrays determined by size and grading
bin_edges1 = np.histogram_bin_edges(o_asca_e, bins='fd')
bin_edges2 = np.histogram_bin_edges(not_xrays_values, bins='fd')

plt.figure(figsize=(15, 7))
plt.hist(o_asca_e, bins=bin_edges1, density=True, color='magenta', alpha=0.5, label='Filtered ASCA')
plt.hist(not_xrays_values, bins=bin_edges2, density=True, color='dodgerblue', alpha=0.5, label='Particles more than 5 pixels')
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
    if key in total_energy:
        s_asca_e.append(total_energy[key])

for cluster_num, grade in asca_grades.items():
    if grade == 0:
        zero[cluster_num] = grade
        zero_energy.append(total_energy[cluster_num])
    elif grade == 1:
        one[cluster_num] = grade
        one_energy.append(total_energy[cluster_num])
    elif grade == 2:
        two[cluster_num] = grade
        two_energy.append(total_energy[cluster_num])
    elif grade == 3:
        three[cluster_num] = grade
        three_energy.append(total_energy[cluster_num])
    elif grade == 4:
        four[cluster_num] = grade
        four_energy.append(total_energy[cluster_num])
    elif grade == 5:
        five[cluster_num] = grade
        five_energy.append(total_energy[cluster_num])
    elif grade == 6:
        six[cluster_num] = grade
        six_energy.append(total_energy[cluster_num])
    elif grade == 7:
        seven[cluster_num] = grade
        seven_energy.append(total_energy[cluster_num])

# Plot each grade in a histogram
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

for energy, label, color in energy_arrays:
    bin_edges = np.histogram_bin_edges(energy, bins='fd')
    plt.hist(energy, bins=bin_edges, density=True, alpha=0.5, label=label, color=color)

plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.legend()
plt.show()

# Histogram of the seperated standard grades
group1 = [
    (zero_energy, "Grade 0", "red"),
    (two_energy, "Grade 2", "yellow"),
    (three_energy, "Grade 3", "green"),
    (four_energy, "Grade 4", "blue"),
    (six_energy, "Grade 6", "brown"),
]

plt.figure(figsize=(15, 7))

for energy, label, color in group1:
    bin_edges = np.histogram_bin_edges(energy, bins='fd')
    plt.hist(energy, bins=bin_edges, density=True, alpha=0.5, label=label, color=color)

plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.legend()
plt.title("Grades 0, 2, 3, 4, 6")
plt.show()

# Histogram of the seperated other grades
group2 = [
    (one_energy, "Grade 1", "orange"),
    (five_energy, "Grade 5", "purple"),
    (seven_energy, "Grade 7", "black"),
]

plt.figure(figsize=(15, 7))

for energy, label, color in group2:
    bin_edges = np.histogram_bin_edges(energy, bins='fd')
    plt.hist(energy, bins=bin_edges, density=True, alpha=0.5, label=label, color=color)

plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.legend()
plt.title("Grades 1, 5, 7")
plt.show()

# Flux Calculations

keylist = np.array(keylist)
keylist_counts = keylist/0.3

# Calculate the Counts per Second
dtime = 50
cps = []
adjusted_times = []

i = 0
while i < len(unix_epoch):
    group_keys = [keylist_counts [i]]
    group_times = [unix_epoch[i]]
    j = i + 1

    while j < len(unix_epoch) and (unix_epoch[j] - unix_epoch[i]) < dtime:
        group_keys.append(keylist_counts [j])
        group_times.append(unix_epoch[j])
        j += 1

    cps_value = sum(group_keys) / len(group_keys)
    avg_time = sum(group_times) / len(group_times)

    cps.append(cps_value)
    adjusted_times.append(avg_time)

    i = j

cps = np.array(cps)
adjusted_times = np.array(adjusted_times)
flux = cps/6.23 # CONSTANT IS SUBJECT TO CHANGE

# Graph the guy
utc_times = [datetime.utcfromtimestamp(t) for t in adjusted_times]

plt.figure(figsize=(15, 7))
plt.plot(utc_times, flux)
plt.xlabel("Time (UTC)")
plt.ylabel('Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
plt.grid()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gcf().autofmt_xdate()
plt.legend()
plt.show()

# Flux for graded values

standard_asca_times = [v['time'] for v in standard_asca.values()]
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
adjusted_time2 = np.array(asca_adjusted_times)
ascaflux = asca_cps/6.23 

# Graph
asca_utc_times = [datetime.utcfromtimestamp(t) for t in asca_adjusted_times]

plt.figure(figsize=(15, 7))
plt.plot(asca_utc_times, ascaflux)
plt.xlabel("Time (UTC)")
plt.ylabel('Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
plt.grid()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gcf().autofmt_xdate()
plt.legend()
plt.show()

plt.figure(figsize=(15, 7))
plt.plot(utc_times, flux, label='All Particles', color = 'purple')
plt.plot(asca_utc_times, ascaflux, linestyle='dotted', label="Standard ASCA", color = 'orange')
plt.xlabel("Time (UTC)")
plt.ylabel('Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
plt.grid()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gcf().autofmt_xdate()
plt.legend()
plt.show()