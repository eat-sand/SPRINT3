import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import defaultdict
import scipy.signal as sp
from datetime import datetime
import matplotlib.dates as mdates

# Raw Data File
data1 = np.loadtxt('rawHits_RN-135_FN-0.txt')

with open('rawHits_RN-135_FN-0.txt', 'r') as file:
    for line in file:
        if line.strip().startswith('# Start of Acquisition (unix):'):
            unix_str = line.strip().split(': ')[-1]
            unix_str = float(unix_str)

data1 = data1[data1[:, 2].argsort()]

x = data1[:,0]
y = data1[:,1]
toa = data1[:,2]
tot = data1[:,3]

# Species Data File
data2 = np.loadtxt('speciesHits_RN-135_FN-0.txt')

asca = data2[:,0]
time_of_arrival = data2[:,1]
energy = data2[:,3]

# Convert clock ticks to seconds
converted_toa = time_of_arrival / 40e6
unix = converted_toa + unix_str

# Manual Histogram
bin_edges = np.arange((min(energy) - 1), (max(energy) + 1) , 1)
counts, edges = np.histogram(energy, bins=bin_edges)
bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

# Graph
plt.figure(figsize=(15, 7))
plt.bar(bin_centers, counts, width=1, color="indigo", align="center")
plt.grid()
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.show()

# Normalized histogram of all particles
bin_edges = np.histogram_bin_edges(energy, bins='fd')

plt.figure(figsize=(15, 7))
plt.hist(energy, bins=bin_edges, color='indigo', density=True)
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.show()

bin_edges = np.arange(-0.5, 7.5 + 1e-5, 1)
counts, edges = np.histogram(asca, bins=bin_edges)
bin_centers = np.arange(0, 8)

plt.figure(figsize=(10, 7))
plt.bar(bin_centers, counts, width=0.7, color="rosybrown", align="center")
plt.xlabel("Event Grades")
plt.ylabel("Count")
plt.xticks(np.arange(0, 8))
plt.grid(axis='y')
plt.show()

standard_grades = [0, 2, 3, 4, 6]
other_grades = [1, 5, 7]

# Get boolean masks
standard_mask = np.isin(asca, standard_grades)
other_mask = np.isin(asca, other_grades)

# Apply masks to get filtered arrays
standard_asca = asca[standard_mask]
standard_energy = energy[standard_mask]
standard_time = unix[standard_mask]

other_asca = asca[other_mask]
other_energy = energy[other_mask]
other_time = unix[other_mask]

# Energy Histogram of Standard ASCA
bin_edges = np.histogram_bin_edges(standard_energy, bins='fd')
plt.figure(figsize=(15, 7))
plt.hist(standard_energy, bins=bin_edges, color='olivedrab', density=True)
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.show()

# Set Bins
bin_edges = np.arange((min(standard_energy) - 1), (max(standard_energy) + 1) , 1)
counts, edges = np.histogram(standard_energy, bins=bin_edges)
bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

# Graph
plt.figure(figsize=(15, 7))
plt.bar(bin_centers, counts, width=1, color="olivedrab", align="center")
plt.grid()
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.xlim(left=0)
plt.show()

# Energy histogram of the Other ACSA
bin_edges = np.histogram_bin_edges(other_energy, bins='fd')
plt.figure(figsize=(15, 7))
plt.hist(other_energy, bins=bin_edges, color='dodgerblue', density=True)
plt.xlabel("Energy (keV)")
plt.ylabel("Normalized Count")
plt.xlim(left=0)
plt.grid()
plt.show()

# Graph Other ACSA
bin_edges = np.arange((min(other_energy) - 1), (max(other_energy) + 1) , 1)
counts, edges = np.histogram(other_energy, bins=bin_edges)
bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

plt.figure(figsize=(15, 7))
plt.bar(bin_centers, counts, width=1, color="dodgerblue", align="center")
plt.grid()
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.show()

# Flux calculation
graph_times = unix

# Calculate the Counts per Second
dtime = 100
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

# ASCA Flux
asca_cps = []
asca_adjusted_times = []

i = 0
while i < len(standard_time):
    group_keys = []
    group_times = [standard_time[i]]
    j = i + 1

    while j < len(standard_time) and (standard_time[j] - standard_time[i]) < dtime:
        group_keys.append(1)
        group_times.append(standard_time[j])
        j += 1

    cps_value = sum(group_keys) / dtime 
    avg_time = sum(group_times) / len(group_times)

    asca_cps.append(cps_value)
    asca_adjusted_times.append(avg_time)

    i = j

asca_cps = np.array(asca_cps)

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