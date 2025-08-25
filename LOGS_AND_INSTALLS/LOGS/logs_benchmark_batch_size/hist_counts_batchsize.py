# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Assisted by watsonx Code Assistant

# import numpy as np
import matplotlib.pyplot as plt
import numpy as np


# Given data for 10 prompts
datafile1 = "/Users/diego/TODELETE/better1"
datafile2 = "/Users/diego/TODELETE/better2"
datafile3 = "/Users/diego/TODELETE/better3"
datafile4 = "/Users/diego/TODELETE/better4"


def process_data(file_path):

    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by commas and convert to integers/float where necessary
            values = line.strip().split(',')
            data_list.append((int(values[0]), int(values[1])))
            
    return data_list

# data
data1 = process_data(datafile1)
data2 = process_data(datafile2)
data3 = process_data(datafile3)
data4 = process_data(datafile4)


# Prepare data for histogram
all_batch_sizes = [item[0] for group in [data1, data2, data3, data4] for item in group]
bin_edges = np.arange(min(all_batch_sizes), max(all_batch_sizes) + 2, 1)  # 20 bins from min to max

hist_data = [
    [item[1] for item in group if item[0] in bin_edges] for group in [data1, data2, data3, data4]
]

# Create histogram
plt.hist(hist_data, bins=bin_edges, stacked=True, alpha=0.7, label=['Group 1', 'Group 2', 'Group 3', 'Group 4'])

# Add title and labels
plt.title('Multi-bar Histogram for Batch Sizes')
plt.xlabel('Batch Size')
plt.ylabel('Counts')
plt.legend()

# Show the plot
plt.show()