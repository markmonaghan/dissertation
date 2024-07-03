import pandas as pd
import numpy as np

# Define the path to the data dictionary CSV file
data_dictionary = r'C:\Users\Mark\Documents\rp\data_dictionaries\nih_xrays_data_dictionary.csv'

all_images = pd.read_csv(data_dictionary)


# Define a function for labeling
def determine_label(finding):
    if 'Fibrosis' in finding:
        return 'Fibrosis'
    elif 'No Finding' in finding:
        return 'No Finding'
    else:
        return 'Non-Fibrosis'

# Define a function for class assignment
def determine_class(finding):
    if 'Fibrosis' in finding:
        return 'Fibrosis'
    else:
        return 'Non-Fibrosis'

all_images['Label'] = None
all_images['Class'] = None

finding_labels_index = all_images.columns.get_loc('Finding Labels')
columns = list(all_images.columns)
columns.insert(finding_labels_index + 1, columns.pop(columns.index('Label')))
columns.insert(finding_labels_index + 2, columns.pop(columns.index('Class')))

all_images = all_images[columns]

# Adding 'Label' and 'Class' columns based on conditions using the functions
all_images['Label'] = all_images['Finding Labels'].apply(determine_label)
all_images['Class'] = all_images['Finding Labels'].apply(determine_class)

all_images.to_csv(r'C:\Users\Mark\Documents\rp\data_dictionaries\NEW_data_dictionary.csv', index=False)
all_images.to_excel(r'C:\Users\Mark\Documents\rp\data_dictionaries\NEW_data_dictionary.xlsx', index=False)

# Display the DataFrame to see the results
print(all_images)
