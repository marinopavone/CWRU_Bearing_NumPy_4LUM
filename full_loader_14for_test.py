import os
import re
import pandas as pd
from custom_function.plot_functions import *
# Define the path to your Data folder
data_dir = "ORIGINAL_Data"

# Define filtering criteria
desired_rpm = [1730, 1750, 1772, 1797]
desired_faults = ["B", "OR@3","OR@6","OR@12", "IR", "Normal"]
desired_experiments = ["DE", "FE","DE12", "FE12","DE48", "FE48"]
desired_severity = [7, 14, 21, 28]
desired_sample_freq = [12, 48, None]

# Prepare list to store parsed file info
files_data = []


pattern = re.compile(
    r'^(?P<RPM>\d+)_'                          # 1730_
    r'(?P<Fault>B|OR@3|OR@6|OR@12|IR|Normal)'               # B, OR, IR or Normal
    r'(?:@(?P<Freq>\d+))?'                     # optional @12 (for OR@12 style)
    r'(?:_(?P<Severity>\d+)_(?P<Experiment>[A-Za-z0-9]+))?'  # optional _7_DE12 etc.
    r'\.npz$'                                  # .npz at end
)

# Walk through data directory
for rpm_folder in os.listdir(data_dir):
    rpm_path = os.path.join(data_dir, rpm_folder)
    if os.path.isdir(rpm_path):
        print(rpm_path)
        if "1797" in rpm_folder:
            print("calma")
            print("calmaq")
        for file_name in os.listdir(rpm_path):
            if file_name.endswith(".npz"):
                match = pattern.match(file_name)
                if match:

                    file_info = match.groupdict()

                    pathpath = os.path.join(rpm_path, file_name)
                    peppe = np.load(os.path.join(rpm_path, file_name))
                    # Convert numeric fields to int if they exist
                    file_info['RPM'] = int(file_info['RPM'])
                    file_info['Severity'] = int(file_info['Severity']) if file_info['Severity'] else None

                    file_info['Path'] = pathpath
                    file_info['Raw'] = peppe["DE"]
                    file_info['Name'] = file_name
                    files_data.append(file_info)
                    print(files_data.__len__())



# Create  pandas DataFrame
df = pd.DataFrame(files_data)

# Reorder columns for clarity
df = df[['RPM', 'Fault', 'Severity', 'Experiment' , 'Raw', 'Name']]
print(df)
# Show result
# selected = df[
#     (df["RPM"].isin([1750,1730])) &
#     (df["Fault"].isin(["IR"])) &
#     (df["Severity"].isin([7,14,21]))
# ]

classification_target = ["B",  "IR", "Normal"]
# plot_signals(selected, title="plot bellissimo" )
training_df = df[  (df["Experiment"].isin(["DE12"])) &
                   (df["Severity"].isin([7,21,0]))&
                   # (df["Fault"].isin(["B", "OR@6", "IR", "Normal"]))]
                   (df["Fault"].isin(classification_target))]

test_df = df[      (df["Experiment"].isin(["DE12"])) &
                   (df["Severity"].isin([14,0]))&
                   # (df["Fault"].isin(["B", "OR@6", "IR", "Normal"]))]
                   (df["Fault"].isin(classification_target))]

import pickle

# Save both DataFrames in one pickle file
with open("ORIGINAL_PICKEL/datasets.pkl", "wb") as f:
    pickle.dump({
        "training_df": training_df,
        "test_df": test_df,
        "classification_target": classification_target

    }, f)