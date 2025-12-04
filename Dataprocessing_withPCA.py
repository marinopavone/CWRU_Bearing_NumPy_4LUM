import pickle
from custom_function.plot_functions import *
from custom_function.preprocessing_functions import *
from custom_function.models_functions import *

with open("ORIGINAL_PICKEL/datasets.pkl", "rb") as f:
    data = pickle.load(f)

train_df = data["training_df"]
test_df = data["test_df"]
fault_classes = data["classification_target"]

# fault_classes = ["B", "OR@6", "IR", "Normal"]

one_hot_map = {
    fault: np.eye(len(fault_classes))[i]
    for i, fault in enumerate(fault_classes)
}
train_df.insert(1, "Classification_label", None)
test_df.insert(1, "Classification_label", None)


train_df["Classification_label"] = [one_hot_map[elem["Fault"]] for index, elem in train_df.iterrows() ]
test_df["Classification_label"] = [one_hot_map[elem["Fault"]] for index, elem in test_df.iterrows() ]

w=1200 # un decimo di secondo a 12kHz
w=4800 # 4 decimo di secondo a 12kHz
fs=12000
train_df.insert(1, "Freq_data", None)
test_df.insert(1, "Freq_data", None)
train_df["Freq_data"] = [split_and_fft(elem["Raw"], w,fs) for index, elem in train_df.iterrows() ]
test_df["Freq_data"] = [split_and_fft(elem["Raw"], w,fs) for index, elem in test_df.iterrows() ]

x_train_spectrums=[]
y_train_lab=[]
for index, elem in train_df.iterrows():
    for segment in elem["Freq_data"]:
        x_train_spectrums.append(segment)
        y_train_lab.append(elem["Classification_label"])
x_train = np.array(x_train_spectrums)

x_test_spectrums=[]
y_test_lab=[]
for index, elem in test_df.iterrows():
    for segment in elem["Freq_data"]:
        x_test_spectrums.append(segment)
        y_test_lab.append(elem["Classification_label"])
x_test = np.array(x_test_spectrums)

# x_train_compressed = average_over_window(x_train, aw=3)
# x_test_compressed = average_over_window(x_test, aw=3)
# y_train_lab = np.array(y_train_lab)
# y_test_lab = np.array(y_test_lab)

from sklearn.decomposition import PCA
n_components = 50      # or e.g. 0.98 for 98% variance kept
pca = PCA(n_components=n_components)
pca.fit(x_train)

x_train_compressed = pca.transform(x_train)
x_test_compressed  = pca.transform(x_test)
y_train_lab = np.array(y_train_lab)
y_test_lab = np.array(y_test_lab)


import pickle

save_path = "ORIGINAL_PICKEL/datasets_processed_PCA.pkl"

data_to_save = {
    "x_train": x_train,
    "x_train_spectrum": x_train_spectrums,
    "x_train_compressed": x_train_compressed,
    "y_train": y_train_lab,

    "x_test": x_test,
    "x_test_spectrum": x_test_spectrums,
    "x_test_compressed": x_test_compressed,
    "y_test": y_test_lab,
}

with open(save_path, "wb") as f:
    pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved successfully →", save_path)
