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

window=1200 # un decimo di secondo a 12kHz
window=4800 # 4 decimi di secondo a 12kHz
window=2400 # 4 decimi di secondo a 12kHz
fs=12000
train_df.insert(1, "Freq_data", None)
test_df.insert(1, "Freq_data", None)
# il vero preprocessing sta qua
train_df["Freq_data"] = [split_and_fft(elem["Raw"], window,fs) for index, elem in train_df.iterrows() ]
test_df["Freq_data"] = [split_and_fft(elem["Raw"], window,fs) for index, elem in test_df.iterrows() ]

from custom_function.preprocessing_functions import buil_x_y

x_train_rpm, x_train_spectrums, y_train_lab = buil_x_y(train_df)
x_test_rpm, x_test_spectrums, y_test_lab = buil_x_y(test_df)

from sklearn.decomposition import PCA
n_components = 50      # or e.g. 0.98 for 98% variance kept
pca = PCA(n_components=n_components)
pca.fit(x_train_spectrums)

x_train_compressed = pca.transform(x_train_spectrums)
x_test_compressed  = pca.transform(x_test_spectrums)
y_train_lab = np.array(y_train_lab)
y_test_lab = np.array(y_test_lab)

x_train = np.column_stack([x_train_rpm, x_train_compressed])
x_test = np.column_stack([x_test_rpm, x_test_compressed])
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





