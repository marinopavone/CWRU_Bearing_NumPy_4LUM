from custom_function.plot_functions import *
from custom_function.preprocessing_functions import *
from custom_function.models_functions import *
import pickle

load_path = "ORIGINAL_PICKEL/datasets_processed_PCA.pkl"

with open(load_path, "rb") as f:
    data = pickle.load(f)

# Extract variables
x_train = data["x_train"]
x_train_spectrum = data["x_train_spectrum"]
x_train_compressed = data["x_train_compressed"]
y_train = data["y_train"]

x_test = data["x_test"]
x_test_spectrum = data["x_test_spectrum"]
x_test_compressed = data["x_test_compressed"]
y_test = data["y_test"]

print("Reload completed.")

config_file = "config_file/nn_config_PCA.json"

nn_model = Generate_NN_model(x_train_compressed, config_file)

history = train_model(nn_model, x_train_compressed, y_train, config_file)

test_results = test_model(nn_model, x_test_compressed, y_test)