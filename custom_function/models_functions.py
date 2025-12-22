import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import json
import random
import os


def load_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)

def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
def Generate_NN_model(X_train, config_file):
    set_global_seed(42)

    cfg = load_config(config_file)
    input_dim = X_train.shape[1]
    layers_sizes = np.insert(cfg["layers"], 0, input_dim)
    activations = cfg["activations"]

    if len(layers_sizes) != len(activations):
        raise ValueError("ERROR: layers[] and activations[] must have the SAME length.")

    input_dim = X_train.shape[1]
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    # Build all layers with corresponding activations
    for units, act in zip(layers_sizes, activations):

        model.add(layers.Dense(units, activation=act))

        # Optional Batch Normalization
        if cfg.get("batch_norm", False) and act.lower() != "softmax":
            model.add(layers.BatchNormalization())

        # Optional Dropout
        if cfg["dropout"] > 0 and act.lower() != "softmax":
            model.add(layers.Dropout(cfg["dropout"]))

    # Build optimizer
    opt_name = cfg["optimizer"].lower()

    if opt_name == "sgd":
        opt = optimizers.SGD(learning_rate=cfg["learning_rate"],
                             momentum=cfg["momentum"])
    elif opt_name == "adam":
        opt = optimizers.Adam(learning_rate=cfg["learning_rate"])
    elif opt_name == "rmsprop":
        opt = optimizers.RMSprop(learning_rate=cfg["learning_rate"])
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # Compile
    model.compile(optimizer=opt,
                  loss=cfg["loss"],
                  metrics=cfg["metrics"])

    print(model.summary())
    return model

def train_model(model, X_train, y_train, config_file):
    cfg = load_config(config_file)

    history = model.fit(
        X_train,
        y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_split=0.1,
        shuffle=True
    )

    return history


import numpy as np
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt

def test_model(model, X_test, y_test, class_names=None):

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    class_acc = []
    class_prec = []
    num_classes = y_test.shape[1]

    for c in range(num_classes):
        idx = np.where(y_true == c)[0]
        acc = accuracy_score(y_true[idx], y_pred[idx])
        prec = precision_score(y_true == c, y_pred == c)
        class_acc.append(acc)
        class_prec.append(prec)

    # Plot accuracy
    plt.figure(figsize=(10,5))
    plt.bar(range(num_classes), class_acc)
    plt.title("Class-wise Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.show()

    # Plot precision
    plt.figure(figsize=(10,5))
    plt.bar(range(num_classes), class_prec)
    plt.title("Class-wise Precision")
    plt.xlabel("Class")
    plt.ylabel("Precision")
    plt.show()

    avg_acc = np.mean(class_acc)
    avg_prec = np.mean(class_prec)

    print("Average Accuracy:", avg_acc)
    print("Average Precision:", avg_prec)

    return class_acc, class_prec, avg_acc, avg_prec
