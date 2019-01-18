# https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


assert os.path.isfile("./data/handwritten_digits_images.csv"), "Data missing"
assert os.path.isfile("./data/handwritten_digits_labels.csv"), "Data missing"

X = pd.read_csv("./data/handwritten_digits_images.csv", header=None).values.astype(np.float32)
y = pd.read_csv("./data/handwritten_digits_labels.csv",header=None).values

scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)
# X_scaled = X

X_train, X_compare, y_train, y_compare = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_compare, y_compare, test_size=0.5, random_state=42)

if __name__ == "__main__":
    print("Data successfully loaded.")
    print("X has shape {} and y has shape {}.".format(X.shape, y.shape))