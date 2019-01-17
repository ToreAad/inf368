import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_csv("./data/handwritten_digits_images.csv", header=None).values
y = pd.read_csv("./data/handwritten_digits_labels.csv",header=None).values

X_train, X_compare, y_train, y_compare = train_test_split(X, y, test_size=0.1, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_compare, y_compare, test_size=0.5, random_state=42)

if __name__ == "__main__":
    print("Data successfully loaded.")
    print("X has shape {} and y has shape {}.".format(X.shape, y.shape))