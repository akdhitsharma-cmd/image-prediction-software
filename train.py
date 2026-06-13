# train_model.py

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load dataset
digits = load_digits()
X = digits.data   # images (flattened 8x8 = 64 pixels)
y = digits.target # labels (0–9)

# Train KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Save model
pickle.dump(model, open("knn_digit_model.pkl", "wb"))