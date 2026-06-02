from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn import naive_bayes
import seaborn as sns

#benjamin laughton CIS2541 project 1 - 6/22/2025


#1:
# load dataset
iris = datasets.load_iris()



# create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# add the target column (species)
df['species'] = iris.target

# display first 5 and last 5 rows
print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())


#2:
# create pairplot using seaborn
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue='species', markers=['o', 's', 'D'])
plt.show()

# separate features (X) and target (y)
X = iris.data  # Features: the 4 measurements
y = iris.target  # Target: the species (0, 1, or 2)

# split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print the shapes to verify
print(f"Train features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Train target shape: {y_train.shape}")
print(f"Test target shape: {y_test.shape}")

# create kNN model with 3 neighbors
knn_model = neighbors.KNeighborsClassifier(n_neighbors=3)

# train the model
knn_model.fit(X_train, y_train)

# make predictions on test data
knn_predictions = knn_model.predict(X_test)

# calculate F1 score + classification report (for extended info)
knn_f1 = metrics.f1_score(y_test, knn_predictions, average='weighted')

print(f"kNN F1 Score: {knn_f1:.2f}")

print("\nkNN Classification Report:")
print(metrics.classification_report(y_test, knn_predictions, target_names=iris.target_names))

# create Naive Bayes model
nb_model = naive_bayes.GaussianNB()

# train the model
nb_model.fit(X_train, y_train)

# make predictions on test data
nb_predictions = nb_model.predict(X_test)

# calculate F1 score + classification report (for extended info)
nb_f1 = metrics.f1_score(y_test, nb_predictions, average='weighted')

print(f"Naive Bayes F1 Score: {nb_f1:.2f}")

print("\nNaive Bayes Classification Report:")
print(metrics.classification_report(y_test, nb_predictions, target_names=iris.target_names))

# compare the results
print("="*50)
print("MODEL COMPARISON")
print("="*50)
print(f"kNN F1 Score: {knn_f1:.2f}")
print(f"Naive Bayes F1 Score: {nb_f1:.2f}")
print("="*50)

if knn_f1 > nb_f1:
    print("kNN performed better")
elif nb_f1 > knn_f1:
    print("Naive Bayes performed better")
else:
    print("Both models performed equally well")

[]