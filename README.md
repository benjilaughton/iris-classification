# Iris Species Classification
A compact supervised-learning study comparing k-Nearest Neighbors and Gaussian Naive Bayes on the classic Iris dataset.

# Overview
An introductory-but-rigorous classification project on the 150-sample Iris dataset, used to compare two fundamentally different classifiers (distance-based vs. probabilistic) and to visualize class separability.

# Approach
EDA: pairplot across the four features to inspect class separability by species.
Models: KNeighborsClassifier (k=3) and GaussianNB.
Evaluation: 20% test split; per-model confusion matrices and weighted F1.

# Results
Both models score F1 ≈ 1.00 on the test split — setosa is linearly separable, with minor overlap between versicolor and virginica.

# Tech stack
Python · scikit-learn · pandas · seaborn · matplotlib

# Run it
pip install scikit-learn pandas seaborn matplotlib

python iris.py

# Repo contents
iris.py — EDA, two-model comparison, evaluation
charts/ — pairplot and confusion matrices

