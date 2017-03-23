from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy")
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel("Value for K for KNN")
plt.ylabel("Cross-Validated Accuracy")
plt.show()

# KNN
knn = KNeighborsClassifier(n_neighbors=20)
print(cross_val_score(knn, X, y, cv=10, scoring="accuracy").mean())

# LogReg
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring="accuracy").mean())



