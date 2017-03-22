from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


def print_result(model, result, extra=""):
    print(f"Predicting with model {type(model).__name__} ({extra}): {result}")


def predict(k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    res = model.predict(X_test)
    score = metrics.accuracy_score(y_test, res)
    print_result(model, score, extra=f"k = {k}")
    return score


def predict_log():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    res = model.predict(X_test)
    score = metrics.accuracy_score(y_test, res)
    print_result(model, score)


predict_log()
scores = []
k_range = range(1, 26)
for k in k_range:
    score = predict(k)
    scores.append(score)

plt.plot(k_range, scores)
plt.xlabel("Value for K for KNN")
plt.ylabel("Testing Accuracy")

plt.show()

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X, y)
res = knn.predict([3, 5, 4, 2])
print(res)
