from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X = iris.data

y = iris.target

print(X.shape)
print(y.shape)


def predict(k):

    knn = KNeighborsClassifier(n_neighbors=k)

    print(knn)

    knn.fit(X, y)
    newx = [[3, 5, 4, 2], [5, 4, 3, 2]]
    res = knn.predict(newx)
    print("Predicting with k = " + str(k) + " " + str(res))


predict(1)
predict(5)
