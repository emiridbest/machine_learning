import numpy as np
import matplotlib.pyplot  as plt
import sklearn.datasets as dt
import sklearn.model_selection as ms
import sklearn.neighbors as ne

# Dataset
iris = dt.load_iris()
x = iris.data
y = iris.target
KNN = ne.KNeighborsClassifier(n_neighbors = 5)
KNN.fit(x,y)
Acc = KNN.score(x,y)
print(Acc)

Acc1 = []
MD = []

for i in range(1,21):
  KNN = ne.KNeighborsClassifier(n_neighbors = i)
  KNN.fit(x,y)
  Acc1.append(KNN.score(x,y))
  MD.append(i)
print(Acc1)
print(MD)


# Visualise ACC
plt.figure()
plt.plot(MD, Acc1, label='KNN', marker='o')
plt.xlabel('K_neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


x_tr, x_tes, y_tr, y_tes = ms.train_test_split(x, y, train_size=0.8, random_state=42)
Acc_tr = []
Acc_tes = []
MD = []

for i in range(1,21):
  KNN = ne.KNeighborsClassifier(n_neighbors = i)
  KNN.fit(x_tr, y_tr)
  Acc_tr.append(KNN.score(x_tr, y_tr))
  Acc_tes.append(KNN.score(x_tes, y_tes))
  MD.append(i)

print(Acc_tr)
print(Acc_tes)
print(MD)

# Visualise
plt.figure()
plt.plot(MD, Acc_tr, label='Train', marker='o')
plt.plot(MD, Acc_tes, label='o', marker='o')
plt.xlabel('K_neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
