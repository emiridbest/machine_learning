import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import sklearn.model_selection as ms
import sklearn.tree as tr
from sklearn import tree

# import datasets
iris = dt.load_iris()
x = iris.data
y = iris.target
x_tr, x_tes, y_tr, y_tes = ms.train_test_split(x,y,train_size=0.8, random_state=42)

# create a decision tree classifier with the current max_depth
DT = tr.DecisionTreeClassifier(max_depth=3)
DT.fit(x_tr, y_tr)
trACC = DT.score(x_tr, y_tr)
tesACC = DT.score(x_tes, y_tes)
print(trACC)
print(tesACC)

model = DT.fit(x_tr, y_tr)
test_representation = tr.export_text(DT)
print(test_representation)

# Visualise the Decision Tree
tree.plot_tree(model)


# Decision tree
trACC = []
tesACC = []
MD = []

for i in range(2,8):
  # Create a decision tree classifier with the current depht
  DT = tr.DecisionTreeClassifier(max_depth=i)
  DT.fit(x_tr, y_tr)
  trACC.append(DT.score(x_tr, y_tr))
  tesACC.append(DT.score(x_tes, y_tes))
  MD.append(i)

print(trACC)
print(tesACC)
print(MD)

plt.figure()
plt.plot(MD, trACC, label='Train', marker='o')
plt.plot(MD, tesACC, label='Test', marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


