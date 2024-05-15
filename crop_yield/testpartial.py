from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

X, y = make_hastie_10_2(random_state=0)
print(X)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,  max_depth=1, random_state=0).fit(X, y)
features = [0, 1, (0, 1)]
PartialDependenceDisplay.from_estimator(clf, X, features)
plt.gcf() 
plt.show()

