from sklearn import svm

# types of classifiers:
# SVC, NuSVC (similar methods) - for multiclass (one vs one)
#
# and LinearSVC (faster but only for linear kernel) - for multiclass (one vs rest)

X = [[0, 0], [1, 1]] # X=(n_samples,n_features)
y = [0, 1] # label of each sample dim(y)=dim(X[0])
clf = svm.SVC()
clf.fit(X, y)

#the model can be use to predict new values
print(clf.predict([[2., 2.]]))

# properties of the support vector machine
# get support vectors
print(clf.support_vectors_)

# get indices of support vectors
print(clf.support_)

# get number of support vectors for each class
print(clf.n_support_)

# multiclass clasification
# SVC and NuSVC "one vs one" approach