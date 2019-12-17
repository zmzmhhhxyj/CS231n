from __future__ import print_function
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import pickle

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

cifar10_dir = r'D:\python\CS231n\assignment1\cs231n\datasets\cifar-10-batches-py'
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
"""
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
"""
# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
"""
dists = classifier.compute_distances_two_loops(X_test)
pickle.dump(dists,open(r"D:\python\CS231n\assignment1\tmp.txt","wb"))
print(dists.shape)
print(dists)

plt.imshow(dists, interpolation='none')
plt.show()
"""
with open(r"D:\python\CS231n\assignment1\tmp.txt","rb") as file:
    dists = pickle.load(file)
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
""" num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
 """
"""
dists_one = classifier.compute_distances_one_loop(X_test)
pickle.dump(dists,open(r"D:\python\CS231n\assignment1\tmp_one.txt","wb"))
"""
"""
with open(r"D:\python\CS231n\assignment1\tmp_one.txt","rb") as file:
    dists_one = pickle.load(file)
print(np.shape(dists_one))
y_test_pred = classifier.predict_labels(dists, k=1)
difference = np.linalg.norm(dists-dists_one,ord="fro")
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')
"""
"""
dists_two = classifier.compute_distances_no_loops(X_test)
pickle.dump(dists,open(r"D:\python\CS231n\assignment1\tmp_two.txt","wb"))
with open(r"D:\python\CS231n\assignment1\tmp_two.txt","rb") as file:
    dists_two = pickle.load(file)
# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')
"""

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_split = np.array_split(X_train,num_folds) 
y_train_split = np.array_split(y_train,num_folds) 
for i in range(num_folds):
    X_train_folds.append(X_train_split[i])#(5,1000,3072)
    y_train_folds.append(y_train_split[i])#(5,1000)

################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
for k in k_choices:
    acc = []
    print(k)
    for i in range(num_folds):
        #(0，4000,3072),每个都是（0，1000，3072），竖着叠加，shape（4000，3072）
        x_train_fold = np.vstack(X_train_folds[0:i]+X_train_folds[i+1:])
        #(，4000)，每个都是（，1000），横向叠加，shape（4000，）
        y_train_fold = np.hstack((y_train_folds[0:i]+y_train_folds[i+1:]))
        x_val = X_train_folds[i]
        y_val = y_train_folds[i]

        classifier = KNearestNeighbor()
        classifier.train(x_train_fold,y_train_fold)
        dists_two = classifier.compute_distances_no_loops(x_val)
        y_val_pred = classifier.predict(x_val,k)
        correct = np.sum(y_val_pred==y_val)/y_val.shape[0]
        acc.append(correct)
    k_to_accuracies[k] = acc
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))