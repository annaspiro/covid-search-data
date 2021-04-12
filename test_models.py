# Anna Spiro
# ML Course Project: COVID Search Data (NY Counties)

"""
4/12 to do: 
-k-fold cross-validation?
-enough data? 
-any other models to try? 

-I think: no need to do vectorizing (as in p08)

observations: 
KNeighborsRegressor always the same (0.381)
DecisionTreeregressor bad & variable 
SGDREgressor around 0.163 but slightly variable 
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from load_data import JoinedData
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.utils import resample
from shared import simple_boxplot

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor

#reload joined_datapoints from saved_data file 
saved_data = open(r'C:\d.pkl', 'rb')
joined_datapoints = pickle.load(saved_data)
saved_data.close()

# setup for ML
# code below adapted from p08

ys = []
examples = []
for datapoint in joined_datapoints:
    ys.append(datapoint.cases)
    examples.append(np.array(datapoint.symptoms))

RANDOM_SEED = 1234

## split off train/validate (tv) pieces.
ex_tv, ex_test, y_tv, y_test = train_test_split(
    examples,
    ys,
    train_size=0.9,
    shuffle=True,
    random_state=RANDOM_SEED,
)

# split off train, validate from (tv) pieces.
ex_train, ex_vali, y_train, y_vali = train_test_split(
    ex_tv, y_tv, train_size=0.9, shuffle=True, random_state=RANDOM_SEED
)

# try different normalizations 
norm = "var"
if norm == "var":
    scale = StandardScaler()
    X_train = scale.fit_transform(ex_train)
    X_vali = scale.transform(ex_vali)
    X_test = scale.transform(ex_test)
elif norm == "max":
    scale = MinMaxScaler()
    X_train = scale.fit_transform(ex_train)
    X_vali = scale.transform(ex_vali)
    X_test = scale.transform(ex_test)
else:
    X_train = ex_train
    X_vali = ex_vali
    X_test = ex_test

# train models 

print("KNeighborsRegressor")
knr = KNeighborsRegressor(n_neighbors=5, weights="distance")
knr.fit(X_train, y_train)
print(knr.score(X_vali, y_vali))

print("DecisionTreeRegressor")
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
print(dtr.score(X_vali, y_vali))

print("SGDRegressor")
sgdr = SGDRegressor()
sgdr.fit(X_train, y_train)
print(sgdr.score(X_vali, y_vali))

print("MLPRegressor")
mlpr = MLPRegressor(max_iter=10000)
mlpr.fit(ex_train, y_train)
print(mlpr.score(ex_vali, y_vali))

"""
# visualize training (code adapted from p09)

#%% Actually compute performance for each % of training data
N = len(y_train)
num_trials = 100
percentages = list(range(5, 100, 5))
percentages.append(100)
scores = {}
acc_mean = []
acc_std = []

# Which subset of data will potentially really matter.
for train_percent in percentages:
    n_samples = int((train_percent / 100) * N)
    print("{}% == {} samples...".format(train_percent, n_samples))
    label = "{}".format(train_percent, n_samples)

    # So we consider num_trials=100 subsamples, and train a model on each.
    scores[label] = []
    for i in range(num_trials):
        X_sample, y_sample = resample(
            X_train, y_train, n_samples=n_samples, replace=False
        )  # type:ignore
        # Note here, I'm using a simple classifier for speed, rather than the best.
        mlp = MLPRegressor(max_iter=10000, random_state=RANDOM_SEED + train_percent + i)
        mlp.fit(X_sample, y_sample)

        # so we get 100 scores per percentage-point.
        scores[label].append(mlp.score(X_vali, y_vali))
    # We'll first look at a line-plot of the mean:
    acc_mean.append(np.mean(scores[label]))
    acc_std.append(np.std(scores[label]))

# First, try a line plot, with shaded variance regions:
import matplotlib.pyplot as plt

means = np.array(acc_mean)
std = np.array(acc_std)
plt.plot(percentages, acc_mean, "o-")
plt.fill_between(percentages, means - std, means + std, alpha=0.2)
plt.xlabel("Percent Training Data")
plt.ylabel("Mean Accuracy")
plt.xlim([0, 100])
plt.title("MLP Shaded Accuracy Plot")
plt.savefig("MLP-area-accuracy.png")
plt.show()

# Second look at the boxplots in-order: (I like this better, IMO)
simple_boxplot(
    scores,
    "MLP Learning Curve",
    xlabel="Percent Training Data",
    ylabel="Accuracy",
    save="MLP-boxplots-accuracy.png",
)
"""



