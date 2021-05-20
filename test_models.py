# Anna Spiro
# ML Course Project: COVID Search Data (NY Counties)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from load_data import JoinedData
import numpy as np
from sklearn.utils import resample
from shared import simple_boxplot, bootstrap_accuracy, bootstrap_r2

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor

feature_names = ['fever', 
    'chills', 
    'cough',
    'shortness_of_breath',
    'shallow_breathing',
    'fatigue',
    'headache',
    'sore_throat',
    'nasal_congestion',
    'nausea',
    'vomiting',
    'diarrhea',
    'dysguesia',  # partial loss of taste
    'ageusia',  # total loss of taste
    'anosmia',  # loss of smell
    'myalgia'] 

#reload joined_datapoints from saved_data file 

data_file = open("saved_data", "rb")
joined_datapoints = pickle.load(data_file)

# setup for ML
# code below adapted from p08

print(len(joined_datapoints))

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
    train_size=0.75,
    shuffle=True,
    random_state=RANDOM_SEED,
)

# split off train, validate from (tv) pieces.
ex_train, ex_vali, y_train, y_vali = train_test_split(
    ex_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
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

print("RandomForestRegressor")
rfr = RandomForestRegressor(max_depth=4, random_state=RANDOM_SEED)
rfr.fit(X_train, y_train)
print(rfr.score(X_vali, y_vali))

print("MLPRegressor")
mlpr = MLPRegressor(max_iter=10000)
mlpr.fit(X_train, y_train)
print(mlpr.score(X_vali, y_vali))

"""
print("DecisionTreeRegressor")
dtr = DecisionTreeRegressor(max_depth=4)
dtr.fit(X_train, y_train)
print(dtr.score(X_vali, y_vali))

print("SGDRegressor")
sgdr = SGDRegressor()
sgdr.fit(X_train, y_train)
print(sgdr.score(X_vali, y_vali))

# look at feature importance (code adapted from p10)

# loop over each tree and ask them how important each feature was!
importances = dict((name, []) for name in feature_names)
for tree in rfr.estimators_:
    for name, weight in zip(feature_names, tree.feature_importances_):
        importances[name].append(weight)

# Think: what does 'how many splits' actually measure? Usefulness, or something else?
simple_boxplot(
    importances,
    title="Tree Importances",
    ylabel="Decision Tree Criterion Importances",
)

# try removing features (code adapted from p10)

from dataclasses import dataclass
import matplotlib.pyplot as plt
import typing as T

@dataclass
class Model:
    vali_score: float
    m: T.Any

graphs: T.Dict[str, T.List[float]] = {}

def train_and_eval(name, x, y, vx, vy):
    # Train and Eval a single model
    options: T.List[Model] = []

    # start with only one option 
    #m = KNeighborsRegressor(n_neighbors=5, weights="distance")
    #m.fit(x, y)
    #options.append(Model(m.score(vx, vy), m))

    #m = RandomForestRegressor(max_depth=4, random_state=RANDOM_SEED)
    #m.fit(x, y)
    #options.append(Model(m.score(vx, vy), m))

    m = MLPRegressor(max_iter=10000)
    m.fit(x, y)
    options.append(Model(m.score(vx, vy), m))

    # pick the best model:
    best = max(options, key=lambda m: m.vali_score)
    # bootstrap its output:
    graphs[name] = bootstrap_r2(best.m, vx, vy)
    # record our progress:
    print("{:20}\t{:.3}\t{}".format(name, np.mean(graphs[name]), best.m))

train_and_eval("Full Model", X_train, y_train, X_vali, y_vali)

for name in feature_names:
    # one-by-one, delete your features:
    without_X = X_train.copy()
    feature_index = feature_names.index(name)
    without_X[:, feature_index] = 0.0 # not sure if this is right
   
  # score a model without the feature to see if it __really__ helps or not:
    train_and_eval("without {}".format(name), without_X, y_train, X_vali, y_vali)

# Inline boxplot code here so we can sort by value:
box_names = []
box_dists = []
for (k, v) in sorted(graphs.items(), key=lambda tup: np.mean(tup[1])):
    box_names.append(k)
    box_dists.append(v)

# Matplotlib stuff:
plt.boxplot(box_dists)
plt.xticks(
    rotation=30,
    horizontalalignment="right",
    ticks=range(1, len(box_names) + 1),
    labels=box_names,
)
plt.title("Feature Removal Analysis")
plt.xlabel("Included?")
plt.ylabel("AUC")
plt.tight_layout()
plt.show()


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
        mlpr = MLPRegressor(max_iter=10000)
        mlpr.fit(X_sample, y_sample)

        # so we get 100 scores per percentage-point.
        scores[label].append(mlpr.score(X_vali, y_vali))
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
plt.title("MLPR Shaded Accuracy Plot")
plt.savefig("MLPR-area-accuracy.png")
plt.show()

# Second look at the boxplots in-order: (I like this better, IMO)
simple_boxplot(
    scores,
    "SGDR Learning Curve",
    xlabel="Percent Training Data",
    ylabel="Accuracy",
    save="SGDR-boxplots-accuracy.png",
)
"""



