# Anna Spiro
# ML Course Project: COVID Search Data (NY Counties)

"""
4/12 to do: 
-understand k-fold cross-validation 
-enough data? 
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from load_data import JoinedData

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor

#reload joined_datapoints from saved_data file 
saved_data = open(r'C:\d.pkl', 'rb')
joined_datapoints = pickle.load(saved_data)
saved_data.close()

# setup for ML
# code below adapted from p05-join.py

ys = []
examples = []
for datapoint in joined_datapoints:
    ys.append(datapoint.cases)
    examples.append(datapoint.symptoms)

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

# try out of box models 

# try different normalizations 
norm = "max"
if norm == "var":
    scale = StandardScaler()
    ex_train = scale.fit_transform(ex_train)
    ex_vali = scale.transform(ex_vali)
    ex_test = scale.transform(ex_test)
elif norm == "max":
    scale = MinMaxScaler()
    ex_train = scale.fit_transform(ex_train)
    ex_vali = scale.transform(ex_vali)
    ex_test = scale.transform(ex_test)
else:
    ex_train = ex_train
    ex_vali = ex_vali
    ex_test = ex_test

print("KNeighborsRegressor")
knr = KNeighborsRegressor(n_neighbors=5, weights="distance")
knr.fit(ex_train, y_train)
print(knr.score(ex_vali, y_vali))

print("DecisionTreeRegressor")
dtr = DecisionTreeRegressor()
dtr.fit(ex_train, y_train)
print(dtr.score(ex_vali, y_vali))

print("SGDRegressor")
sgdr = SGDRegressor()
sgdr.fit(ex_train, y_train)
print(sgdr.score(ex_vali, y_vali))

"""
print("MLPRegressor")
mlpr = MLPRegressor(max_iter=100000)
mlpr.fit(ex_train, y_train)
print(mlpr.score(ex_vali, y_vali))
"""






