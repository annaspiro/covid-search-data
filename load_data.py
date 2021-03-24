# code adapted from p04-models.py

from os import strerror
import numpy as np
import math as math
import csv 
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier

# new helpers:
from shared import dataset_local_path, bootstrap_accuracy, simple_boxplot, TODO

# stdlib:
from dataclasses import dataclass
import json
from typing import Dict, Any, List

# import google symptoms data 
# code to read csv adapted from https://realpython.com/python-csv/

# setup dataclass 
@dataclass
class SymptomsData:
    date: str
    county: str 
    fever: float 
    chills: float
    cough: float 
    shortness_of_breath: float
    shallow_breathing: float 
    fatigue: float
    headache: float
    sore_throat: float
    nasal_congestion: float
    nausea: float
    vomiting: float 
    diarrhea: float 
    dysguesia: float # partial loss of taste 
    ageusia: float # total loss of taste 
    anosmia: float # loss of smell 
    myalgia: float # muscle pain

# first, get csv data into list of SymptomsData

# keep relevant dates as a set so faster to search later 
relevant_dates = set() 

# 2020 data 
last_year_datapoints: List[SymptomsData] = []

with open("2020_NY_data.csv") as csv_file: 
    csv_reader = csv.DictReader(csv_file, delimiter=',')

    # want to start on row 53 (before is combined VT data)
    relevant_rows = [row for idx, row in enumerate(csv_reader) if idx in range(53,3277)]

    for row in relevant_rows: 
        # get rid of "" values 
        if row["symptom:Shallow breathing"] ==  "" or row["symptom:Chills"] == "" or row["symptom:Cough"] == "" or row["symptom:Shortness of breath"] == "" or row["symptom:Shallow breathing"] == "" or row["symptom:Fatigue"] == "" or row["symptom:Headache"] == "" or row["symptom:Sore throat"] == "" or row["symptom:Nasal congestion"] == "" or row["symptom:Nausea"] == "" or row["symptom:Vomiting"] == "" or row["symptom:Diarrhea"] == "" or row["symptom:Dysgeusia"] == "" or row["symptom:Ageusia"] == "" or row["symptom:Anosmia"] == "" or row["symptom:Myalgia"] == "":
             continue

        # first COVID case confirmed on 2020-03-01
        if int(row["date"][5:7]) < 3: 
            continue 

        relevant_dates.add(row["date"])

        last_year_datapoints.append(SymptomsData(
        date = row["date"],
        county = row["sub_region_2"].rsplit(' ',1)[0], # don't want "County"
        fever = float(row["symptom:Fever"]) ,
        chills = float(row["symptom:Chills"]),
        cough = float(row["symptom:Cough"]) ,
        shortness_of_breath = float(row["symptom:Shortness of breath"]),
        shallow_breathing = float(row["symptom:Shallow breathing"]),
        fatigue = float(row["symptom:Fatigue"]),
        headache = float(row["symptom:Headache"]),
        sore_throat = float(row["symptom:Sore throat"]),
        nasal_congestion = float(row["symptom:Nasal congestion"]),
        nausea = float(row["symptom:Nausea"]),
        vomiting = float(row["symptom:Vomiting"]),
        diarrhea = float(row["symptom:Diarrhea"]),
        dysguesia = float(row["symptom:Dysgeusia"]), 
        ageusia = float(row["symptom:Ageusia"]), 
        anosmia = float(row["symptom:Anosmia"]), 
        myalgia = float(row["symptom:Myalgia"])))

# 2021 data 
this_year_datapoints: List[SymptomsData] = []

with open("2021_NY_data.csv") as csv_file: 
    csv_reader = csv.DictReader(csv_file, delimiter=',')

    # want to start on row 53 (before is combined VT data)
    relevant_rows = [row for idx, row in enumerate(csv_reader) if idx in range(11,631)]

    for row in relevant_rows: 
        # get rid of "" values 
        if row["symptom:Shallow breathing"] ==  "" or row["symptom:Chills"] == "" or row["symptom:Cough"] == "" or row["symptom:Shortness of breath"] == "" or row["symptom:Shallow breathing"] == "" or row["symptom:Fatigue"] == "" or row["symptom:Headache"] == "" or row["symptom:Sore throat"] == "" or row["symptom:Nasal congestion"] == "" or row["symptom:Nausea"] == "" or row["symptom:Vomiting"] == "" or row["symptom:Diarrhea"] == "" or row["symptom:Dysgeusia"] == "" or row["symptom:Ageusia"] == "" or row["symptom:Anosmia"] == "" or row["symptom:Myalgia"] == "":
             continue
        
        relevant_dates.add(row["date"])

        this_year_datapoints.append(SymptomsData(
        date = row["date"],
        county = row["sub_region_2"].rsplit(' ',1)[0],
        fever = float(row["symptom:Fever"]) ,
        chills = float(row["symptom:Chills"]),
        cough = float(row["symptom:Cough"]) ,
        shortness_of_breath = float(row["symptom:Shortness of breath"]),
        shallow_breathing = float(row["symptom:Shallow breathing"]),
        fatigue = float(row["symptom:Fatigue"]),
        headache = float(row["symptom:Headache"]),
        sore_throat = float(row["symptom:Sore throat"]),
        nasal_congestion = float(row["symptom:Nasal congestion"]),
        nausea = float(row["symptom:Nausea"]),
        vomiting = float(row["symptom:Vomiting"]),
        diarrhea = float(row["symptom:Diarrhea"]),
        dysguesia = float(row["symptom:Dysgeusia"]), 
        ageusia = float(row["symptom:Ageusia"]), 
        anosmia = float(row["symptom:Anosmia"]), 
        myalgia = float(row["symptom:Myalgia"])))


total_sypmtoms_datapoints = last_year_datapoints + this_year_datapoints

# add in new weekly COVID cases to datapoints 

# setup dataclass 
@dataclass
class CasesData:
    date: str
    county: str 
    cases: int 

cases_datapoints: List[CasesData] = []

# testing 
new_york = 0
num_rows = 0 

with open("cases_data.csv") as csv_file: 
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    list_reader = list(csv_reader)

    for i in range(len(list_reader)): 
        row = list_reader[i]

        if row["state"] == "New York" and row["date"] in relevant_dates:
            current_cases = int(row["cases"])
            next_days = 6 
            rows_ahead = 1 

            while next_days != 0: 
                next_row = list_reader[i + rows_ahead]
                if next_row["county"] == row["county"]: # the same county -- so next day in the week 
                    current_cases += int(next_row["cases"])
                    next_days -= 1
                rows_ahead += 1 

            cases_datapoints.append(CasesData(
                date = row["date"],
                county = row["county"],
                cases = current_cases 
            ))


#print("num symptoms datapoints = " + str(len(total_sypmtoms_datapoints)))
print("num cases  = " + str(len(cases_datapoints)))
print(cases_datapoints[0])










"""
# set up for ML 

examples = []
ys = []

with open(dataset_local_path("poetry_id.jsonl")) as fp:
    for line in fp:
        info = json.loads(line)
        # Note: the data contains a whole bunch of extra stuff; we just want numeric features for now.
        keep = info["features"]
        # whether or not it's poetry is our label.
        ys.append(info["poetry"])
        # hold onto this single dictionary.
        examples.append(keep)

## CONVERT TO MATRIX:

feature_numbering = DictVectorizer(sort=True)
X = feature_numbering.fit_transform(examples) / 1000

print("Features as {} matrix.".format(X.shape))


## SPLIT DATA:

RANDOM_SEED = 12345678

# Numpy-arrays are more useful than python's lists.
y = np.array(ys)
# split off train/validate (tv) pieces.
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, train_size=0.75, shuffle=True, random_state=RANDOM_SEED
)
# split off train, validate from (tv) pieces.
X_train, X_vali, y_train, y_vali = train_test_split(
    X_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
)

print(X_train.shape, X_vali.shape, X_test.shape)
"""