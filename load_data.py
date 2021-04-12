# Anna Spiro
# ML Course Project: COVID Search Data (NY Counties)

import csv
import pickle

# stdlib:
from dataclasses import dataclass
from typing import List, Set

# import google symptoms data
# data found here: https://pair-code.github.io/covid19_symptom_dataset/?country=US (chose to use NY county data)
# code to read csv adapted from https://realpython.com/python-csv/

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
    dysguesia: float  # partial loss of taste
    ageusia: float  # total loss of taste
    anosmia: float  # loss of smell
    myalgia: float  # muscle pain

# note: symptoms chosen based on this list: https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html

def read_in_csv(filename, start_row, end_row, date_constraint):
    """
    Return datapoints as List[SymptomsData] and relevant dates as Set(str)
    """
    datapoints: List[SymptomsData] = []

    # keep relevant dates as a set so that it's faster to find relevant casese datapoints
    relevant_dates: Set[str] = set()

    # relevant_dates = set()

    with open(filename) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")

        # skip rows that are combined NY data)
        relevant_rows = [
            row
            for idx, row in enumerate(csv_reader)
            if idx in range(start_row, end_row)
        ]

        for row in relevant_rows:
            # get rid of "" values
            if (
                row["symptom:Shallow breathing"] == ""
                or row["symptom:Chills"] == ""
                or row["symptom:Cough"] == ""
                or row["symptom:Shortness of breath"] == ""
                or row["symptom:Shallow breathing"] == ""
                or row["symptom:Fatigue"] == ""
                or row["symptom:Headache"] == ""
                or row["symptom:Sore throat"] == ""
                or row["symptom:Nasal congestion"] == ""
                or row["symptom:Nausea"] == ""
                or row["symptom:Vomiting"] == ""
                or row["symptom:Diarrhea"] == ""
                or row["symptom:Dysgeusia"] == ""
                or row["symptom:Ageusia"] == ""
                or row["symptom:Anosmia"] == ""
                or row["symptom:Myalgia"] == ""
            ):
                continue

            if (
                date_constraint
            ):  # only necessary for 2020 data (first COVID case confirmed on 2020-03-01)
                if int(row["date"][5:7]) < 3:
                    continue

            relevant_dates.add(row["date"])

            datapoints.append(
                SymptomsData(
                    date=row["date"],
                    county=row["sub_region_2"].rsplit(" ", 1)[0],  # don't want "County"
                    fever=float(row["symptom:Fever"]),
                    chills=float(row["symptom:Chills"]),
                    cough=float(row["symptom:Cough"]),
                    shortness_of_breath=float(row["symptom:Shortness of breath"]),
                    shallow_breathing=float(row["symptom:Shallow breathing"]),
                    fatigue=float(row["symptom:Fatigue"]),
                    headache=float(row["symptom:Headache"]),
                    sore_throat=float(row["symptom:Sore throat"]),
                    nasal_congestion=float(row["symptom:Nasal congestion"]),
                    nausea=float(row["symptom:Nausea"]),
                    vomiting=float(row["symptom:Vomiting"]),
                    diarrhea=float(row["symptom:Diarrhea"]),
                    dysguesia=float(row["symptom:Dysgeusia"]),
                    ageusia=float(row["symptom:Ageusia"]),
                    anosmia=float(row["symptom:Anosmia"]),
                    myalgia=float(row["symptom:Myalgia"]),
                )
            )

    return datapoints, relevant_dates

# combine csv data from 2020 and 2021

last_year_info = read_in_csv("2020_NY_data.csv", 53, 3277, True)
this_year_info = read_in_csv("2021_NY_data.csv", 11, 631, False)

# note: len of this = 1024
total_sypmtoms_datapoints = last_year_info[0] + this_year_info[0]

relevant_dates = last_year_info[1].union(this_year_info[1])

# get cases data 

@dataclass
class CasesData:
    date: str
    county: str
    cases: int

cases_datapoints: List[CasesData] = []

with open("cases_data.csv") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    list_reader = list(csv_reader)

    for i in range(len(list_reader)):
        row = list_reader[i]

        if row["state"] == "New York" and row["date"] in relevant_dates:
            current_cases = int(row["cases"])
            next_days = 6
            rows_ahead = 1

            # case data is given by day, we want weekly sum
            while next_days != 0:
                next_row = list_reader[i + rows_ahead]
                if (
                    next_row["county"] == row["county"]
                ):  # the same county -- so this row is the next day in the week
                    current_cases += int(next_row["cases"])
                    next_days -= 1
                rows_ahead += 1

            cases_datapoints.append(
                CasesData(date=row["date"], county=row["county"], cases=current_cases)
            )

# join symptoms and cases data
# decision: if there is no case data, make cases = 0 

@dataclass
class JoinedData:
    date: str
    county: str
    symptoms: List[float]
    cases: int

joined_datapoints: List[JoinedData] = []

for symptoms_datapoint in total_sypmtoms_datapoints:
    current_date = symptoms_datapoint.date
    current_county = symptoms_datapoint.county
    current_cases = 0

    # list of symptoms data
    current_symptoms = [
        symptoms_datapoint.fever,
        symptoms_datapoint.chills,
        symptoms_datapoint.cough,
        symptoms_datapoint.shortness_of_breath,
        symptoms_datapoint.shallow_breathing,
        symptoms_datapoint.fatigue,
        symptoms_datapoint.headache,
        symptoms_datapoint.sore_throat,
        symptoms_datapoint.nasal_congestion,
        symptoms_datapoint.nausea,
        symptoms_datapoint.vomiting,
        symptoms_datapoint.diarrhea,
        symptoms_datapoint.dysguesia,
        symptoms_datapoint.ageusia,
        symptoms_datapoint.anosmia,
        symptoms_datapoint.myalgia,
    ]

    for cases_datapoint in cases_datapoints:

        if (
            cases_datapoint.date == symptoms_datapoint.date
            and cases_datapoint.county == symptoms_datapoint.county
        ):
            current_cases = cases_datapoint.cases

    joined_datapoints.append(
        JoinedData(
            date=current_date,
            county=current_county,
            symptoms=current_symptoms,
            cases=current_cases,
        )
    )

saved_data = open(r'C:\d.pkl', 'wb')
pickle.dump(joined_datapoints, saved_data)
saved_data.close()

