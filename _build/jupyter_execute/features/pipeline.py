# Übung: Tranformation

Datasets erstellen (Splitten) evtl. in Aufbereitung?!

Packete importieren

import pandas as pd
import numpy as np
import sklearn as sklearn
import pickle
from sklearn.model_selection import train_test_split

Laden der Datensets aus Pickle File

with open('datasets.pickle', 'rb') as handle:
    datasets = pickle.load(handle)

