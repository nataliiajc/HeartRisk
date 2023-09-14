
"""
This model predict the possibility of heart disease from 11 features contained in the dataset.

This data has been obtained from : https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import zipfile
from pathlib import Path

file_name = "heartrisk.zip"
general_dir = os.path.join(os.getcwd(),'heart')
zip_ref = zipfile.ZipFile(file_name,'r')
zip_ref.extractall(general_dir)
zip_ref.close()
print(general_dir)

"""We want to predict if patients will suffer heart disease

Attribute Information
- Age: age of the patient [years]
- Sex: sex of the patient [M: Male, F: Female]
- ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP: resting blood pressure [mm Hg]
- Cholesterol: serum cholesterol [mm/dl]
- FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
- ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
- Oldpeak: oldpeak = ST [Numeric value measured in depression]
- ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- HeartDisease: output class [1: heart disease, 0: Normal]
"""

# Leemos el CSV
data = pd.read_csv('/content/heart/heart.csv')

data.head()

data.shape

data.columns

y = data['HeartDisease']
y_np = np.array([y])
print("Zeros " + str(np.count_nonzero(y_np)))
nonzeros = y_np.size - np.count_nonzero(y_np)
print( "Nonzeros " + str(nonzeros))

""" - We want to obtain the probability of a patient to suffer from a heart disease.
- The number of people with heart disease is similar to the number of people without heart disease, so we can consider it is a  balanced problem
"""

X = data.iloc[:,:-1]
Y = data.iloc[:,-1:]
print(X.shape)
print(Y.shape)

#We split the data in test and train
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


categories = [('Sex',['F','M']),
              ('ChestPainType',['ASY','NAP','ATA','TA']),
              ('RestingECG',['Normal','ST','LVH']),
              ('ExerciseAngina',['N','Y']),
              ('ST_Slope',['Flat','Up','Down'])]

ohe_columns = [x[0] for x in categories]
ohe_categories = [x[1] for x in categories]
enc = OneHotEncoder(sparse_output=False, categories=ohe_categories)

from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

param_grid = {'model__max_features': [ 'sqrt', 'log2'],
              'model__max_depth' : [5, 6, 7, 8, 9],
              'model__criterion' :['gini', 'entropy']
             }

tree = tree.DecisionTreeClassifier(random_state = 42)

data_pipe = Pipeline(steps=[
    ('one-hot',OneHotEncoder(sparse_output=False, categories=ohe_categories)),
    ('scale',MinMaxScaler())
])

col_trans = ColumnTransformer(transformers=[
    ('pipeline',data_pipe,ohe_columns)])

final_pipe = Pipeline(steps=[
    ('col_trans', col_trans),
    ('model', tree)
])

grid_search = GridSearchCV(final_pipe, param_grid=param_grid, cv=5,scoring='accuracy', verbose=True)
grid_search.fit(X_train,Y_train)

print("Best Score of train set: "+str(grid_search.best_score_))
print("Best parameter set: "+str(grid_search.best_params_))
print("Test Score: "+str(grid_search.score(X_test,Y_test)))

from sklearn import set_config

set_config(display='diagram')
display(final_pipe)

example_data = pd.DataFrame(
    [[	20,	'F',	'ATA',	140,	289,	0,	'Normal',	122,	'N', 0.0,	'Up']],
    columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
       'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
)
print(example_data)

grid_search.predict(example_data)

print(grid_search.best_estimator_)

import pickle
pkl_filename = "pickle_model.plk"
with open(pkl_filename,'wb') as file:
  pickle.dump(grid_search.best_estimator_,file)

with open(pkl_filename, "rb") as file:
    new_model = pickle.load(file)
prediction = new_model.predict(example_data) # Passing in variables for prediction
print("The result is",prediction)

"""fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [Date Retrieved] from https://www.kaggle.com/fedesoriano/heart-failure-prediction."""
