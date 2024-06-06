from preprocessing import preprocess_data
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import save_model, split_data

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

df = preprocess_data(path='/content/drive/MyDrive/BI /first prototype/Turbine_Data_Kelmarsh_1_clean.csv')
df = df.drop(columns=['Date and time'])

X_train, X_test, y_train, y_test = split_data(df=df)

ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)
# show(ebm.explain_global())