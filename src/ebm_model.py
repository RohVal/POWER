from preprocessing import preprocess_data
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import save_model, split_data

from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show
from interpret import preserve

df = preprocess_data(path="../dataset/Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv")
df = df.drop(columns=['Date and time'])

X_train, X_test, y_train, y_test = split_data(df=df)

ebm = ExplainableBoostingRegressor()
ebm.fit(X_train, y_train)
preserve(ebm.explain_global())
preserve(show(ebm.explain_local(X_test[:5], y_test[:5]), 0))