## Models to be used and discussed in the report:
- Linear Regression: multiple & ridge
- XGBoost: with parameter optimization algorithms (grid search, Bayessian) and without (interpretable: low number of trees, low depth)
- LSTM (important for the field of wind power forecasting)
- EBM / IGANN  (GAM)


## To-Dos:
- <del>Preprocessing: discuss exclude observations where: Y = 0 & Wind Speed < Cut-In speed</del>
- Preproccesing: explain negative power values
- <del>XGBoost: figure out what a leaf value represents</del>


## Web-Application:
- Pages for different Models
- Possibility to make predictions for different turbines separately and for the whole wind farm (all the turbines combined)
- Include different accuracy metrics in the output (MAE, R2, etc.)
- Replace wind direction values with nominal ones

## Questions to the prof:
- Is it ok to do the final presentation without PowerPoint (by simply showing our code, relevant statistical graphics and the application)?

## Questions for Charlie:
- Where might negative power values come from? (~9% of the rows)

_____________________________________

## Stucture of the preliminary presentation

- Introduction (Petr Kh.):
  - Current state of affairs, overview
  - Introducing the dataset, Feature Selection & Technical/Physical infos (+ graphic with linreg)
  - Pre-Processing (6 rows for hourly predictions)
  - A brief literature overview (research trends, approaches)
  - Challenges

- Models (Nikolas & Rohit):
  - XGBoost
    - Hyperparameters Optimization: Grid Search, Bayessian, manual/default
    - Results (incl. comparison)
  - LSTM
    - Hyperparameters Optimization
    - Results
  - EBM
    - Results
  - IGANN
    - (how it is interpretable)
    - Hyperparameters Optimization
    - Results

- Application (Peter):
  - Current state, how it works
  - How it will look like:
    - Feature: choosing a model (method)
    - Feature: compare the results of two different models (methods)
    - Feature: choose a single turbine / whole wind farm
