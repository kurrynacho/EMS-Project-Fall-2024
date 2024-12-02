import statsmodels.tsa.api as sm
from statsmodels.tsa.holtwinters import Holt
import itertools
import importlib
import pandas as pd
from pmdarima.pipeline import Pipeline
from pmdarima.arima import auto_arima
from pmdarima.arima import AutoARIMA
from scipy.stats import boxcox
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import preprocessing
from scipy.stats import boxcox
from scipy.special import inv_boxcox

importlib.reload(preprocessing)
def monthly_prophet_predict(data,s):
    return prophet_predict(data,s,monthly=True)
def weekly_prophet_predict(data,s):
    return prophet_predict(data,s,weekly=True)
def prophet_predict(data, s, lockdown=False, monthly=False, weekly=False):
    prophet_df=pd.DataFrame({'ds': data['date'], 'y': data['count']})
    if lockdown==True:
        lockdowns = pd.DataFrame([
        {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
        {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
        {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
        {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
    ])
        for t_col in ['ds', 'ds_upper']:
            lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
        lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
        m = Prophet(seasonality_mode='additive', holidays=lockdowns, weekly_seasonality=True)
    else:
        m = Prophet(seasonality_mode='additive', weekly_seasonality=True)

    m.fit(prophet_df)
    if monthly:
        future = m.make_future_dataframe(periods=s, freq='M')
    elif weekly:
        future = m.make_future_dataframe(periods=s, freq='W')
    else:
        future = m.make_future_dataframe(periods=s)
    forecast = m.predict(future)
    return forecast['yhat'].iloc[-s:]

def get_residue_arima(data, type='mult', window=90, m=7):
    trend = preprocessing.get_trend(data, window_size=window)
    if type=='mult':
        residue_df=pd.DataFrame({'date': data['date'], 'count': data['count']/trend})
    if type=='add':
        residue_df=pd.DataFrame({'date': data['date'], 'count': data['count']-trend})
    arima = auto_arima(residue_df['count'], m=m, stepwise=True, information_criterion='bic')
    return arima


def expsmoothing_predict(data, s):
    model = Holt(data['count']).fit(optimized=True)

    # Forecast the next s time steps
    predictions = model.forecast(steps=s)
    return predictions

def autoarima_predict(data, s, m=7):
    pipeline = Pipeline(steps=[
        ("arima", AutoARIMA(stepwise=True,m=m, information_criterion  = 'bic'))])
    pipeline.fit(data['count'])
    predictions = pipeline.predict(s)
    return predictions
    
def smoothing_arima(data, s, type='mult', window=90, arima=None):
    trend = preprocessing.get_trend(data, window_size=window)
    trend_df=pd.DataFrame({'date': data['date'], 'count': trend})

    trend_predictions= expsmoothing_predict(trend_df,s)
    if type == 'mult':
        detrended = data['count']/trend
        residual_df=pd.DataFrame({'date': data['date'], 'count': detrended})
        if arima:
            arima.fit(residual_df['count'])
            residual_predictions=arima.predict(n_periods=s)
        else:
            residual_predictions= autoarima_predict(residual_df,s)
        return trend_predictions*residual_predictions
    elif type == 'add':
        detrended = data['count']-trend
        residual_df=pd.DataFrame({'date': data['date'], 'count': detrended})
        if arima:
            arima.fit(residual_df['count'])
            residual_predictions=arima.predict(n_periods=s)
        else:
            residual_predictions= autoarima_predict(residual_df,s)
        return trend_predictions+residual_predictions
    else:
        raise ValueError("third input must be mult or add")
    
def prophet_arima(data, s, type='mult', window=90, arima=None):

    trend = preprocessing.get_trend(data, window_size=window)

    trend_df = pd.DataFrame({'date': data['date'], 'count': trend})

    trend_predictions= prophet_predict(trend_df,s)

    if type == 'mult':
        detrended = data['count']/trend
        residual_df=pd.DataFrame({'date': data['date'], 'count': detrended})
        if arima:
            arima.fit(residual_df['count'])
            residual_predictions=arima.predict(n_periods=s)
        else:
            residual_predictions= autoarima_predict(residual_df,s)
        return trend_predictions*residual_predictions
    elif type == 'add':
        detrended = data['count']-trend
        residual_df=pd.DataFrame({'date': data['date'], 'count': detrended})
        if arima:
            arima.fit(residual_df['count'])
            residual_predictions=arima.predict(n_periods=s)
        else:
            residual_predictions= autoarima_predict(residual_df,s)
        return trend_predictions+residual_predictions
    
    else:
        raise ValueError("third input must be mult or add")
    
def monthly_prophet_predict_withlockdown(data,s):
    return prophet_predict_withlockdown(data, s, monthly=True)
def weekly_prophet_predict_withlockdown(data,s):
    return prophet_predict_withlockdown(data,s,weekly=True)
def prophet_predict_withlockdown(data, s, monthly=False, weekly=False):
    prophet_df=pd.DataFrame({'ds': data['date'], 'y': data['count']})

    lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
    {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
    {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
    {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
])
    for t_col in ['ds', 'ds_upper']:
        lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
    lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
    m = Prophet(seasonality_mode='multiplicative', holidays=lockdowns, weekly_seasonality=True)
    m.fit(prophet_df)
    m.fit(prophet_df)
    if monthly:
        future = m.make_future_dataframe(periods=s, freq='M')
    elif weekly:
        future = m.make_future_dataframe(periods=s, freq='W')
    else:
        future = m.make_future_dataframe(periods=s)
    forecast = m.predict(future)
    return forecast['yhat'].iloc[-s:]

def prophet_arima_withlockdown(data, s, type='mult', window=90, arima=None):

    trend = preprocessing.get_trend(data, window_size=window)

    trend_df = pd.DataFrame({'date': data['date'], 'count': trend})

    trend_predictions= prophet_predict_withlockdown(trend_df,s)

    if type == 'mult':
        detrended = data['count']/trend
        residual_df=pd.DataFrame({'date': data['date'], 'count': detrended})
        if arima:
            arima.fit(residual_df['count'])
            residual_predictions=arima.predict(n_periods=s)
        else:
            residual_predictions= autoarima_predict(residual_df,s)
        return trend_predictions*residual_predictions
    elif type == 'add':
        detrended = data['count']-trend
        residual_df=pd.DataFrame({'date': data['date'], 'count': detrended})
        if arima:
            arima.fit(residual_df['count'])
            residual_predictions=arima.predict(n_periods=s)
        else:
            residual_predictions= autoarima_predict(residual_df,s)
        return trend_predictions+residual_predictions
    
    else:
        raise ValueError("third input must be mult or add")

def prophet_arima_withlockdown(data, s, type='mult', window=90, arima=None):

    trend = preprocessing.get_trend(data, window_size=window)

    trend_df = pd.DataFrame({'date': data['date'], 'count': trend})

    trend_predictions= prophet_predict_withlockdown(trend_df,s)

    if type == 'mult':
        detrended = data['count']/trend
        residual_df=pd.DataFrame({'date': data['date'], 'count': detrended})
        if arima:
            arima.fit(residual_df['count'])
            residual_predictions=arima.predict(n_periods=s)
        else:
            residual_predictions= autoarima_predict(residual_df,s)
        return trend_predictions*residual_predictions
    elif type == 'add':
        detrended = data['count']-trend
        residual_df=pd.DataFrame({'date': data['date'], 'count': detrended})
        if arima:
            arima.fit(residual_df['count'])
            residual_predictions=arima.predict(n_periods=s)
        else:
            residual_predictions= autoarima_predict(residual_df,s)
        return trend_predictions+residual_predictions
    
    else:
        raise ValueError("third input must be mult or add")

def naive_predict(data, s):
    last_value = data.iloc[-1]['count']
    return last_value * np.ones(s)

def constant_predict(data, s):
    return np.average(data['count'])*np.ones(s)

def ttsplit_predictions(data, f, s, extra_models=[], smoothing_params=None, printprogress=False, arima_trace=False, do_arima=True, fixed_residue_models=False, residue_window_size=60, weekly=False, monthly=False):
    # Base models
    model_list = [naive_predict, constant_predict] + extra_models

    # Initialize predictions dictionary
    predictions_dict = {}
    if do_arima and weekly:
        predictions_dict['arima'] = []
        arima = auto_arima(data['count'].values[:(s * (-f))], m=4, stepwise=True, information_criterion='bic')
    elif do_arima and monthly:
        predictions_dict['arima'] = []
        arima = auto_arima(data['count'].values[:(s * (-f))], m=12, stepwise=True, information_criterion='bic')
    elif do_arima:
        predictions_dict['arima'] = []
        arima = auto_arima(data['count'].values[:(s * (-f))], m=7, stepwise=True, information_criterion='bic')

    mult_residue_arima=None
    add_residue_arima=None
    if fixed_residue_models:
        if monthly:
            mult_residue_arima=get_residue_arima(data, type='mult', window=residue_window_size, m=12)
            "add_residue_arima=get_residue_arima(data, type='add', window=residue_window_size,m=12)"
        if weekly:
            mult_residue_arima=get_residue_arima(data, type='mult', window=residue_window_size, m=4)
            "add_residue_arima=get_residue_arima(data, type='add', window=residue_window_size,m=4)"
        else:
            mult_residue_arima=get_residue_arima(data, type='mult', window=residue_window_size, m=7)
            "add_residue_arima=get_residue_arima(data, type='add', window=residue_window_size,m=7)"


    # Initialize predictions for all models
    for model in model_list:
        predictions_dict[model.__name__] = []

    # Handle smoothing_arima parameter grid
    if smoothing_params:
        smoothing_param_combinations = list(itertools.product(*smoothing_params.values()))
        for param_combo in smoothing_param_combinations:
            param_key = f"smoothing_arima_{'_'.join(map(str, param_combo))}"
            predictions_dict[param_key] = []
            param_key = f"prophet_arima_{'_'.join(map(str, param_combo))}"
            predictions_dict[param_key] = []
            param_key = f"prophet_arima_withlockdown_{'_'.join(map(str, param_combo))}"
            predictions_dict[param_key] = []

    # Perform time series splitting
    for i in range(-f, 0):
        if printprogress:
            print("Starting fold " + str(i))

        # Create train data for current fold
        y_tt = data.iloc[:(s * i)].copy()

        # Predict with base models
        for model in model_list:
            predictions_dict[model.__name__].append(model(y_tt, s))
            if printprogress:
                print(model.__name__ + " done")

        # Predict with ARIMA
        if do_arima:
            if i > -f:
                arima.update(data['count'].values[s * (i - 1):s * i])
            arima_predictions = arima.predict(n_periods=s)
            predictions_dict['arima'].append(arima_predictions)
            if printprogress:
                print("arima done")

        # Predict with smoothing_arima for each parameter combination
        if smoothing_params:
            for param_combo in smoothing_param_combinations:
                if 'mult' in param_combo:
                    param_key = f"smoothing_arima_{'_'.join(map(str, param_combo))}"
                    smoothing_kwargs = dict(zip(smoothing_params.keys(), param_combo))
                    predictions_dict[param_key].append(smoothing_arima(y_tt, s,arima=mult_residue_arima, **smoothing_kwargs))

                    if printprogress:
                        print(param_key + " done")

                    param_key = f"prophet_arima_{'_'.join(map(str, param_combo))}"
                    smoothing_kwargs = dict(zip(smoothing_params.keys(), param_combo))
                    predictions_dict[param_key].append(prophet_arima(y_tt, s, arima=mult_residue_arima,**smoothing_kwargs))

                    if printprogress:
                        print(param_key + " done")

                    param_key = f"prophet_arima_withlockdown_{'_'.join(map(str, param_combo))}"
                    smoothing_kwargs = dict(zip(smoothing_params.keys(), param_combo))
                    predictions_dict[param_key].append(prophet_arima_withlockdown(y_tt, s, arima=mult_residue_arima,**smoothing_kwargs))
                    if printprogress:
                        print(param_key + " done")

                if 'add' in param_combo:
                    param_key = f"smoothing_arima_{'_'.join(map(str, param_combo))}"
                    smoothing_kwargs = dict(zip(smoothing_params.keys(), param_combo))
                    predictions_dict[param_key].append(smoothing_arima(y_tt, s,arima=add_residue_arima, **smoothing_kwargs))

                    if printprogress:
                        print(param_key + " done")

                    param_key = f"prophet_arima_{'_'.join(map(str, param_combo))}"
                    smoothing_kwargs = dict(zip(smoothing_params.keys(), param_combo))
                    predictions_dict[param_key].append(prophet_arima(y_tt, s, arima=add_residue_arima,**smoothing_kwargs))

                    if printprogress:
                        print(param_key + " done")

                    param_key = f"prophet_arima_withlockdown_{'_'.join(map(str, param_combo))}"
                    smoothing_kwargs = dict(zip(smoothing_params.keys(), param_combo))
                    predictions_dict[param_key].append(prophet_arima_withlockdown(y_tt, s, arima=add_residue_arima,**smoothing_kwargs))
                    if printprogress:
                        print(param_key + " done")

    # Concatenate predictions for each model
    for model_name in predictions_dict:
        predictions_dict[model_name] = np.concatenate(predictions_dict[model_name])

    return predictions_dict

def bc_ttsplit_predictions(data, f, s, extra_models=[], smoothing_params=None, printprogress=False, arima_trace=False, do_arima=True, fixed_residue_models=False, residue_window_size=90):
    
    bc_data = data.copy()
    bc_data['count'], lam =boxcox(data['count'])

    bc_predictions=ttsplit_predictions(bc_data, f, s, extra_models=extra_models, smoothing_params=smoothing_params, printprogress=printprogress, arima_trace=arima_trace, do_arima=do_arima, fixed_residue_models=fixed_residue_models, residue_window_size=residue_window_size)
    for model in bc_predictions:
        bc_predictions[model] = inv_boxcox(bc_predictions[model], lam)
    return bc_predictions

def holdout_values(data,f,s):
    return data['count'].values[-f*s:]

def evaluate_predictions(actual_values, predictions_dict):
    df = pd.DataFrame(columns = ['Model', 'MSE', 'MAE', 'NMSE'])
    naive_mse = mean_squared_error(actual_values,predictions_dict['naive_predict'])
    for model in predictions_dict:
        mse = mean_squared_error(actual_values, predictions_dict[model])
        mae = mean_absolute_error(actual_values, predictions_dict[model])
        df.loc[len(df)]=[model, mse, mae, mse/naive_mse]

    return df

def train_test_split(data, f, s, extra_models =[]):
    predictions_dict = ttsplit_predictions(data, f, s, extra_models)
    actual_values = data[-f*s:]['count']
    return evaluate_predictions(actual_values, predictions_dict)

def bc_train_test_split(data, f, s, extra_models =[]):
    predictions_dict = bc_ttsplit_predictions(data, f, s, extra_models)
    actual_values = data['count'][-f*s:]
    return evaluate_predictions(actual_values, predictions_dict)
