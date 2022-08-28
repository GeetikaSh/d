#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
from yahoo_fin.stock_info import get_data
import yahoo_fin.stock_info as si
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
import plotly.graph_objects as go
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import lag_plot
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

def get_stock_data(ticker, start_date, end_date):
    stock_data= get_data(ticker, start_date=start_date.strftime("%m/%d/%Y"), end_date=end_date.strftime("%m/%d/%Y"), index_as_date = False, interval='1d')
    return stock_data  
    
all_stocks = si.tickers_nasdaq()

sidebar = st.sidebar
st.title(f"MA7098 DABI Project")
st.markdown("This application has been designed to test and compare multiple machine learning models on time series data and forecast them effectively")

sidebar.header("Pick a ticker:")
with sidebar.expander("View Data"):
    dataform = st.form(key = "dform")
    stock_selector = dataform.selectbox(
        "Select a Ticker",
        all_stocks)
    ss_idx = all_stocks.index(stock_selector)
    d = dataform.date_input(
        "Please select the Time Period for analysis",
        value = (datetime.date(2001, 1, 1), datetime.date.today()),
        min_value = datetime.date(2001, 1, 1),
        max_value = datetime.date.today())
    trend_level = dataform.selectbox("Trend Level", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                               help="This will group and display data based on your input")
    chart_selector = dataform.selectbox("Chart Type: ", ('Line','Bar','Candle Stick'))
    show_data = dataform.checkbox("Show Data")
    databutton = dataform.form_submit_button(label='View Data')
    
with sidebar.expander("All models"):
    modelform = st.form(key = "modform")
    stock_selector = modelform.selectbox("Select a Ticker",all_stocks, index = ss_idx)
    model_selector = modelform.selectbox("Select a Model", ["ARIMA", "LSTM - Log Returns & Close Price"])
    n = modelform.selectbox("Select the number of days to look back", [3,7,30,60,100], help="Implemented only for LSTM")
    modelbutton = modelform.form_submit_button(label='Show Results!')

if modelbutton:
    if model_selector == "LSTM - Log Returns & Close Price":
        stock = stock_selector
        precovid_df = get_data(stock, start_date="01/01/2015", end_date="12/31/2019", index_as_date = True, interval="1d")
        covid_df = get_data(stock, start_date="01/01/2020", end_date="06/30/2020", index_as_date = True, interval="1d")
        postcovid_df = get_data(stock, start_date="07/01/2020", end_date="07/18/2022", index_as_date = True, interval="1d")
        df = precovid_df.copy()
        df["returns"] = df.close.pct_change()
        df["log_returns"] = np.log(1+ df["returns"])
        fig = plt.figure(figsize = (10, 5))
        plt.plot(df.log_returns, label = "Log Returns")
        plt.legend()
        st.pyplot(fig)
        df.dropna(inplace = True)
        X = df[["close","log_returns"]]
        scaler = MinMaxScaler(feature_range=(0,1)).fit(X)
        X_scaled = scaler.transform(X)
        y = [x[0] for x in X_scaled]
        split = int(len(X_scaled)*0.8)
        X_train = X_scaled[:split]
        X_test = X_scaled[split:len(X_scaled)]
        Y_train = y[:split]
        Y_test = y[split:len(y)]
        #n=3
        n = int(n)
        Xtrain=[]
        ytrain = []
        Xtest = []
        ytest = []
        for i in range(n, len(X_train)):
          Xtrain.append(X_train[i-n : i, :X_train.shape[1]])
          ytrain.append(Y_train[i])
        for i in range(n, len(X_test)):
          Xtest.append(X_test[i-n:i, :X_test.shape[1]])
          ytest.append(Y_test[i])
        val = np.array(ytrain[0])
        val = np.c_[val, np.zeros(val.shape)]
        scaler.inverse_transform(val)
        Xtrain, ytrain = (np.array(Xtrain), np.array(ytrain))
        Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))
        Xtest, ytest = (np.array(Xtest), np.array(ytest))
        Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))
        model = Sequential()
##        model.add(LSTM(4, input_shape = (Xtrain.shape[1], Xtrain.shape[2])))
##        model.add(Dense(1))
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))

        model.add(Dense(units = 1))
        model.compile(loss="mean_squared_error", optimizer = "adam")
        model.fit(Xtrain, ytrain, epochs=100, validation_data=(Xtest, ytest), batch_size=16, verbose =1)
        model.summary(print_fn=lambda x: st.text(x))
        trainPredict = model.predict(Xtrain)
        testPredict = model.predict(Xtest)
        trainPredict = np.c_[trainPredict, np.zeros(trainPredict.shape)]
        testPredict = np.c_[testPredict, np.zeros(testPredict.shape)]
        trainPredict = scaler.inverse_transform(trainPredict)
        trainPredict = [x[0] for x in trainPredict]
        testPredict = scaler.inverse_transform(testPredict)
        testPredict = [x[0] for x in testPredict]
        trainScore = mean_squared_error(list(df["close"][n:split]), trainPredict, squared=False)
        st.write(f"Train Score: {trainScore}")
        testScore = mean_squared_error(list(df["close"][split+n:]), testPredict, squared=False)
        st.write(f"Test Score: {testScore}")
        dff = pd.DataFrame({"actual":list(df["close"][n:split]),"predicted":trainPredict})
        fig = plt.figure(figsize = (10, 5))
        plt.plot(dff["actual"], label="Actual")
        plt.plot(dff["predicted"], label="Predicted")
        plt.title("Model Performance on Train Data: ")
        plt.legend()
        st.pyplot(fig)
        dff = pd.DataFrame({"actual":list(df["close"][split+n:]),"predicted":testPredict})
        fig = plt.figure(figsize = (10, 5))
        plt.plot(dff["actual"], label = "Actual")
        plt.plot(dff["predicted"], label="Predicted")
        plt.title("Model Performance on Test Data: ")
        plt.legend()
        st.pyplot(fig)
        fig = plt.figure(figsize = (10, 5))
        plt.title("Model Performance on Post-Covid Data")
    if model_selector == "ARIMA":
        stock = stock_selector
        precovid_df = get_data(stock, start_date="01/01/2015", end_date="12/31/2019", index_as_date = True, interval="1d")
        covid_df = get_data(stock, start_date="01/01/2020", end_date="06/30/2020", index_as_date = True, interval="1d")
        postcovid_df = get_data(stock, start_date="07/01/2020", end_date="07/18/2022", index_as_date = True, interval="1d")
        df = precovid_df.copy()
        lag = 1
        st.line_chart(df["close"])
        fig = plt.figure(figsize = (10, 5))
        lag_plot(df['close'], lag=lag)
        plt.title(f"{stock} - Autocorrelation plot with lag = {lag}")
        st.pyplot(fig)
        df["close"] = df["close"].interpolate(method="linear")
        model = pm.auto_arima(df.close, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
        st.write(model.summary())
        fig = model.plot_diagnostics(figsize=(7,5))
        st.pyplot(fig)
        n_periods = 185
        fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
        df["date"] = df.index
        index_of_fc = np.arange(len(df.close), len(df.close)+n_periods)
        index_of_fc = pd.date_range(start=df["date"].iloc[-1], periods=n_periods+1, freq='d', closed='right')

        # make series for plotting purpose
        fc_series = pd.Series(fc, index=index_of_fc)
        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)
        # Plot
        fig = plt.figure(figsize = (10, 5))
        plt.plot(df.close, label="Actual")
        plt.plot(postcovid_df.close, label="Post CoViD")
        plt.plot(covid_df.close, label = "During CoVid")
        #plt.plot(fc_series, color='darkgreen', label = "Forecast")
        plt.fill_between(lower_series.index, 
                         lower_series, 
                         upper_series, 
                         color='k', alpha=.15, label="Prediction Range")
        plt.legend()
        plt.title("Final Forecast")
        st.pyplot(fig)

if databutton:
    st.markdown(f"Currently Selected Ticker: {stock_selector}")
    st.write('Currently Selected Time Period: ', d[0].strftime("%A - %d %B, %Y"), ' to ', d[1].strftime("%A - %d %B, %Y"))
    data = get_stock_data(stock_selector, d[0], d[1])
    st.markdown(f"Currently Selected trend level: {trend_level}")
    trend_kwds = {"Daily": "1D", "Weekly": "1W", "Monthly": "1M", "Quarterly": "1Q", "Yearly": "1Y"}
    trend_data = data.copy()
    trend_data = trend_data.resample(trend_kwds[trend_level], on='date', origin='start').agg(
        {"open": "first",
         "close": "last",
         "low": "min",
         "high": "max",
         }#"volume" = ("volume","sum")}
        ).dropna()[['open', 'high', 'low', 'close']].reset_index()
    if show_data:
        st.write(f"The raw data for {stock_selector} can be found below: ")
        st.dataframe(trend_data)
    chart_dict = {'Line':'line','Bar':'bar'}
    if chart_selector == 'Candle Stick':
        fig = go.Figure(data=[go.Candlestick(x=trend_data['date'],
                open=trend_data['open'],
                high=trend_data['high'],
                low=trend_data['low'],
                close=trend_data['close'])])
        st.plotly_chart(fig, use_container_width=False)
    else:
        fig=trend_data.iplot(kind=chart_dict[chart_selector], asFigure=True, xTitle="Date", yTitle="Price (in ₹)", x="date", y="close",
                             title=f"{trend_level} chart of {stock_selector}")
        st.plotly_chart(fig, use_container_width=False)


# In[ ]:




