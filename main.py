# import libraries
import streamlit as st #streamlit using for build web app
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from keras.models import Sequential 
from keras.layers import Dense, LSTM 
from sklearn.preprocessing import MinMaxScaler #using for scale data
from sklearn.metrics import mean_squared_error #using for calculate MSE
from dateutil.relativedelta import relativedelta 
import tensorflow as tf
import time #using for calculate time



st.set_page_config(
    page_title="TLCN - FFNN - Time Series",
    page_icon="🕹",
)


st.markdown("<h1 style='text-align: center; color: while;'>Forecasting App Using Feed Forward Neural Network </h1>", unsafe_allow_html=True)

# Define a list of datasets
datasets = ['AMAZON.csv', 'GOOGLE.csv', 'APPLE.csv','Manhattan_NewYork_2010_24-2023.csv','Weather_dataset.csv']

# Add some text widgets to make the select box appear smaller
st.text("\n\n\n")

# Create a select box for the datasets
dataset_name = st.selectbox('Select a dataset', datasets)

# Add some text widgets to make the select box appear smaller
st.text("\n\n\n")

# Load the selected dataset
data = pd.read_csv(f"./data/{dataset_name}")

# Show data
st.write(data)

st.write(f"Number of records: {len(data)}")

st.markdown("<h3 style='text-align: center; color: while;'>Explore of the dataset </h3>", unsafe_allow_html=True)

# EDA for stocks (amzon, google, apple)
def eda_stocks(dataset):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    plt.clf() 
    data['Close'].plot(color = 'blue')
    plt.title('Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    st.pyplot(plt.gcf())
    plt.close()

# EDA for weather (manhattan)    
def eda_weather_Ny(dataset):
    data['DATE'] = pd.to_datetime(data['DATE'])
    data.set_index('DATE', inplace=True)
    plt.clf() 
    data['TMAX'].plot(color = 'blue')
    plt.title('Max Temperature Over Time')
    plt.xlabel('DATE')
    plt.ylabel('Temperature')
    st.pyplot(plt.gcf())
    plt.close()   

# EDA for weather (WES)
def eda_weather_WES(dataset):
    data['Date Time'] = pd.to_datetime(data['Date Time'])
    data.set_index('Date Time', inplace=True)
    plt.clf() 
    data['Temperature'].plot(color = 'blue')
    plt.title('Temperature Over Time')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    st.pyplot(plt.gcf())
    plt.close()

# Scale data       
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data)

# Split data => train(0.8), test(0.2)
def split_data(data, split_ratio=0.8):
    train_size = int(len(data) * split_ratio)
    train, test = data[:train_size], data[train_size:]
    return train, test

# Model FFNN, LSTM for amazon
def model_amazon():
    seq_size = 12
    model = Sequential()
    model.add(Dense(5, input_dim=seq_size, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model.load_weights('FFNN_Model_AMAZON.h5')
    return model

def model_LSTM_Amazon():
    seq_size = 12
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(5, return_sequences=False, input_shape= (seq_size, 1)))
    model_LSTM.add(Dense(1))
    model_LSTM.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model_LSTM.load_weights('LSTM_Model_AMAZON.h5')
    return model_LSTM

# Model FFNN, LSTM for google
def model_google():
    seq_size = 11
    model = Sequential()
    model.add(Dense(4, input_dim=seq_size, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model.load_weights('FFNN_Model_GOOGLE.h5')
    return model

def model_LSTM_Google():
    seq_size = 11
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(4, return_sequences=False, input_shape= (seq_size, 1)))
    model_LSTM.add(Dense(1))
    model_LSTM.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model_LSTM.load_weights('LSTM_Model_GOOGLE.h5')
    return model_LSTM



# Model FFNN, LSTM for apple
def model_apple():
    seq_size = 12
    model = Sequential()
    model.add(Dense(5, input_dim=seq_size, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model.load_weights('FFNN_Model_APPLE.h5')
    return model

def model_LSTM_Apple():
    seq_size = 12
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(5, return_sequences=False, input_shape= (seq_size, 1)))
    model_LSTM.add(Dense(1))
    model_LSTM.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model_LSTM.load_weights('LSTM_Model_APPLE.h5')
    return model_LSTM

# Model FFNN, LSTM for manhattan
def model_manhattan():
    seq_size = 85
    model = Sequential()
    model.add(Dense(9, input_dim=seq_size, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model.load_weights('FFNN_Model_Temperature_NY.h5')
    return model  
def model_LSTM_manhattan():
    seq_size = 85
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(9, return_sequences=False, input_shape= (seq_size, 1)))
    model_LSTM.add(Dense(1))
    model_LSTM.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model_LSTM.load_weights('LSTM_Model_TempNY.h5')
    return model_LSTM

# Model FFNN, LSTM for weather
def model_weather_WES():
    seq_size = 90
    model = Sequential()
    model.add(Dense(16, input_dim=seq_size, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model.load_weights('FFNN_Model_Temperature.h5')
    return model     

def model_LSTM_weather_WES():
    seq_size = 90
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(16, return_sequences=False, input_shape= (seq_size, 1)))
    model_LSTM.add(Dense(1))
    model_LSTM.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model_LSTM.load_weights('LSTM_Model_TempWES.h5')
    return model_LSTM


# Convert an array of values into a dataset matrix
def to_sequences(dataset,timestep , seq_size=1): 
    x = []
    y = []

    for i in range(0,len(dataset)-seq_size-1,timestep):
        #print(i)
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])

    return np.array(x),np.array(y)

# Forecasting on test sets
def eda_model(data):
    if dataset_name == 'AMAZON.csv':
        train, test = split_data(scale_data(data['Close'].values.reshape(-1,1)))
        model = model_amazon()
        x,y = to_sequences(test,1,12)
    elif dataset_name == 'GOOGLE.csv':
        train, test = split_data(scale_data(data['Close'].values.reshape(-1,1)))
        model = model_google()
        x,y = to_sequences(test,1,11)
    elif dataset_name == 'APPLE.csv':
        train, test = split_data(scale_data(data['Close'].values.reshape(-1,1)))
        model = model_apple()
        x,y = to_sequences(test,1,12)
    elif dataset_name == 'Manhattan_NewYork_2010_24-2023.csv':
        data.ffill(inplace=True)
        train, test = split_data(scale_data(data['TMAX'].values.reshape(-1,1)))
        model = model_manhattan()
        x,y = to_sequences(test,1,85)
    else:
        train, test = split_data(scale_data(data['Temperature'].values.reshape(-1,1)))
        model = model_weather_WES()
        x,y = to_sequences(test,1,90)
    with st.spinner('The prediction process is ongoing...'):   
        testPredict = model.predict(x)
        testScore = mean_squared_error(y, testPredict)
        
        plt.clf() 
        plt.plot(y,label="Actual value")
        plt.plot(testPredict,label="Predicted value")
        plt.legend()
        st.pyplot(plt.gcf())
        st.write(f"Test Score: {testScore}")

# get date for forecasting by sub-time series
def get_date_forcecast(data): 
    data1 = data.copy(deep=True)
    data1.reset_index(inplace=True)
    if dataset_name == 'AMAZON.csv' or dataset_name == 'GOOGLE.csv' or dataset_name == 'APPLE.csv':
        data1['Date'] = pd.to_datetime(data1['Date'])
        train_size =  int(len(data1) * 0.8)
        if dataset_name == 'AMAZON.csv' or dataset_name == 'APPLE.csv':
            train, test = data1[:train_size], data1[train_size+12:]
        else:
            train, test = data1[:train_size], data1[train_size+11:]
        cols1,_ = st.columns((1,2))
        format = 'DD/MM/YYYY' 
        start_date = test['Date'].min() 
        end_date = test['Date'].max()
        date_range = cols1.date_input('Select date range', (start_date, end_date), format=format)        
        
        df1 = data1[data1['Date'].dt.date == date_range[0]]
        df2 = data1[data1['Date'].dt.date == date_range[1]]
        index_1 = df1.index.values[0] if not df1.empty else None
        index_2 = df2.index.values[0]  if not df2.empty else None
        
    elif dataset_name == 'Manhattan_NewYork_2010_24-2023.csv':
        data1['DATE'] = pd.to_datetime(data1['DATE'])
        train_size =  int(len(data1) * 0.8)
        train, test = data1[:train_size], data1[train_size+85:]
        cols1,_ = st.columns((1,2))
        format = 'DD/MM/YYYY' 
        start_date = test['DATE'].min() 
        end_date = test['DATE'].max()    
        date_range = cols1.date_input('Select date range', (start_date, end_date), format=format)        
        
        df1 = data1[data1['DATE'].dt.date == date_range[0]]
        df2 = data1[data1['DATE'].dt.date == date_range[1]]
        index_1 = df1.index.values[0] if not df1.empty else None
        index_2 = df2.index.values[0] if not df2.empty else None 
    else:
        data1['Date Time'] = pd.to_datetime(data1['Date Time'])
        train_size =  int(len(data1) * 0.8)
        train, test = data1[:train_size], data1[train_size+90:]
        cols1,_ = st.columns((1,2))
        format = 'DD/MM/YYYY' 
        start_date = test['Date Time'].min() 
        end_date = test['Date Time'].max()
        date_range = cols1.date_input('Select date range', (start_date, end_date), format=format)        
        
        df1 = data1[data1['Date Time'].dt.date == date_range[0]]
        df2 = data1[data1['Date Time'].dt.date == date_range[1]]
        index_1 = df1.index.values[0] if not df1.empty else None
        index_2 = df2.index.values[0] if not df2.empty else None
         
    st.write(f"Start date: {date_range[0]} -- End date: {date_range[1]} -- Number of days selected: {(date_range[1]- date_range[0]).days}")
    st.write(index_1,index_2)

    return index_1,index_2

# Forecasting by sub-time series
def eda_child_timeseries(data):
    if dataset_name == 'AMAZON.csv':
        data_set = scale_data(data['Close'].values.reshape(-1,1))        
        model = model_amazon()
        model_LSTM = model_LSTM_Amazon()
        index_1, index_2 = get_date_forcecast(data)
        st.write(data_set[index_1:index_2])
        x,y = to_sequences(data_set[index_1-12:index_2+1],1,12)
        
    elif dataset_name == 'GOOGLE.csv':
        data_set= scale_data(data['Close'].values.reshape(-1,1))
        model = model_google()
        model_LSTM = model_LSTM_Google()
        index_1, index_2 = get_date_forcecast(data)
        st.write(data_set[index_1:index_2])
        x,y = to_sequences(data_set[index_1-11:index_2+1],1,11)
    elif dataset_name == 'APPLE.csv':
        data_set = scale_data(data['Close'].values.reshape(-1,1))
        model = model_apple()
        model_LSTM = model_LSTM_Apple()
        index_1, index_2 = get_date_forcecast(data)
        st.write(data_set[index_1:index_2])
        x,y = to_sequences(data_set[index_1-12:index_2+1],1,12)
    elif dataset_name == 'Manhattan_NewYork_2010_24-2023.csv':
        data.ffill(inplace=True)
        data_set= scale_data(data['TMAX'].values.reshape(-1,1))
        model = model_manhattan()
        model_LSTM = model_LSTM_manhattan()
        index_1, index_2 = get_date_forcecast(data)
        st.write(data_set[index_1:index_2])
        x,y = to_sequences(data_set[index_1-85:index_2+1],1,85)
    else:
        data_set= scale_data(data['Temperature'].values.reshape(-1,1))
        model = model_weather_WES()
        model_LSTM = model_LSTM_weather_WES()
        index_1, index_2 = get_date_forcecast(data)
        st.write(data_set[index_1:index_2])
        x,y = to_sequences(data_set[index_1-90:index_2+1],1,90)
        
    sum_value = len(data_set[index_1:index_2])
    if st.checkbox('Compare with LSTM'):
        st.write('You have chosen to compare FFNN with LSTM')
        with st.spinner('The prediction process is ongoing...'):
            
            start_FFNN = time.time()  
            testPredict = model.predict(x)
            testScore = mean_squared_error(y, testPredict)  
            end_FFNN = time.time()
            
            start_LSTM = time.time()
            testPredict_LSTM = model_LSTM.predict(x)
            testScore_LSTM = mean_squared_error(y, testPredict_LSTM)
            end_LSTM = time.time()
            col1, col2 = st.columns(2)
            # Vẽ biểu đồ FFNN trên cột thứ nhất
            
            if sum_value < 4:
                with col1:
                    testPredict = model.predict(x)
                    testScore = mean_squared_error(y, testPredict)
                    plt.clf() 
                    plt.plot(y, 'o', label="Actual value")
                    plt.plot(testPredict, 'o', label="Predicted value")
                    plt.legend()
                    plt.xticks(np.arange(0, len(y), 1))
                    st.pyplot(plt.gcf())
                    plt.close()
                    st.write(f"Test Score: {testScore}")
                    st.write(f"Time predict: {end_FFNN - start_FFNN}")
                with col2:
                    testPredict_LSTM = model_LSTM.predict(x)
                    testScore_LSTM = mean_squared_error(y, testPredict_LSTM)
                    plt.clf() 
                    plt.plot(y, 'o', label="Actual value")
                    plt.plot(testPredict_LSTM, 'o', label="Predicted value")
                    plt.legend()
                    plt.xticks(np.arange(0, len(y), 1))
                    st.pyplot(plt.gcf())
                    plt.close()
                    st.write(f"Test Score: {testScore_LSTM}")
                    st.write(f"Time predict: {end_LSTM - start_LSTM}")
            else:
                with col1:
                    plt.clf() 
                    plt.plot(y,label="Actual value")
                    plt.plot(testPredict,label="Predicted value")
                    plt.title('FFNN')
                    plt.legend()
                    plt.xticks(np.arange(0, len(y), 1))
                    st.pyplot(plt.gcf())
                    plt.close()
                    st.write(f"Test Score: {testScore}")
                    st.write(f"Time predict: {end_FFNN - start_FFNN}")
                # Vẽ biểu đồ LSTM trên cột thứ hai
                with col2:
                    plt.clf() 
                    plt.plot(y,label="Actual value") 
                    plt.plot(testPredict_LSTM,label="Predicted value")
                    plt.title('LSTM')
                    plt.legend()
                    plt.xticks(np.arange(0, len(y), 1))
                    st.pyplot(plt.gcf())
                    plt.close()
                    st.write(f"Test Score: {testScore_LSTM}")
                    st.write(f"Time predict: {end_LSTM - start_LSTM}")
        
    else:
        if sum_value < 4:
            with st.spinner('The prediction process is ongoing...'):
                start_FFNN_1 = time.time()
                testPredict = model.predict(x)
                testScore = mean_squared_error(y, testPredict)
                end_FFNN_1 = time.time()
                plt.clf() 
                plt.plot(y, 'o', label="Actual value")
                plt.plot(testPredict, 'o', label="Predicted value")
                plt.legend()
                plt.xticks(np.arange(0, len(y), 1))
                st.pyplot(plt.gcf())
                plt.close()
                st.write(f"Test Score: {testScore}")
                st.write(f"Time predict: {end_FFNN_1 - start_FFNN_1}")
        else:
            with st.spinner('The prediction process is ongoing...'):
                start_FFNN_2 = time.time()   
                testPredict = model.predict(x)
                testScore = mean_squared_error(y, testPredict)
                end_FFNN_2 = time.time()
                plt.clf() 
                plt.plot(y,label="Actual value")
                plt.plot(testPredict,label="Predicted value")
                plt.legend()
                plt.xticks(np.arange(0, len(y), 1))
                st.pyplot(plt.gcf())
                plt.close()
                st.write(f"Test Score: {testScore}")
                st.write(f"Time predict: {end_FFNN_2 - start_FFNN_2}")


# control flow 
if dataset_name == 'AMAZON.csv': 
    eda_stocks(data)
    st.markdown("<h3 style='text-align: center; color: while;'>Forecasting on test sets </h3>", unsafe_allow_html=True)
    eda_model(data)
    st.markdown("<h3 style='text-align: center; color: while;'>Forecasting by sub-time series</h3>", unsafe_allow_html=True)
    eda_child_timeseries(data)
elif dataset_name == 'GOOGLE.csv':
    eda_stocks(data)
    st.markdown("<h3 style='text-align: center; color: while;'>Forecasting on test sets </h3>", unsafe_allow_html=True)
    eda_model(data)
    eda_child_timeseries(data)
elif dataset_name == 'APPLE.csv':
    eda_stocks(data)
    st.markdown("<h3 style='text-align: center; color: while;'>Forecasting on test sets </h3>", unsafe_allow_html=True)    
    eda_model(data)
    eda_child_timeseries(data)
elif dataset_name == 'Manhattan_NewYork_2010_24-2023.csv':
    eda_weather_Ny(data)
    st.markdown("<h3 style='text-align: center; color: while;'>Forecasting on test sets </h3>", unsafe_allow_html=True)    
    eda_model(data)
    eda_child_timeseries(data)
else:
    eda_weather_WES(data)
    st.markdown("<h3 style='text-align: center; color: while;'>Forecasting on test sets </h3>", unsafe_allow_html=True)    
    eda_model(data)
    eda_child_timeseries(data)

