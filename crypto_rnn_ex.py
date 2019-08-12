import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization #NOTE - CuDNNLSTM is deprecated ~ need to install tf.compat package
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint 

#recurrent neural network for predicting crypto prices
# tu will stand for time unit (minutes)

#use last SEQ_LEN tu of all data to try and predict next FUTURE_PERIOD_PREDICT tu of RATIO_TO_PREDICT
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"tutorial_1"

#if price is gonna go up return a 1 (should buy)
def classify(future, current):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    #scaling
    df=df.drop('future', 1) #we were only using this to get target, would give model data it should learn if left in
    for col in df.columns:
        if col!="target":
            df[col] = df[col].pct_change()#normalise all columns to std dist
            df.dropna(inplace=True) #drop missing values
            df[col]=preprocessing.scale(df[col].values)#scale from 0 to 1 ~ adjust for differences in data between coins
    df.dropna(inplace=True)
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN) #create a sort of 'rolling container' that updates to last SEQ_LEN values
    for i in df.values:
        prev_days.append([n for n in i[:-1]]) #append everything in last element of value
        if len(prev_days) == SEQ_LEN:   #once prev_days is populated
            sequential_data.append([np.array(prev_days), i[-1]]) #<Value, label>
    random.shuffle(sequential_data) #periodically shuffle to avoid getting stuck in the wrong local minima - 'averaging' gradient descent
    buys = []
    sells = []

    for seq, target in sequential_data:
        if target==0:
            sells.append([seq, target])
        elif target==1:
            buys.append([seq, target])
    random.shuffle(sells)   #probably unesesary but might as well
    random.shuffle(buys)

    #balance the data so we have a 50/50 split of buys/sells
    #otherwise machine will just optimise to buys and get stuck in a rut
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    sequential_data = buys+sells
    random.shuffle(sequential_data)

    x = []
    y = []
    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)
    
    return np.array(x),y    #<x as 2d array, y as 1d array>


#dataframe
main_df = pd.DataFrame()    #empty dataframe
#BCH-USD.csv  BTC-USD.csv  ETH-USD.csv  LTC-USD.csv
ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]

#join datasets on unix-time index, get close/volume values for each dataset w/ label
for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"
    df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])
    df.rename(columns={"close": f"{ratio}_close", "volume":f"{ratio}_volume"}, inplace=True)
    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]
    if len(main_df)==0:    #if main_df is empty, set to first thing, else add new df to main_df
        main_df=df
    else:
        main_df=main_df.join(df)

main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT) #add new column for price in FUTURE_PERIOD_PREDICT time units
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"])) #add buy/sell column for each interval


#get validation data set
#we test the model against this set to ensure that gains in accuracy follow for data that the model hasn't seen
#preventing overfitting

#index for last 5% of data
times = (main_df.index.values)
last_5pct = times[-int(len(times)*0.05)]#negative index in python just means start from right

validation_main_df = main_df[main_df.index >= last_5pct] # main_df where time 
main_df = main_df[main_df.index<last_5pct]  #subtract validation set from main set

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

model = Sequential()
#nodes in layer, description of input
#return_sequence=True means we will return whole string of output, false would be just last value.
    #use true when feeding to lstm layer so we preserve 'past' values in order to use lstm property
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

#no return sequence bc we're feeding to dense layer
model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

#output layer
model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_FINAL-{epoch:02d}-{val_acc:.3f}"    #unique file path
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

history = model.fit(train_x, train_y,
    batch_size=BATCH_SIZE, epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint])

try:
    model.save("/models/rnn_final.h5")
except:
    print("don't specfify file name")
    model.save("/models")
