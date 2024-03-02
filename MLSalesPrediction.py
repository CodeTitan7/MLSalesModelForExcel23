import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import math

datatrain = pd.read_csv('train.csv')

datatrain['date'] = pd.to_datetime(datatrain['date'], format='%d-%m-%Y')
datatrain.sort_values('date', inplace=True)
datatrain['year'] = datatrain['date'].dt.year
datatrain['month'] = datatrain['date'].dt.month
datatrain['day'] = datatrain['date'].dt.day
datatrain.drop(['date'], axis=1, inplace=True)

if datatrain.isnull().values.any():
    datatrain.dropna(inplace=True)

stateencodings = pd.get_dummies(datatrain['state'], prefix='state')
storeencodings = pd.get_dummies(datatrain['store'], prefix='store')
productencodings = pd.get_dummies(datatrain['product'], prefix='product')
datatrain = pd.concat([datatrain, stateencodings, storeencodings, productencodings],axis=1)
datatrain.drop(['state','store','product'], axis=1, inplace=True)

featurestrain = datatrain[['year','month','day','row_id'] + list(stateencodings.columns) + list(storeencodings.columns) + list(productencodings.columns)].values
labelstrain = datatrain['num_sold'].values

scaler = StandardScaler()
featurestrain_scaled = scaler.fit_transform(featurestrain)

model = keras.Sequential([
    layers.Dense(128, input_shape=[featurestrain_scaled.shape[1]]),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(1, activation='relu')
])

model.compile(optimizer='RMSprop', loss='mean_absolute_error')

def adjust_learning_rate(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        lr = lr * 0.9
    return lr

lr_scheduler_callback = LearningRateScheduler(adjust_learning_rate)

early_stopping_callback = EarlyStopping(monitor='val_loss',patience=5,
                                        restore_best_weights=True)

history = model.fit(featurestrain_scaled,
                    labelstrain,
                    epochs=200,
                    batch_size=64,
                    verbose=1,
                    callbacks=[lr_scheduler_callback, early_stopping_callback])

model.save('salesmodel.keras')

datatest = pd.read_csv('test.csv')

datatest['date'] = pd.to_datetime(datatest['date'], format='%d-%m-%Y')
datatest.sort_values('date', inplace=True)
datatest['year'] = datatest['date'].dt.year
datatest['month'] = datatest['date'].dt.month
datatest['day'] = datatest['date'].dt.day
datatest.drop(['date'], axis=1, inplace=True)

if datatest.isnull().values.any():
    datatest.dropna(inplace=True)

stateencodings_test = pd.get_dummies(datatest['state'], prefix='state')
storeencodings_test = pd.get_dummies(datatest['store'], prefix='store')
productencodings_test = pd.get_dummies(datatest['product'], prefix='product')
datatest = pd.concat(
    [datatest, stateencodings_test, storeencodings_test, productencodings_test],
    axis=1)
datatest.drop(['state','store','product'], axis=1, inplace=True)

featurestest = datatest[['year','month','day','row_id'] +
                          list(stateencodings_test.columns) +
                          list(storeencodings_test.columns) +
                          list(productencodings_test.columns)].values

featurestest_scaled = scaler.transform(featurestest)

predictions = model.predict(featurestest_scaled).flatten()

rounded_predictions = predictions.round()

outputdata = pd.DataFrame({
    'row_id': datatest['row_id'],
    'num_sold': rounded_predictions.astype(int)
})

outputdata.to_csv('samplesubmission.csv',
                   index=False,
                   columns=['row_id','num_sold'])
