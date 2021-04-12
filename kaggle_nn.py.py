import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_parameters(row: list):
    
    row_array = np.array(row[2:55])
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_row = scaler.fit_transform(row_array)

    return rescaled_row

def get_label(row: list):
    success = False
    while not success:
        try:
            retval = int(row[-1])
            success = True
            return retval
        except ValueError:
            row.pop()

headers = [
    'ID', 
    'md5', 
    'Machine', 
    'SizeOfOptionalHeader', 
    'Characteristics', 
    'MajorLinkerVersion', 
    'MinorLinkerVersion', 
    'SizeOfCode', 
    'SizeOfInitializedData', 
    'SizeOfUninitializedData', 
    'AddressOfEntryPoint', 
    'BaseOfCode', 
    'BaseOfData', 
    'ImageBase', 
    'SectionAlignment', 
    'FileAlignment', 
    'MajorOperatingSystemVersion', 
    'MinorOperatingSystemVersion', 
    'MajorImageVersion', 
    'MinorImageVersion', 
    'MajorSubsystemVersion', 
    'MinorSubsystemVersion', 
    'SizeOfImage', 
    'SizeOfHeaders', 
    'CheckSum', 
    'Subsystem', 
    'DllCharacteristics', 
    'SizeOfStackReserve', 
    'SizeOfStackCommit', 
    'SizeOfHeapReserve', 
    'SizeOfHeapCommit', 
    'LoaderFlags', 
    'NumberOfRvaAndSizes', 
    'SectionsNb', 
    'SectionsMeanEntropy', 
    'SectionsMinEntropy', 
    'SectionsMaxEntropy', 
    'SectionsMeanRawsize', 
    'SectionsMinRawsize', 
    'SectionMaxRawsize', 
    'SectionsMeanVirtualsize', 'SectionsMinVirtualsize', 'SectionMaxVirtualsize', 'ImportsNbDLL', 'ImportsNb', 'ImportsNbOrdinal', 'ExportNb', 'ResourcesNb', 'ResourcesMeanEntropy', 'ResourcesMinEntropy', 'ResourcesMaxEntropy', 'ResourcesMeanSize', 'ResourcesMinSize', 'ResourcesMaxSize', 'LoadConfigurationSize', 'VersionInformationSize', 'legitimate']

dataframe = pd.read_csv("Kaggle-data.csv", names=headers)
array = dataframe.values

X = array[1:, 1:56]
Y = array[1:, -2:-1]

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

X_training, X_testing, y_training, y_testing = train_test_split(
    rescaledX, Y, test_size=0.3
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1000, input_dim=55, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1, activation="tanh")
])

optimizer = tf.keras.optimizers.Adam(lr=0.0000001, clipvalue=1.0)

model.compile(
    optimizer=optimizer,
    loss="mean_squared_error",
    metrics=["accuracy"]
)

model.fit(np.asarray(X_training).astype(np.float32), np.asarray(y_training).astype(np.float32), epochs=10, batch_size=5)
model.evaluate(np.asarray(X_testing).astype(np.float32), np.asarray(y_testing).astype(np.float32), verbose=2)


