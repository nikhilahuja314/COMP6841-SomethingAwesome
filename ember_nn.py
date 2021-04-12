# NB: import/exports not processed for memory reasons

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf 
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher

from functools import reduce

def process_headers(data):

    mlb = MultiLabelBinarizer()
    ohe = OneHotEncoder(sparse=False)
    le = LabelEncoder()

    label_sequence = data['label']
    sha_sequence = data['sha256']
    characteristics_sequence = data['header.coff.characteristics']
    machine_sequence = data['header.coff.machine']
    subsystem_sequence = data['header.optional.subsystem']
    dll_sequence = data['header.optional.dll_characteristics']
    magic_sequence = data['header.optional.magic']

    characteristics_dataframe = pd.DataFrame(mlb.fit_transform(characteristics_sequence), columns=mlb.classes_, index=characteristics_sequence.index)
    characteristics_dataframe = pd.merge(sha_sequence, characteristics_dataframe, left_index=True, right_index=True)
    characteristics_dataframe = pd.merge(label_sequence, characteristics_dataframe, left_index=True, right_index=True)

    machine_labels = le.fit_transform(machine_sequence)
    machine_labels = machine_labels.reshape(len(machine_labels), 1)
    machine_dataframe = pd.DataFrame(ohe.fit_transform(machine_labels))
    machine_dataframe = pd.merge(sha_sequence, machine_dataframe, left_index=True, right_index=True)
    machine_dataframe = pd.merge(label_sequence, machine_dataframe, left_index=True, right_index=True)

    subsystem_labels = le.fit_transform(subsystem_sequence)
    subsystem_labels = subsystem_labels.reshape(len(subsystem_labels), 1)
    subsystem_dataframe = pd.DataFrame(ohe.fit_transform(subsystem_labels))
    subsystem_dataframe = pd.merge(sha_sequence, subsystem_dataframe, left_index=True, right_index=True)
    subsystem_dataframe = pd.merge(label_sequence, subsystem_dataframe, left_index=True, right_index=True)

    dll_dataframe = pd.DataFrame(mlb.fit_transform(dll_sequence), columns=mlb.classes_, index=dll_sequence.index)
    dll_dataframe = pd.merge(sha_sequence, dll_dataframe, left_index=True, right_index=True)
    dll_dataframe = pd.merge(label_sequence, dll_dataframe, left_index=True, right_index=True)
    
    magic_labels = le.fit_transform(magic_sequence)
    magic_labels = magic_labels.reshape(len(magic_labels), 1)
    magic_dataframe = pd.DataFrame(ohe.fit_transform(magic_labels))
    magic_dataframe = pd.merge(sha_sequence, magic_dataframe, left_index=True, right_index=True)
    magic_dataframe = pd.merge(label_sequence, magic_dataframe, left_index=True, right_index=True)

    remaining_dataframe = pd.DataFrame(zip(
        data['header.optional.major_image_version'],
        data['header.optional.minor_image_version'],
        data['header.optional.major_linker_version'],
        data['header.optional.minor_linker_version'],
        data['header.optional.major_operating_system_version'],
        data['header.optional.minor_operating_system_version'],
        data['header.optional.major_subsystem_version'],
        data['header.optional.minor_subsystem_version'],
        data['header.optional.sizeof_code'],
        data['header.optional.sizeof_headers'],
        data['header.optional.sizeof_heap_commit']),
        columns=[
            data['header.optional.major_image_version'].name,
            data['header.optional.minor_image_version'].name,
            data['header.optional.major_linker_version'].name,
            data['header.optional.minor_linker_version'].name,
            data['header.optional.major_operating_system_version'].name,
            data['header.optional.minor_operating_system_version'].name,
            data['header.optional.major_subsystem_version'].name,
            data['header.optional.minor_subsystem_version'].name,
            data['header.optional.sizeof_code'].name,
            data['header.optional.sizeof_headers'].name,
            data['header.optional.sizeof_heap_commit'].name
        ])
    
    remaining_dataframe = pd.merge(sha_sequence, remaining_dataframe, left_index=True, right_index=True)
    remaining_dataframe = pd.merge(label_sequence, remaining_dataframe, left_index=True, right_index=True)

    frames = [characteristics_dataframe, machine_dataframe, subsystem_dataframe, dll_dataframe, magic_dataframe, remaining_dataframe]

    headers_dataframe = reduce(lambda left,right: pd.merge(left,right,on='sha256'), frames)

    return headers_dataframe

def process_general(data):

    sha_seq = data['sha256']
    label_sequence = data['label']
    general_dataframe =  pd.DataFrame(zip(
        data['general.size'],
        data['general.vsize'],
        data['general.has_debug'],
        data['general.exports'],
        data['general.imports'],
        data['general.has_relocations'],
        data['general.has_resources'],
        data['general.has_signature'],
        data['general.has_tls'],
        data['general.symbols']
    ),
    columns=[
        data['general.size'].name,
        data['general.vsize'].name,
        data['general.has_debug'].name,
        data['general.exports'].name,
        data['general.imports'].name,
        data['general.has_relocations'].name,
        data['general.has_resources'].name,
        data['general.has_signature'].name,
        data['general.has_tls'].name,
        data['general.symbols'].name
    ])

    general_dataframe = pd.merge(sha_seq, general_dataframe, left_index=True, right_index=True)
    general_dataframe = pd.merge(label_sequence, general_dataframe, left_index=True, right_index=True)
    return general_dataframe

def process_strings(data):

    sha_seq = data['sha256']
    label_sequence = data['label']
    strings_dataframe = pd.DataFrame(zip(
        data['strings.numstrings'],
        data['strings.avlength'],
        data['strings.entropy'],
        data['strings.paths'],
        data['strings.urls'],
        data['strings.registry'],
        data['strings.MZ']
    ),
    columns=[
        data['strings.numstrings'].name,
        data['strings.avlength'].name,
        data['strings.entropy'].name,
        data['strings.paths'].name,
        data['strings.urls'].name,
        data['strings.registry'].name,
        data['strings.MZ'].name
    ])

    strings_dataframe = pd.merge(sha_seq, strings_dataframe, left_index=True, right_index=True)
    strings_dataframe = pd.merge(label_sequence, strings_dataframe, left_index=True, right_index=True)
    return strings_dataframe

def process_imports(data):
    mlb = MultiLabelBinarizer()
    
    imports_seq = data['imports']
    sha_seq = data['sha256']
    label_sequence = data['label']

    imports_dataframe = pd.DataFrame(mlb.fit_transform(imports_seq), columns=mlb.classes_, index=imports_seq.index)
    imports_dataframe = pd.merge(sha_seq, imports_dataframe, left_index=True, right_index=True)
    imports_dataframe = pd.merge(label_sequence, imports_dataframe, left_index=True, right_index=True)

    return imports_dataframe
  
def process_exports(data):
    pass

def process_sections(data):
    
    mlb = MultiLabelBinarizer()

    
    section_props_seq = data['props']
    sha_seq = data['sha256']
    label_sequence = data['label']

    section_entropy_dataframe = pd.DataFrame(data=data['entropy'])
    section_entropy_dataframe = pd.merge(sha_seq, section_entropy_dataframe, left_index=True, right_index=True)
    section_entropy_dataframe = pd.merge(label_sequence, section_entropy_dataframe, left_index=True, right_index=True)

    section_props_dataframe = pd.DataFrame(mlb.fit_transform(section_props_seq), columns=mlb.classes_, index=section_props_seq.index)
    section_props_dataframe = pd.merge(sha_seq, section_props_dataframe, left_index=True, right_index=True)
    section_props_dataframe = pd.merge(label_sequence, section_props_dataframe, left_index=True, right_index=True)

    frames = [section_entropy_dataframe, section_props_dataframe]
    section_dataframe = reduce(lambda left,right: pd.merge(left,right,on='sha256'), frames)

    return section_dataframe


with open('ember/hmm.json') as file:
    contents = file.readlines()

stripped_contents = [json.loads(line.strip()) for line in contents]

for obj in stripped_contents:
    del obj['histogram']
    del obj['byteentropy']
    del obj['appeared']
    del obj['imports']

stripped_contents = [item for item in stripped_contents if item['label'] != -1]

dataframe = pd.json_normalize(stripped_contents)
section_frame = pd.json_normalize(stripped_contents, ['section', 'sections'], meta=['sha256', 'label'])

headers = process_headers(dataframe)
strings = process_strings(dataframe)
general = process_general(dataframe)
sections = process_sections(section_frame)
#imports = process_imports(dataframe)
#exports = process_exports(dataframe)

dataframes = [headers, strings, general, sections]
main_dataframe = reduce(lambda left, right: pd.merge(left, right, on='sha256'), dataframes)

value_array = main_dataframe.values

feature_data = value_array[0:, 2:117]
labels = value_array[0:, 0]

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_feature_data = scaler.fit_transform(feature_data)

training_features, testing_features, training_labels, testing_labels = train_test_split(
    rescaled_feature_data, labels, test_size=0.3
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1000, input_dim=114, activation="relu"),
    tf.keras.layers.Dense(500, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="tanh")
])

optimizer = tf.keras.optimizers.Adam(lr=0.0000001)

model.compile(
    optimizer=optimizer,
    loss="mean_squared_error",
    metrics=["accuracy"]
)

model.fit(np.asarray(training_features).astype(np.float32), np.asarray(training_labels).astype(np.float32), epochs=10, batch_size=5)
model.evaluate(np.asarray(testing_features).astype(np.float32), np.asarray(testing_labels).astype(np.float32), verbose=2)