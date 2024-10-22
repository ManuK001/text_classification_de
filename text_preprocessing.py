import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib.pyplot as plt
import copy

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


# from tqdm import tqdm
# from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
# from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

# from datasets import Dataset, DatasetDict
# from transformers import DataCollatorWithPadding

import re
batch_size = 32
model_checkpoint = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,use_fast=True)

def remove_tags(text):
  TAG_RE = re.compile(r'<[^>]+>')
  if type(text) == str :
    return TAG_RE.sub('', text)
  
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


def label_encoder_fit_and_transform(data, column):
    """Apply operation on training data and return preprocessor"""
       # Reset dataframe index to start from 0
    data.reset_index(drop=True, inplace=True)

    # Input dataframe
    processed_data = pd.DataFrame()

    # Extract target column
    #for column in columns_list:
    processed_data[column] = data.pop(column)
    print(processed_data.columns)
    #log_progress('label encoding targeted columns', level=1)
    #processed_data = label_encoder_fit_and_transform_columns(processed_data, column)
    encoder = LabelEncoder()
    processed_data = encoder.fit_transform(processed_data)
    # Copy back encoded columns
    #for column in columns:
    data[column] = processed_data

    return data

def generate_train_test_val_data(input_file_path):
    dataset = pd.read_excel(input_file_path)

    print(dataset.shape)
    print("The column headers :")
    print(list(dataset.columns.values))

    dataset['Text'] = dataset['Text'].apply(preprocess_text)

    #from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    print(dataset.shape)
    print("The column headers :")
    print(list(dataset.columns.values))
    #encoder = LabelEncoder()
    #dataset = encoder.fit_transform(dataset['Sub_Classification'])
    dataset = label_encoder_fit_and_transform(dataset, 'Scope')

    print(len(dataset.Scope.unique()))

    # Let's say we want to split the data in 80:10:10 for train:valid:test dataset
    train_size = 0.8
    valid_size=0.1

    train_index = int(len(dataset)*train_size)

    df_train = dataset[0:train_index]
    df_rem = dataset[train_index:]

    valid_index = int(len(dataset)*valid_size)

    df_valid = dataset[train_index:train_index+valid_index]
    df_test = dataset[train_index+valid_index:]

    print(df_train.shape)
    print(df_valid.shape)
    print(df_test.shape)

    df_train.to_csv("Enhanced_Synthetic_Data_Train.csv", index=False)
    df_valid.to_csv("Enhanced_Synthetic_Data_Val.csv", index=False)
    df_test.to_csv("Enhanced_Synthetic_Data_Test.csv", index=False)

def preprocess_function(records):
        return tokenizer(records['Text'], truncation=True, return_token_type_ids=True, max_length = 50)

def get_encoded_train_validation_tf_data(dataset):
    encoded_dataset = dataset.map(preprocess_function, batched=True, )

    pre_tokenizer_columns = set(dataset["train"].features)

    tokenizer_columns = list(set(encoded_dataset["train"].features) - pre_tokenizer_columns)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf",)

    tf_train_dataset = encoded_dataset["train"].to_tf_dataset(
        columns=tokenizer_columns,
        label_cols=["Scope"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    tf_validation_dataset = encoded_dataset["valid"].to_tf_dataset(
        columns=tokenizer_columns,
        label_cols=["Scope"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return tf_train_dataset, tf_validation_dataset