########### Necessary imports ###########

import random

import numpy as np

from sklearn.model_selection import train_test_split

from datasets import load_dataset

import torch

from tqdm import tqdm

from get_dataset import prepare_data
from get_model import build_model
from train import trainer

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

DATASET_NAME = "virattt/financial-qa-10K"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_NAME = 'Instruct-Fine-tuned'
PAD_TOKEN = "|pad|>"

df = prepare_data(name = DATASET_NAME, tokenizer = MODEL_NAME).get_data()

# split the data-> train/val/test
train, test = train_test_split(df, test_size = 0.2, random_state = seed, shuffle = False)
train, val = train_test_split(train, test_size = 0.2/0.8, random_state = seed, shuffle = True)

# save to json
train.to_json('train.json', orient = 'records', lines = True)
val.to_json('val.json', orient = 'records', lines = True)
test.to_json('test.json', orient = 'records', lines = True)

dataset = load_dataset('json', data_files = {'train' : 'train.json', 'val' : 'val.json', 'test' : 'test.json'})

# view a sample from train
print(f"**Sample from the train set**\n{dataset['train'][0]['text']}")

model = build_model(name = MODEL_NAME, pad_token = PAD_TOKEN).get_lora_model(rank = 32, alpha = 16, dropout = 0.03)

trainer(seed = seed, name = MODEL_NAME, max_seq_length = 512, model = model, dataset = dataset, pad_token = PAD_TOKEN).train(saved_name = SAVE_NAME)
