!pip install transformers
!pip install torch
!pip install torchinfo

!pip install xlrd==1.2.0
!pip install pandas==1.2.0

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import transformers
import torchinfo
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pandas import DataFrame, Series
from pandas.core.generic import NDFrame
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torchinfo import summary
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
import xlrd
from keras_preprocessing.sequence import pad_sequences
from tqdm import trange 
from torch.nn import CrossEntropyLoss

# specify GPU device
device = torch.device("cuda")


# import database
df = pd.read_excel(r'/content/TestBot.xls')
df = df.fillna("")
df.head()
df['label'].value_counts()
num_classes = df['label'].nunique()

# encode the labels (classes)
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
# Check class distribution
df['label'].value_counts(normalize=True)

# prepping our data
train_labels = df['label']
train_labels = train_labels.tolist()
train_text = pd.Series(df['text'], dtype="string")
train_list0 = train_text.tolist()
train_list = []
for i in train_list0:
    i = i.replace(u'\xa0', u' ')
    train_list.append(i)
train_list = ["[CLS] " + query + " [SEP]" for query in train_list]
# checking our data
print(train_list)
all(isinstance(n, str) for n in train_list)
print (train_labels)

# installing and importing the model/tokenizer
!pip install datasets evaluate transformers[sentencepiece]
# importing camembert tokenizer and camembertforsequenceclassification model
from transformers import CamembertTokenizer, CamembertForSequenceClassification
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
bert = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=num_classes)

# Get lenght of all the messages in the train set
seq_len = [len(i.split()) for i in train_list]
print(max(seq_len))
# defining constants
epochs = 10
MAX_LEN = 64
batch_size = 7

# using tokenizer to convert sentences into tokenizer
input_ids  = [tokenizer.encode(query, add_special_tokens=True, max_length=MAX_LEN, truncation=True) for query in train_text]
# padding our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Creating attention masks
attention_masks = []
# Creating a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]  
    attention_masks.append(seq_mask)
# checking
print (input_ids)
print (attention_masks)

# Use train_test_split to split our data into train and validation sets for training #crossvalidation
train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = train_test_split(input_ids, train_labels, attention_masks, random_state=43, test_size=0.2)

# Convert all of our data into torch tensors, the required datatype for our model
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
# Checking
train_inputs.size()
train_labels.size()
train_masks.size()

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory
train_data = TensorDataset(train_inputs, train_masks, train_labels)
# sampler for sampling the Data during training
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
# for validation set
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Defining the parameters and metrics to optimize
param_optimizer = list(bert.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=10e-8)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
# push the model to GPU
bert.cuda()
summary(bert)

# TRAINING

# Store our loss and accuracy for plotting if we want to visualize training evolution per epochs after the training process
train_loss_set = []
# empty list to save model predictions
total_preds=[]
CUDA_LAUNCH_BLOCKING = 1
# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):  
    # Tracking variables for training
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # Train the model
    bert.train()
    for step, batch in enumerate(train_dataloader):
        # Add batch to device CPU or GPU
        batch = [t.to(device) for t in batch]
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        outputs = bert(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        # Add it to train loss list
        train_loss_set.append(loss.item())   
        # Backward pass
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # model outputs are stored on GPU. So, push it to CPU
        outputs.logits=outputs.logits.detach().cpu().numpy()
        # append the model predictions #outputs
        total_preds.append(outputs.logits)

    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    # Tracking variables for validation
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    
    # Validation of the model
    bert.eval()
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to device CPU or GPU
        batch = [t.to(device) for t in batch]
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            output = bert(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = output.logits
            # Move logits and labels to CPU if GPU is used
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    
# Get prediction with test Data
def get_prediction(str):
    test_text = [str]
    test_text = ["[CLS] " + query + " [SEP]" for query in test_text]
    bert.eval()
    
    tokens_test_data = tokenizer(
    test_text,
    max_length = MAX_LEN,
    padding = 'max_length',
    truncation = True,
    return_token_type_ids = False)

    test_seq = torch.tensor(tokens_test_data['input_ids'])
    test_mask = torch.tensor(tokens_test_data['attention_mask'])

    preds = None

    with torch.no_grad():
      preds = bert(test_seq.to(device), test_mask.to(device))
      preds = preds.logits.detach().cpu().numpy()
      preds = np.argmax(preds, axis = 1)
      print('Intent Identified: ', le.inverse_transform(preds)[0])
    
    return le.inverse_transform(preds)[0]
    
# Testing model predictions
get_prediction('Quand puis-je récupérer mon colis?')
