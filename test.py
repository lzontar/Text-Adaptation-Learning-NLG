import os
import pandas as pd
from simpletransformers.language_generation import LanguageGenerationModel
import logging
logging.getLogger().setLevel(logging.CRITICAL)

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, WarmUp

_data_absolute_path = 'Data/'

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

# Function to first select topN tokens from the probability list and then based on the selected N word distribution
# get random token ID
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)


def generate_some_text(input_str, text_len = 250):

    cur_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0).long().to(device)

    model.eval()
    with torch.no_grad():

        for i in range(text_len):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(only one) batch and the last predicted embedding
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=10) #Randomly(from the given probability distribution) choose the next word from the top n words
            cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word

        output_list = list(cur_ids.squeeze().to('cpu').numpy())
        output_text = tokenizer.decode(output_list)
        print(output_text)

MODEL_EPOCH = 4

models_folder = "Model/"

model_path = os.path.join(models_folder, f"gpt2_medium_joker_{MODEL_EPOCH}.pt")
model.load_state_dict(torch.load(model_path))

jokes_output_file_path = f'generated_{MODEL_EPOCH}.jokes'

model.eval()
if os.path.exists(jokes_output_file_path):
    os.remove(jokes_output_file_path)

generate_some_text(" The Matrix is everywhere. It is all around us. Even now, in this very room. You can see it when you look out your window or when you turn on your television. You can feel it when you go to work... when you go to church... when you pay your taxes. It is the world that has been pulled over your eyes to blind you from the truth. ")
