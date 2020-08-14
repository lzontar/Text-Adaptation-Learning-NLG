import json
import os
import pandas as pd
from simpletransformers.language_generation import LanguageGenerationModel
import logging
logging.getLogger().setLevel(logging.CRITICAL)
from google.colab import files
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import google
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, WarmUp

_data_absolute_path = 'Data/'

device = 'cpu'
# if torch.cuda.is_available():
#     device = 'cuda'


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

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400

# generate_some_text(" The Matrix is everywhere. It is all around us. Even now, in this very room. You can see it when you look out your window or when you turn on your television. You can feel it when you go to work... when you go to church... when you pay your taxes. It is the world that has been pulled over your eyes to blind you from the truth. ")
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
# scheduler = WarmUp(optimizer, warmup_steps=WARMUP_STEPS)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

tmp_adapts_tens = None
models_folder = "Model/"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for epoch in range(EPOCHS):

    print(f"EPOCH {epoch} started" + '=' * 30)
    rootdir = _data_absolute_path + 'research_articles/document_parses/pdf_json'
    dataset = []
    for subdir, dirs, files in os.walk(rootdir):
        ix = 0
        for f in files:
            if ix == 0:
                break
            file = open(_data_absolute_path + 'research_articles/document_parses/pdf_json/' + f, 'r')
            json_file = json.loads(str(file.read()))
            research_article = ''
            for paragraph in json_file['body_text']:
                research_article = research_article + paragraph['text']

            file.close()
            dataset.append(research_article)
            ix = ix + 1
    dataset = ["I am the beast!"]
    adapt_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for idx, adapt in enumerate(adapt_loader):

        #################### "Fit as many adapt sequences into MAX_SEQ_LEN sequence as possible" logic start ####
        adapt_tens = torch.tensor(tokenizer.encode(adapt[0])).unsqueeze(0).to(device)
        # Skip sample from dataset if it is longer than MAX_SEQ_LEN
        if adapt_tens.size()[1] > MAX_SEQ_LEN:
            continue

        # The first adapt sequence in the sequence
        if not torch.is_tensor(tmp_adapts_tens):
            tmp_adapts_tens = adapt_tens
            continue
        else:
            # The next adapt does not fit in so we process the sequence and leave the last adapt
            # as the start for next sequence
            if tmp_adapts_tens.size()[1] + adapt_tens.size()[1] > MAX_SEQ_LEN:
                work_adapts_tens = tmp_adapts_tens
                tmp_adapts_tens = adapt_tens
            else:
                # Add the adapt to sequence, continue and try to add more
                tmp_adapts_tens = torch.cat([tmp_adapts_tens, adapt_tens[:, 1:]], dim=1)
                continue
        ################## Sequence ready, process it trough the model ##################

        outputs = model(work_adapts_tens, labels=work_adapts_tens)
        loss, logits = outputs[:2]
        loss.backward()
        sum_loss = sum_loss + loss.detach().data

        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0
            batch_count += 1
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0

    # Store the model after each epoch to compare the performance of them
    torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_medium_adaptr_{epoch}.pt"))
    google.colab.files.download(models_folder + "gpt2_medium_adaptr_" + epoch +".pt")

