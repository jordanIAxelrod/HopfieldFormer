import os

import numpy as np

import HopfieldFormer as hf
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
import datasets
from transformers import AutoTokenizer, GPT2LMHeadModel
from tqdm import tqdm
import pandas as pd
def main(dataset, model):
    '''
    This function takes a dataset and trains
    :param dataset: a list of two datasets train and test
    :param model: HopfieldFormer
    :return: None
    '''
    try:
        print(os.getcwd())
        print(os.listdir('..'))

        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=1e-5)

        train_dataloader = torch.utils.data.DataLoader(dataset[0], batch_size=5, num_workers=7, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(dataset[1], batch_size=5, num_workers=7, shuffle=True)

        loss_func = nn.CrossEntropyLoss()
        lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=3e-4, epochs=1, steps_per_epoch=len(train_dataloader))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model.to(device)
        min_loss = np.inf
        train_loss = []
        for epoch in range(16):
            model.train()
            loop = tqdm(train_dataloader)
            for input in loop:

                input = tokenizer(input['text'], padding=True, return_tensors='pt', truncation=True)
                X, y = input.to(device), input['input_ids'][:, 1:]
                pred = model(**X).logits[:, :-1]
                loss = loss_func(pred.reshape(-1, 50257), y.reshape(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                perplexity = torch.exp(loss.detach())
                train_loss.append(perplexity.cpu())
                loop.set_postfix(perplexity=perplexity)
            train_loss = pd.DataFrame({"Train Loss": train_loss})
            train_loss.to_csv('../models/train_loss.csv')
            torch.save(model, '../models/BaseHopfieldNetwork.pth')
            model.eval()
            val_loss = 0
            for input in tqdm(test_dataloader):
                input = tokenizer(input['text'], padding=True, return_tensors='pt', truncation=True)
                X, y = input.to(device), input['input_ids'][:, 1:]
                pred = model(**X).logits[:, :-1]
                loss = loss_func(pred.reshape(-1, 50257), y.reshape(-1))
                val_loss += loss.detach().cpu()
            val_loss /= len(test_dataloader)
            if val_loss < min_loss:
                torch.save(model, '../models/BaseHopfieldNetwork.pth')
    except:
        torch.save(model, '../models/BaseHopfieldNetwork.pth')




if __name__ == '__main__':
    gpt = GPT2LMHeadModel.from_pretrained('gpt2')
    model = hf.HopfieldFormer(gpt, 4)
    openweb = datasets.load_dataset("openwebtext", split='train')
    generator = torch.Generator().manual_seed(1902821)
    openweb_train, openweb_test, openweb_val, _ = torch.utils.data.random_split(openweb, [.1, .02, .02, 1 - .1 - .04],
                                                                                generator=generator)
    main([openweb_train, openweb_test])

