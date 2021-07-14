#!/usr/bin/env python
# coding: utf-8


import requests
import json

import numpy as np

from time import sleep
import re
import torch
import os
from scipy.signal import stft
import random
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import time
import torch.utils.data
import random
from torch.nn.utils.rnn import pad_sequence
from transformers import *
import argparse

import copy




model_dir = "ru_conversational_cased_L-12_H-768_A-12_pt"
model = copy.deepcopy(BertForMaskedLM.from_pretrained(model_dir))
tokenizer = BertTokenizer.from_pretrained(model_dir)


def load_group(file_obj):
    
    id_group = int.from_bytes(file_obj.read(4), byteorder='little', signed=True)
    text_sz = int.from_bytes(file_obj.read(4), byteorder='little', signed=True)
    if (text_sz == 0):
        return -1, -1, -1
    dt = np.dtype('<i4')
    sentence = np.frombuffer(file_obj.read(4 * text_sz), dtype=dt).copy()
    return id_group, text_sz, torch.tensor(sentence[:128]).long()

def load_groups(file_obj, post_count, gr_dict):
    list_of_posts = list() 
    while True:
        new_group = load_group(file_obj)
        if (new_group[0] == -1):
            break
        if (new_group[0] in gr_dict):
            list_of_posts.append(new_group)
    new_gen = np.random.default_rng(777)
    ids = np.arange(0, len(list_of_posts))
    new_gen.shuffle(ids)
    for i in ids:
        yield list_of_posts[i]

manseed = 11
def sharps_and_letters(stri):
    strt = stri.lower()
    for i in strt:
        if (ord(i) < ord('а') or ord(i) > ord('я')) and (i != '#'):
            return False
    return True

def make_batches(file_obj, post_count, gr_dict, batch_size=32, test_frac = 0.3, val_frac = 1/3, test_train_split_seed = 22, noise_scheme='bert_standard'):
    print("HERE", flush=True)
    if (noise_scheme != 'bert_standard'):
        raise NotImplementedError

    my_generator = np.random.default_rng(test_train_split_seed)

    test_seqs = []
    test_ids = []
    train_seqs = []
    train_ids = []
    val_seqs = []
    val_ids = []
    global manseed
    rand_m = my_generator.random(size = max(list(gr_dict.values())) + 1)
    print((rand_m > 0.3).sum(), flush=True)
    for id_group, text_sz, sent in load_groups(file_obj, post_count, gr_dict):
        if rand_m[gr_dict[id_group]] < test_frac:
            if rand_m[gr_dict[id_group]] > test_frac * val_frac:
                test_seqs.append(sent)
                test_ids.append(gr_dict[id_group])
            else:
                val_seqs.append(sent)
                val_ids.append(gr_dict[id_group])
        else:
            train_seqs.append(sent)
            train_ids.append(gr_dict[id_group])
        torch.manual_seed(manseed)
        manseed *= 5
        manseed %= 13
        if (len(test_seqs) == batch_size):
            inpseq = pad_sequence(test_seqs, batch_first=True)

            masked_mask = (torch.rand(inpseq.shape) > 0.15)
            masked_mask = masked_mask | (inpseq < 105)
            
            for i1 in range(inpseq.shape[0]):
                for j1 in range(inpseq.shape[1]):
                    text_tok = tokenizer.batch_decode([[inpseq[i1][j1]]])[0]
                    if not(sharps_and_letters(text_tok)):
                        masked_mask[i1][j1] = True


            
            rand_tok = (~masked_mask) * (torch.rand(inpseq.shape) > 0.9)
            same_tok = ((~masked_mask) & (~rand_tok)) * (torch.rand(inpseq.shape) < 1/9)

            target_padding_mask = (inpseq != 0)
            
            newinp = (inpseq * masked_mask + ((~masked_mask) & (~rand_tok) & (~same_tok)) * 103 + (same_tok * inpseq) + (rand_tok * torch.randint(105, 119547, inpseq.shape))) * target_padding_mask
            labels = (inpseq * ((~masked_mask) & target_padding_mask) + (masked_mask | (~target_padding_mask)) * (-100))
    
            yield 1, newinp, labels, target_padding_mask, test_ids

            test_seqs = []
            test_ids = []

        elif (len(train_seqs) == batch_size):
            inpseq = pad_sequence(train_seqs, batch_first=True)

            masked_mask = (torch.rand(inpseq.shape) > 0.15)
            masked_mask = masked_mask | (inpseq < 105)

            for i1 in range(inpseq.shape[0]):
                for j1 in range(inpseq.shape[1]):
                    text_tok = tokenizer.batch_decode([[inpseq[i1][j1]]])[0]
                    if not(sharps_and_letters(text_tok)):
                        masked_mask[i1][j1] = True



            rand_tok = (~masked_mask) * (torch.rand(inpseq.shape) > 0.9)
            same_tok = ((~masked_mask) & (~rand_tok)) * (torch.rand(inpseq.shape) < 1/9)

            target_padding_mask = (inpseq != 0)

            newinp = (inpseq * masked_mask + ((~masked_mask) & (~rand_tok) & (~same_tok)) * 103 + (same_tok * inpseq) + (rand_tok * torch.randint(105, 119547, inpseq.shape))) * target_padding_mask
            labels = (inpseq * ((~masked_mask) & target_padding_mask) + (masked_mask | (~target_padding_mask)) * (-100))

            yield 0, newinp, labels, target_padding_mask, train_ids

            train_seqs = []
            train_ids = []

        elif len(val_seqs) == batch_size:
            inpseq = pad_sequence(val_seqs, batch_first=True)

            masked_mask = (torch.rand(inpseq.shape) > 0.15)
            masked_mask = masked_mask | (inpseq < 105)

            for i1 in range(inpseq.shape[0]):
                for j1 in range(inpseq.shape[1]):
                    text_tok = tokenizer.batch_decode([[inpseq[i1][j1]]])[0]
                    if not(sharps_and_letters(text_tok)):
                        masked_mask[i1][j1] = True



            rand_tok = (~masked_mask) * (torch.rand(inpseq.shape) > 0.9)
            same_tok = ((~masked_mask) & (~rand_tok)) * (torch.rand(inpseq.shape) < 1/9)

            target_padding_mask = (inpseq != 0)

            newinp = (inpseq * masked_mask + ((~masked_mask) & (~rand_tok) & (~same_tok)) * 103 + (same_tok * inpseq) + (rand_tok * torch.randint(105, 119547, inpseq.shape))) * target_padding_mask
            labels = (inpseq * ((~masked_mask) & target_padding_mask) + (masked_mask | (~target_padding_mask)) * (-100))

            yield 2, newinp, labels, target_padding_mask, val_ids

            val_seqs = []
            val_ids = []





class BertModified(nn.Module):

    def __init__(self, basic_model, group_embedding, use_group_ids):
        super().__init__()
        self.model = basic_model
        self.gr_embs = nn.Embedding.from_pretrained(group_embedding)
        self.pooling = nn.Linear(group_embedding.shape[1], 768)
        self.use_gr_ids = use_group_ids


    def forward(self, newinp, target_padding, labels, group_ids=None):
        if (group_ids is None):
            loss, _ = self.model.forward(newinp, attention_mask=target_padding, labels=labels, return_dict=False)
            del _

            return loss
        
        gr_embs = self.pooling(self.gr_embs(group_ids))
        print(gr_embs.shape, flush=True)
        gr_embs = gr_embs.reshape((gr_embs.shape[0], 768))
        embs = self.model.bert.embeddings(newinp)
        print(gr_embs.shape, embs.shape, flush=True)
        embs[:, 0, :] += gr_embs
        for i in range(12):
            layer = self.model.bert.encoder.layer[i]
            embs = layer(embs, attention_mask=self.model.get_extended_attention_mask(target_padding, input_shape=embs.shape, device=self.model.device))[0]

        scores = self.model.cls(embs)
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(scores.view(-1, self.model.config.vocab_size), labels.view(-1))
        return masked_lm_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True, help='Version for logs saving')
    parser.add_argument('--n-epochs', default=10, type=int, help='Total number of epochs')
    parser.add_argument('--warmup-epochs', default=2.0, type=float, help='Warmup Epochs (accepts all positive real values)')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate')
    parser.add_argument('--batch-size', default=256, type=int, help='Batch Size')
    parser.add_argument('--cycle-n', default=-1, type=int, help='Number of cycles used in learning rate scheduler (for scheduler with hard restart)')
    parser.add_argument('--is-linear', default=False, type=bool, help='Whether to use linear scheduler or a scheduler with hard restarts.')
    parser.add_argument('--use-gr-embeds', default=False, type=bool, help='Whether to use group embeddings in model')
    parser.add_argument('--pret', default=0.06, type=float, help='Pretraining Epochs (accepts all positive real values)')
    args = parser.parse_args()
    return args

class BertPeacock(nn.Module):

    def __init__(self, bertmod_pretrained, group_embedding, use_group_ids, peacock_dim=32, peacock_i=10):
        super().__init__()
        self.model = copy.deepcopy(bertmod_pretrained.model)


        self.gr_embs = nn.Embedding.from_pretrained(group_embedding)
        self.mlp = nn.Sequential(nn.Linear(768 + 128, 512), nn.GELU(), nn.Linear(512, peacock_dim), nn.Softmax(dim=1))
        self.pw = peacock_dim
        self.pi = peacock_i
        self.use_gr_ids = use_group_ids
        self.peacock = nn.ModuleList([copy.deepcopy(self.model.bert.encoder.layer[self.pi]) for i in range(self.pw)])

    def forward(self, newinp, target_padding, labels, group_ids=None):
        if (group_ids is None):
            loss, _ = self.model.forward(newinp, attention_mask=target_padding, labels=labels, return_dict=False)
            del _

            return loss

        gr_embs = (self.gr_embs(group_ids))
        embs = self.model.bert.embeddings(newinp)
        for i in range(12):
            if (i != self.pi):
                layer = self.model.bert.encoder.layer[i]
                embs = layer(embs, attention_mask=self.model.get_extended_attention_mask(target_padding, input_shape=embs.shape, device=self.model.device))[0]
            else:
                newv = self.mlp(gr_embs)
                print(f"MLP DIM: {newv.shape}")
                for j in range(self.pw):
                    if (j == 0):
                        out = newv[:, j][:, None, None] * self.peacock[j](embs, attention_mask=self.model.get_extended_attention_mask(target_padding, input_shape=embs.shape, device=self.model.device))[0]
                    else:
                        out += newv[:, j][:, None, None] * self.peacock[j](embs, attention_mask=self.model.get_extended_attention_mask(target_padding, input_shape=embs.shape, device=self.model.device))[0]
                embs = out

        scores = self.model.cls(embs)
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(scores.view(-1, self.model.config.vocab_size), labels.view(-1))
        return masked_lm_loss


def main():

    args = parse_args()

    u_man = np.load("u_man.npy").reshape(309710, 768)
    s_man = np.load("s_man.npy").reshape(768)
    new_emb = u_man @ np.diag(np.sqrt(s_man))





    ids = list(map(lambda x: int(re.split('[A-z]', x)[-1]), open("ids_from_5000_to_200000.txt").read().split('\n')))





    gr_dict = dict(list(zip(ids, range(309710))))





    file = open('vk_graph.emb')





    file.readline()





    embs = np.zeros((309710, 128))
    for i in range(309710):
        mass = (list(map(float, file.readline().split())))
        embs[gr_dict[int(mass[0])]] = mass[1:]





    total_embedding = np.concatenate((new_emb[:, :], (embs)), axis=1)


    total_embedding -= total_embedding.mean(axis=0)
    total_embedding *= 100000
    total_embedding /= (total_embedding.std(axis=0) + 1)


    total_embedding.shape





    bmodel = torch.load(f'Models/bert_epoch_0_vfull_pure_9_base.pt')
    
    tot_model = BertPeacock(bmodel, torch.tensor(total_embedding.copy()).float(), use_group_ids=True) 



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.get_device_name(0), flush=True)
   



    tot_model = tot_model.to(device)




    gr_count = 309710






    lr = args.lr
    warmup_epochs = args.warmup_epochs
    n_epochs = args.n_epochs
    batch_size=args.batch_size

    use_gr_ids = args.use_gr_embeds
    is_linear = args.is_linear
    cycle_n = args.cycle_n

    version = args.version
    print(version, flush=True)


    opt = AdamW(tot_model.parameters(), lr=4*lr)
    dset_size = 26528481 // batch_size

    sched = None
    num_pretraining_steps = dset_size * args.pret
    if is_linear:
        sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=dset_size * warmup_epochs, num_training_steps=(n_epochs - warmup_epochs) * dset_size)
    else:
        sched = get_cosine_with_hard_restarts_schedule_with_warmup(opt, num_warmup_steps=dset_size * warmup_epochs, num_training_steps=(n_epochs - warmup_epochs) * dset_size, num_cycles=cycle_n)





    debug_out = open(f"Logs/debug_v{version}.txt", "wt", buffering=1)
    ep_losses = open(f"Logs/epoch_v{version}.txt", "wt", buffering=1)





    config_out = open(f"Logs/config_v{version}.txt", "wt", buffering=1)





    print(f'lr: {lr}', file=config_out)
    print(f'warmup_epochs: {warmup_epochs}', file=config_out)
    print(f'n_epochs: {n_epochs}', file=config_out)
    print(f'batch_size: {batch_size}', file=config_out)
    print(f'use_gr_ids: {use_gr_ids}', file=config_out)
    print(f'is_linear: {is_linear}', file=config_out)
    print(f'cycle_n: {cycle_n}', file=config_out)



    cou_steps = 0
    print(device)
    tot_model.train()
    data_file = open("posts_128_max_cou_1000.bin", 'rb')
    for epoch in range(n_epochs):


        train_losses = []
        train_mask_losses = []
        test_losses = []
        test_mask_losses = []

        startt = time.time()
        modt = 0
        cou = 0
        data_file = open("posts_128_max_cou_1000.bin", 'rb')
        for is_testing, newinp, labels, target_padding_mask, gr_ids in make_batches(data_file, 37897830, gr_dict, batch_size=batch_size):
            start_mod = time.time()

            if is_testing == 0:
                tot_model.train()
            else:
                tot_model.eval()

            if (use_gr_ids and cou_steps >= num_pretraining_steps):
                loss = 0
                for spl in range(4):
                    newinp1 = newinp[spl * 8: spl * 8 + 8]
                    tpad1 = target_padding_mask[spl * 8: spl * 8 + 8]
                    labs1 = labels[spl * 8: spl * 8 + 8]
                    gr_idst = gr_ids[spl * 8: spl * 8 + 8]
                    loss += tot_model(newinp1.to(device), tpad1.to(device), labs1.to(device), group_ids=torch.tensor(gr_idst).long().to(device))
                loss /= 4
            else:
                loss = tot_model(newinp.to(device), target_padding_mask.to(device), labels.to(device))

            if (is_testing == 0):

                loss.backward()
                if (epoch == 0 and len(train_losses) < 4200):
                    torch.nn.utils.clip_grad_norm_(tot_model.parameters(), 1.0)

                cou_steps += 1
                opt.step()
                sched.step()
                opt.zero_grad()

                train_losses.append(loss.data.item())
                del loss

                if (len(train_losses) % 100 == 0):
                    print(cou, len(train_losses) + len(test_losses), "TRAIN:", sum(train_losses[-100:]) / len(train_losses[-100:]), time.time() - startt, modt + time.time() - start_mod, file=debug_out)

            elif is_testing == 1:

                test_losses.append(loss.data.item())
                del loss
    
                if (len(test_losses) % 100 == 0):
                    print(cou, len(train_losses) + len(test_losses), "TEST:", sum(test_losses[-100:]) / len(test_losses[-100:]),  time.time() - startt, modt + time.time() - start_mod, file=debug_out)
            if ((len(test_losses) + len(train_losses)) % 100000 == 99999):
                torch.save(tot_model, f"Models/peacock_midepoch_{epoch}_v{version}_{len(test_losses)}.pt")
            if (epoch == 0 and len(train_losses) == 4200):
                for param in tot_model.model.parameters():
                    param.requires_grad = False
                opt = AdamW(tot_model.parameters(), lr=4*lr)
                dset_size = 26528481 // batch_size

                sched = None
                num_pretraining_steps = 0
                sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=dset_size * warmup_epochs, num_training_steps=(n_epochs - warmup_epochs) * dset_size)


            modt += time.time() - start_mod

        torch.save(test_losses, f"Losses/test_losses{epoch}_v{version}_log")
        torch.save(train_losses, f"Losses/train_losses{epoch}_v{version}_log")
        torch.save(tot_model, f"Models/bert_epoch_{epoch}_v{version}.pt")
        print(f"TRAIN TOTAL: {sum(train_losses) / len(train_losses)},\t TEST TOTAL: {sum(test_losses) / len(test_losses)},\t TIME: {time.time() - startt}", file=ep_losses)


if __name__ == '__main__':
        main()



