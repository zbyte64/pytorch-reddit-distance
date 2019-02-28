from dataset import *
from model import Distance
from torch import optim 
import torch
import os


def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Distance().to(device)
    ds = reddit_triplet_datasource(100)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for i, batch in enumerate(ds):
        op = {
            'body': batch['op_body'].to(device),
            'subreddit': batch['subreddit'].to(device),
            'created_utc': batch['op_created_utc'].to(device),
        }
        p = {
            'body': batch['p_body'].to(device),
            'created_utc': batch['p_created_utc'].to(device),
        }
        n = {
            'body': batch['n_body'].to(device),
            'created_utc': batch['n_created_utc'].to(device),
        }
        model.zero_grad()
        d_t = model(op, p)
        d_f = model(op, n)
        loss = torch.mean(torch.relu(d_t - d_f + 1e-8))
        loss.backward()
        optimizer.step()
        if not i % 10:
            print('Loss:', loss.detach().item())
        if not i % 100 and i:
            torch.save(model, 'model.pt')


if __name__ == '__main__':
    train()