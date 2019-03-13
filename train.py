from dataset import *
from model import Distance
from torch import optim
import torch
import os


def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Distance().to(device)
    ds = reddit_triplet_datasource(os.environ.get('DATA_DIR', 'reddit_data'), 100, sigma=1.)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
        sigma = batch['sigma'].to(device)
        loss = torch.mean(torch.relu(d_t - d_f + sigma))
        loss.backward()
        optimizer.step()
        if not i % 10:
            print('Loss:', loss.detach().item())
        if not i % 100 and i:
            torch.save(model, 'model.pt')


if __name__ == '__main__':
    train()
