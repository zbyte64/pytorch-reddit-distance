from dataset import *
from model import Distance
from torch import optim
import torch
import os
import math
from submodules import OpenAIAdam


def train(batch_size=100):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if os.path.exists('model.pt'):
        model = torch.load('model.pt')
    else:
        model = Distance()
    model = model.to(device).train()
    ds = reddit_triplet_datasource(os.environ.get('DATA_DIR', 'reddit_data'), batch_size, sigma=math.pi/2)
    epoch_size = int(1e6)
    #optimizer = optim.Adam(model.parameters(), lr=3e-4)
    optimizer = OpenAIAdam(model.parameters(), t_total=epoch_size, lr=3e-4, weight_decay=1e-5, schedule='warmup_cosine')
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
        #torch.nn.utils.clip_grad_norm_(model.comment_embedder.parameters(), 1)
        optimizer.step()
        if not i % 10:
            print('Loss:', loss.detach().item())
        if not i % 100 and i:
            torch.save(model, 'model.pt')


if __name__ == '__main__':
    train(20)
