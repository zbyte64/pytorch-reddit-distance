from dataset import *
from model import Distance
from torch import optim
import torch
import os


def test():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load('model.pt').to(device)
    ds = reddit_triplet_datasource(os.environ.get('DATA_DIR', 'reddit_data'), 100, sigma=1.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for i, batch in enumerate(ds):
        p = model.encode_post(batch['subreddit'].to(device), batch['op_body'].to(device))
        print(p.detach())
        break
        


if __name__ == '__main__':
    test()
