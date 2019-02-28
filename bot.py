import torch
from dataset import transformer


def find_top_k(op, comments, k=1, subreddit='politics'):
    model = torch.load('model.pt')
    ce = model.comment_embedder
    encoded = ce(subreddit, comments)