import torch
from torch import nn
from torchnlp.nn import Attention
import math
import datetime


circle_embedder = lambda x, max_x: [torch.cos(2*math.pi*x/max_x), torch.sin(2*math.pi*x/max_x)]
dow_embedder = lambda x: circle_embedder(x, 60 * 60 * 24 * 7)
hour_embedder = lambda x: circle_embedder(x, 60 * 60 * 24)
time_embedder = lambda t0, t1: torch.stack([*dow_embedder(t0), *hour_embedder(t0), (t1-t0)/3e6], dim=1)


class Distance(nn.Module):
    def __init__(self, comment_embedder=None, vector_size=300):
        super(Distance, self).__init__()
        if comment_embedder is None:
            comment_embedder = Attention(vector_size)
        self.comment_embedder = comment_embedder
        self.subreddit_to_query = nn.Linear(vector_size, vector_size)
        self.value_to_coords = nn.Linear(vector_size, 20)
        self.karma_time_decay1 = nn.Linear(5, 2)
        self.karma_time_decay2 = nn.Linear(2, 1)
    
    def forward(self, original_post, response_post):
        subreddit = original_post['subreddit']
        query = self.subreddit_to_query(subreddit)
        op_v, _ = self.comment_embedder(query, original_post['body'])
        rp_v, _ = self.comment_embedder(query, response_post['body'])
        op_v, rp_v = self.value_to_coords(op_v), self.value_to_coords(rp_v)
        time_embedding = time_embedder(original_post['created_utc'], response_post['created_utc'])
        karma_drift = torch.relu(self.karma_time_decay1(time_embedding))
        karma_drift = torch.relu(self.karma_time_decay2(karma_drift))
        return torch.sum((op_v - rp_v) ** 2, dim=1, keepdim=True) + karma_drift

