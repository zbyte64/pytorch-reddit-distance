import torch
from torch import nn
from torchnlp.nn import Attention
import math
import datetime


circle_embedder = lambda x, max_x: [torch.cos(2*math.pi*x/max_x), torch.sin(2*math.pi*x/max_x)]
dow_embedder = lambda x: circle_embedder(x, 60 * 60 * 24 * 7)
hour_embedder = lambda x: circle_embedder(x, 60 * 60 * 24)
month_embedder = lambda x: circle_embedder(x, 60 * 60 * 24 * 30)
time_embedder = lambda t0, t1: torch.stack([*dow_embedder(t0), *hour_embedder(t0), *month_embedder(t1-t0)], dim=1)


def great_circle_distance(p1, p2):
    v = torch.bmm(p1.view(p1.shape[0], 1, 3), p2.view(p2.shape[0], 3, 1))
    v = v.clamp(max=1, min=-1)
    return torch.acos(v)
    

class Distance(nn.Module):
    def __init__(self, comment_embedder=None, vector_size=300):
        super(Distance, self).__init__()
        if comment_embedder is None:
            comment_embedder = Attention(vector_size)
        self.comment_embedder = comment_embedder
        self.value_to_coords = nn.Linear(vector_size, 3)
        self.karma_time_decay1 = nn.Linear(6, 2)
        self.karma_time_decay2 = nn.Linear(2, 1)
    
    def encode_post(self, query, body):
        v, _ = self.comment_embedder(query, body)
        v = v.squeeze(1)
        return torch.tanh(self.value_to_coords(v))
    
    def forward(self, original_post, response_post):
        query = original_post['subreddit']
        op_v = self.encode_post(query, original_post['body'])
        rp_v = self.encode_post(query, response_post['body'])
        time_embedding = time_embedder(original_post['created_utc'], response_post['created_utc'])
        karma_drift = torch.relu(self.karma_time_decay1(time_embedding))
        karma_drift = torch.relu(self.karma_time_decay2(karma_drift))
        #return torch.sum((op_v - rp_v) ** 2, dim=1, keepdim=True) + karma_drift
        return great_circle_distance(op_v, rp_v) + karma_drift

