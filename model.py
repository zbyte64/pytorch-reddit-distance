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


#https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 100):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len] #Variable(self.pe[:,:seq_len], requires_grad=False)
        return x


class CommentEmbedder(nn.Module):
    '''
    Self-attention comment embedding
    '''
    def __init__(self, vector_size=300):
        super(CommentEmbedder, self).__init__()
        self.positional_encoder = PositionalEncoder(vector_size)
        self.comment_embedder = Attention(vector_size)

    def forward(self, body):
        p_body = self.positional_encoder(body)
        s = list()
        for i in range(p_body.shape[1]): #(batch, sequence, values)
            v, _ = self.comment_embedder(p_body[:,i].unsqueeze(1), p_body)
            s.append(v)
        return torch.cat(s, dim=1)


class SubredditDistance(nn.Module):
    def __init__(self, vector_size=300):
        super(SubredditDistance, self).__init__()
        self.comment_embedder = CommentEmbedder(vector_size)
        self.subreddit_projection = Attention(vector_size)
        self.value_to_coords = nn.Linear(vector_size, 30)
        self.karma_time_decay1 = nn.Linear(6, 2)
        self.karma_time_decay2 = nn.Linear(2, 1)

    def project_post(self, query, body):
        e_body = self.comment_embedder(body)
        v, _ = self.subreddit_projection(query, e_body)
        return self.value_to_coords(v)

    def forward(self, original_post, response_post):
        query = original_post['subreddit']
        op_v = self.project_post(query, original_post['body'])
        rp_v = self.project_post(query, response_post['body'])
        time_embedding = time_embedder(original_post['created_utc'], response_post['created_utc'])
        karma_drift = torch.relu(self.karma_time_decay1(time_embedding))
        karma_drift = torch.relu(self.karma_time_decay2(karma_drift))
        return torch.sum((op_v - rp_v) ** 2, dim=1, keepdim=True) + karma_drift
