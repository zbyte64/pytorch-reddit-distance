import torch
import os
from dataset import transformer, get_subreddit_embedding
import praw


def find_top_k(op, comments, k=1, subreddit='politics'):
    subreddit_t = get_subreddit_embedding(os.environ.get('DATA_DIR', 'reddit_data'))
    body_t = transformer(100)
    model = torch.load('model.pt').to('cpu')
    query = subreddit_t(subreddit).unsqueeze(0)
    o_body = body_t(op.body).unsqueeze(0)
    c_body = torch.stack([body_t(c.body) for c in comments])
    print(c_body.shape, o_body.shape, query.shape)
    o_v = model.project_post(query, o_body).squeeze(0)
    c_v = model.project_post(query.repeat(len(comments), 1, 1), c_body)
    c_d = torch.sum((c_v - o_v) ** 2, dim=2).squeeze(1)
    print(c_d.shape, c_v.shape, o_v.shape)
    _, c_i = torch.topk(c_d, k, largest=False, sorted=True)
    return c_i


def suggest_responses(comment_id, redditor_name, k=5, subreddit='politics'):
    reddit = praw.Reddit(client_id=os.environ['REDDIT_ID'],
                         client_secret=os.environ['REDDIT_SECRET'],
                         user_agent='reddit distance bot')
    op = reddit.comment(comment_id)
    assert op.body
    redditor = reddit.redditor(redditor_name)
    comments = list(redditor.comments.top())
    assert comments
    top_i = find_top_k(op, comments, k=k, subreddit=subreddit)
    print(top_i.shape)
    top_r = [comments[i] for i in top_i.flatten().tolist()]
    return top_r


if __name__ == '__main__':
    import sys
    comment_id, redditor_name, subreddit = sys.argv[-3:]
    best_responses = suggest_responses(comment_id, redditor_name, subreddit)
    print('Preferred resonses:')
    for r in best_responses:
        print(r.body)
