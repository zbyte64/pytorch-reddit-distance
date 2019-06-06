from file_read_backwards import FileReadBackwards
import json
import os
from collections import defaultdict
from torchnlp.word_to_vector import FastText
from torchnlp.utils import pad_tensor
from bpemb import BPEmb
from submodules import GPT2Tokenizer, GPT2Model
import torch
import torch.nn.functional as F
import random
from torch import multiprocessing
from tqdm import tqdm
import numpy as np
import heapq
import itertools
import io
import gzip
import base64
from functools import partial
from itertools import chain, islice


EMPTY_BODY_TOKENS = set(['[removed]', '[deleted]', ''])
EMBED_SIZE = 768


def comment_stream(filename, encode=None):
    '''
    Read comment objects from a file in reverse, skipping over empty bodies
    Tokenizes body as e_body
    '''
    if encode is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        encode = lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)[:tokenizer.max_len])
    with FileReadBackwards(filename, encoding="utf-8") as frb:
        for l in frb:
            o = json.loads(l)
            if o['body'].strip() in EMPTY_BODY_TOKENS:
                continue
            o['e_body'] = encode(o['body'])
            if not len(o['e_body']):
                continue
            yield o
            

def comment_triplet_stream(filenames):
    '''
    Yeilds triples of comments:
        OP: original post for origin distance calculation
        Positive: a response with positive karma
        Negative: a response with lesser karma or outside sample
    '''
    orphans = defaultdict(list) #parent_id => list
    sample_q = list()
    for filename in filenames:
        last_gen_parents = set(orphans.keys())
        for comment in comment_stream(filename):
            parent_id, _id = comment['parent_id'], comment['id']
            link_id = 't1_' + _id
            responses = orphans.pop(link_id, None)
            if responses:
                comment['n_children'] = len(responses)
                responses.sort(key=lambda c: c['score'], reverse=True)
                for i, r in enumerate(responses):
                    r['child_index'] = i
                for p, n in zip(responses, responses[1:]):
                    if random.randint(0, 1):
                        yield (comment, p, n)
                    elif len(sample_q):
                        rc = random.choice(sample_q)
                        yield (comment, responses[-1], rc)
                    else:
                        yield (comment, p, n)
                #select outside comment
                if len(sample_q) and len(responses) == 1:
                    rc = random.choice(sample_q)
                    yield (comment, responses[-1], rc)
                #else:
                #    assert False
            if parent_id.startswith('t1_') and comment['score'] > 0:
                orphans[parent_id].append(comment)
                if len(sample_q) == 1000:
                    sample_q[random.randint(0, 1000-1)] = comment
                else:
                    sample_q.append(comment)
        for p in last_gen_parents:
            orphans.pop(p, None)


def collect_filenames(directory, suffix=None, rebalance=False):
    filenames = []
    for cur_path, directories, files in os.walk(directory):
        for file in files:
            if suffix is None:
                if '.' in file or file == 'README':
                    continue
            elif not file.endswith(suffix):
                continue
            filenames.append(os.path.join(cur_path, file))
    if rebalance:
        from itertools import chain
        _filenames = list(chain(*zip(filenames[::-2], filenames[1::2])))
        if len(filenames) % 2 == 1:
            _filenames.append(filenames[0])
        filnames = _filenames
    return filenames


def conditional_pad_tensor(s, length):
    if s.shape[0] > length:
        return s[:length]
    return pad_tensor(s, length)


class GPT2Encoder(torch.nn.Module):
    def __init__(self):
        super(GPT2Model).__init__()
        self.model = GPT2Model.from_pretrained("gpt2").eval()
    
    def encode(self, token_ids, max_len=512):
        token_ids = torch.tensor([token_ids[:max_len]]).to(self.device)
        hidden_state, presents = self.model(token_ids)
        return hidden_state[0]
    
    def encode_batch(self, token_ids, max_len=512):
        padded_token_ids = list(map(lambda r: conditional_pad_tensor(
            torch.tensor(r[:max_len], dtype=torch.long).to(self.device), max_len).unsqueeze(0), token_ids))
        padded_token_ids = torch.cat(padded_token_ids, dim=0)
        hidden_state, presents = self.model(padded_token_ids)
        return hidden_state


def randomize(iterable, bufsize=1000):
    ''' generator that randomizes an iterable. space: O(bufsize). time: O(n). '''
    #https://gist.github.com/razamatan/736048/ace53f68d213a81e134c235cb27917deabf7c5fc
    buf = list()
    for x in iterable:
        if len(buf) == bufsize:
            i = random.randrange(bufsize)
            yield buf[i]
            buf[i] = x
        else:
            buf.append(x)
    random.shuffle(buf)
    while buf: yield buf.pop()


def reddit_triplet_datasource(data_dir, batch_size=100, heads=100, sigma=1e-1):
    encoder = GPT2Encoder().device('cuda')
    body_t = lambda x: x
    subreddit_t = get_subreddit_embedding(data_dir)
    time_t = lambda x: torch.tensor(int(x), dtype=torch.float)
    filenames = collect_filenames(data_dir)
    assert filenames, 'No data found'
    def infinite_triplets(start_index=None):
        if start_index is None:
            start_index = random.randint(0, len(filenames)-1)
        s = comment_triplet_stream(filenames[start_index:])
        while True:
            try:
                yield next(s)
            except StopIteration:
                s = comment_triplet_stream(filenames)
                yield next(s)
            except json.decoder.JSONDecodeError:
                pass
    heads = max(heads, len(filenames))
    read_heads = [infinite_triplets() for i in range(heads)]
    def next_entry():
        while True:
            for ds in read_heads:
                yield next(ds)

    batch = defaultdict(list)
    for op_d, p_d, n_d in randomize(next_entry()):
        batch['op_body'].append(body_t(op_d['e_body']))
        batch['subreddit'].append(subreddit_t(op_d['subreddit']))
        batch['op_created_utc'].append(time_t(op_d['created_utc']))
        batch['p_body'].append(body_t(p_d['e_body']))
        batch['p_created_utc'].append(time_t(p_d['created_utc']))
        batch['n_body'].append(body_t(n_d['e_body']))
        batch['n_created_utc'].append(time_t(n_d['created_utc']))
        if 't1_' + op_d['id'] == n_d['parent_id']:
            assert n_d['parent_id'] == p_d['parent_id']
            sibling_sigma = sigma * (n_d['child_index'] - p_d['child_index']) / op_d['n_children']
            batch['sigma'].append(torch.tensor(sibling_sigma))
        else:
            topic_drift = F.cosine_similarity(subreddit_t(op_d['subreddit']), subreddit_t(n_d['subreddit']), dim=0, eps=1e-6)
            topic_drift = (topic_drift.mean() * sigma).clamp(0, sigma)
            sibling_sigma = sigma * (1 - (1 + p_d['child_index']) / op_d['n_children'])
            batch['sigma'].append(topic_drift + sibling_sigma)
        if len(batch['op_body']) == batch_size:
            batch['op_body'] = encoder.encode_batch(batch['op_body'])
            batch['p_body'] = encoder.encode_batch(batch['p_body'])
            batch['n_body'] = encoder.encode_batch(batch['n_body'])
            yield {k: torch.stack(v) if isinstance(v, list) else v for k, v in batch.items()}
            batch = defaultdict(list)


def mean_vector_by(group_by, data_dir, num_processes=2, chunksize=1, batch_size=4):
    multiprocessing.set_sharing_strategy('file_system')
    filenames = collect_filenames(data_dir, rebalance=True)
    groups = defaultdict(lambda: torch.zeros(1, EMBED_SIZE, dtype=torch.float32))
    sum_karma = defaultdict(lambda: 0)
    work_pool = multiprocessing.Pool(processes=num_processes)
    f = partial(_file_mean_vector_by, group_by, batch_size=batch_size)
    progress = tqdm(total=len(filenames))
    try:
        output = work_pool.imap_unordered(f, filenames, chunksize)
        for (o_groups, o_sum_karma) in output:
            for k in o_sum_karma.keys():
                groups[k] += o_groups[k]
                sum_karma[k] += o_sum_karma[k]
            progress.update()
    finally:
        work_pool.close()
        work_pool.join()
        progress.close()
    for k, karma in sum_karma.items():
        v = groups[k] / karma
        groups[k] = v.type(torch.float).cpu()
    return dict(groups)


def batcher(iterable, size):
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        v = list(batchiter)
        if len(v):
            yield v
        else:
            break


@torch.no_grad()
def _file_mean_vector_by(group_by, filename, batch_size=4):
    encoder = GPT2Encoder().device('cuda')
    groups = defaultdict(lambda: torch.zeros(1, EMBED_SIZE, dtype=torch.float32))
    sum_karma = defaultdict(lambda: 0)
    comments = filter(lambda x: x['score'] > 0, comment_stream(filename))
    batched_comments = batcher(comments, batch_size)
    for batch in batched_comments:
        body = encoder.encode_batch([o['e_body'] for o in batch])
        e_body = body.mean(1).cpu()
        for i, obj in enumerate(batch):
            score = obj['score']
            m = e_body[i]
            #detect and skip nans
            if torch.sum(m == m).item() == 0:
                #print('bad body', obj)
                continue
            k = obj[group_by]
            groups[k] += m*score
            sum_karma[k] += score
    #cast as plain dict because we cant pickle lambdas passed to defaultdict
    return dict(groups), dict(sum_karma)


def generate_subreddit_embedding(path, data_dir):
    print('generating subreddit embedding...')
    subreddit_t = mean_vector_by('subreddit', data_dir, 
        batch_size=10, num_processes=2, chunksize=1)
    assert len(subreddit_t)
    torch.save(subreddit_t, path)


def get_subreddit_embedding(data_dir):
    path = 'subreddit_embedding.pt'
    if not os.path.exists(path):
        generate_subreddit_embedding(path, data_dir)
    else:
        subreddit_t = torch.load(path)
    return lambda x: subreddit_t[x]


def collect_top_comments(k, data_dir, num_processes=4):
    multiprocessing.set_sharing_strategy('file_system')
    filenames = collect_filenames(data_dir)
    work_pool = multiprocessing.Pool(processes=num_processes)
    try:
        output = [work_pool.apply_async(_file_collect_top_comments, args=(k, f)) for f in filenames]
        all_top_k = []
        for result in tqdm(output):
            all_top_k.append(result.get())
    finally:
        work_pool.close()
        work_pool.join()
    top_k = heapq.nlargest(k, itertools.chain(all_top_k), key=lambda x: x['score'])
    return top_k


def _file_collect_top_comments(k, filename):
    return heapq.nlargest(k, comment_stream(filename), key=lambda x: x['score'])


if __name__ == '__main__':
    import sys
    directory = sys.argv[-1]
    generate_subreddit_embedding('subreddit_embedding.pth', directory)
