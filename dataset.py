from file_read_backwards import FileReadBackwards
import json
import os
from collections import defaultdict
from torchnlp.word_to_vector import FastText
from torchnlp.utils import pad_tensor
from bpemb import BPEmb
import torch
import random
from torch import multiprocessing
from tqdm import tqdm
import numpy as np
import heapq
import itertools


EMPTY_BODY_TOKENS = set(['[removed]', '[deleted]', ''])


def comment_stream(filename):
    '''
    Read comment objects from a file in reverse, skipping over empty bodies
    '''
    with FileReadBackwards(filename, encoding="utf-8") as frb:
        for l in frb:
            o = json.loads(l)
            if o['body'].strip() in EMPTY_BODY_TOKENS:
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
    for filename in filenames:
        last_gen_parents = set(orphans.keys())
        for comment in comment_stream(filename):
            body, parent_id, _id = comment['body'], comment['parent_id'], comment['id']
            link_id = 't1_' + _id
            responses = orphans.pop(link_id, None)
            if responses:
                comment['n_children'] = len(responses)
                responses.sort(key=lambda c: c['score'], reverse=True)
                for p, n in zip(responses, responses[1:]):
                    if random.randint(0, 1):
                        yield (comment, p, n)
                    elif len(orphans):
                        ro = random.choice(list(orphans.keys()))
                        rc = random.choice(orphans[ro])
                        yield (comment, responses[-1], rc)
                    else:
                        yield (comment, p, n)
                #select outside comment
                if len(orphans) and len(responses) == 1:
                    ro = random.choice(list(orphans.keys()))
                    rc = random.choice(orphans[ro])
                    yield (comment, responses[-1], rc)
                #else:
                #    assert False
            if parent_id.startswith('t1_') and comment['score'] > 0:
                orphans[parent_id].append(comment)
        for p in last_gen_parents:
            orphans.pop(p, None)


def collect_filenames(directory):
    filenames = []
    for cur_path, directories, files in os.walk(directory):
        for file in files:
            if '.' in file or file == 'README':
                continue
            filenames.append(os.path.join(cur_path, file))
    return filenames


def conditional_pad_tensor(s, length):
    if s.shape[0] > length:
        return s[:length]
    return pad_tensor(s, length)


def transformer(sequence_length, dtype=torch.float):
    '''
    Transforms a text sequence into an encoded sequence of vectors
    '''
    assert sequence_length > 0
    embedder = BPEmb(lang='en', vs=50000, dim=300)
    cast = lambda v: torch.from_numpy(v).type(dtype)
    embed = embedder.embed
    if sequence_length == 1:
        vectorizer = lambda sequence: cast(np.mean(embed(sequence), axis=0, keepdims=True))
    else:
        vectorizer = lambda sequence: conditional_pad_tensor(cast(embed(sequence)), sequence_length)
    return vectorizer


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
    body_t = transformer(100)
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
    heads = max(heads, len(filenames))
    read_heads = [infinite_triplets() for i in range(heads)]
    def next_entry():
        while True:
            for ds in read_heads:
                yield next(ds)

    batch = defaultdict(list)
    for op_d, p_d, n_d in randomize(next_entry()):
        batch['op_body'].append(body_t(op_d['body']))
        batch['subreddit'].append(subreddit_t(op_d['subreddit']))
        batch['op_created_utc'].append(time_t(op_d['created_utc']))
        batch['p_body'].append(body_t(p_d['body']))
        batch['p_created_utc'].append(time_t(p_d['created_utc']))
        batch['n_body'].append(body_t(n_d['body']))
        batch['n_created_utc'].append(time_t(n_d['created_utc']))
        if 't1_' + op_d['id'] == n_d['parent_id']:
            batch['sigma'].append(torch.tensor(sigma/op_d['n_children']))
        else:
            #topic_drift = (subreddit_t(op_d['subreddit']) - subreddit_t(n_d['subreddit']))**2
            batch['sigma'].append(torch.tensor(sigma))
        if len(batch['op_body']) == batch_size:
            yield {k: torch.stack(v) for k, v in batch.items()}
            batch = defaultdict(list)


def mean_vector_by(group_by, data_dir, num_processes=4):
    multiprocessing.set_sharing_strategy('file_system')
    filenames = collect_filenames(data_dir)
    #randomize processing for more uniform speed prediction
    random.shuffle(filenames)
    body_t = transformer(1, dtype=torch.float64)
    groups = defaultdict(lambda: torch.zeros(1, 300, dtype=torch.float64))
    sum_karma = defaultdict(lambda: 0)
    work_pool = multiprocessing.Pool(processes=num_processes)
    output = [work_pool.apply_async(_file_mean_vector_by, args=(group_by, f)) for f in filenames]
    for result in tqdm(output):
        o_groups, o_sum_karma = result.get()
        for k in o_sum_karma.keys():
            groups[k] += o_groups[k]
            sum_karma[k] += o_sum_karma[k]
    work_pool.close()
    work_pool.join()
    for k, karma in sum_karma.items():
        v = groups[k] / karma
        groups[k] = v.type(torch.float)
    return dict(groups)


def _file_mean_vector_by(group_by, filename):
    body_t = transformer(1, dtype=torch.float64)
    groups = defaultdict(lambda: torch.zeros(1, 300, dtype=torch.float64))
    sum_karma = defaultdict(lambda: 0)
    for obj in comment_stream(filename):
        score = obj['score']
        if score < 1:
            continue
        m = body_t(obj['body'])
        #detect and skip nans
        if torch.sum(m == m).item() == 0:
            #print('bad body', obj)
            continue
        k = obj[group_by]
        groups[k] += m*score
        sum_karma[k] += score
    #cast as plain dict because we cant pickle lambdas passed to defaultdict
    return dict(groups), dict(sum_karma)


def get_subreddit_embedding(data_dir):
    path = 'subreddit_embedding.pt'
    if not os.path.exists(path):
        print('generating subreddit embedding...')
        subreddit_t = mean_vector_by('subreddit', data_dir)
        torch.save(subreddit_t, path)
    else:
        subreddit_t = torch.load(path)
    return lambda x: subreddit_t[x]


def collect_top_comments(k, data_dir, num_processes=4):
    multiprocessing.set_sharing_strategy('file_system')
    filenames = collect_filenames(data_dir)
    work_pool = multiprocessing.Pool(processes=num_processes)
    output = [work_pool.apply_async(_file_collect_top_comments, args=(k, f)) for f in filenames]
    all_top_k = []
    for result in tqdm(output):
        all_top_k.append(result.get())
    work_pool.close()
    work_pool.join()
    top_k = heapq.nlargest(k, itertools.chain(all_top_k), key=lambda x: x['score'])
    return top_k


def _file_collect_top_comments(k, filename):
    return heapq.nlargest(k, comment_stream(filename), key=lambda x: x['score'])


if __name__ == '__main__':
    import sys
    directory = sys.argv[-1]
    filenames = collect_filenames(directory)
    print(filenames)
    for t in triple_stream(filenames):
        print(t)
