from file_read_backwards import FileReadBackwards
import json
import os
from collections import defaultdict
from torchnlp.word_to_vector import FastText
from torchnlp.utils import pad_tensor
from bpemb import BPEmb
import torch
import random


def comment_stream(filename):
    with FileReadBackwards(filename, encoding="utf-8") as frb:
        for l in frb:
            o = json.loads(l)
            yield o


EMPTY_BODY_TOKENS = set(['[removed]', '[deleted]'])


def comment_triplet_stream(filenames):
    '''
    Yeilds triples of comments:
        OP: original post for origin distance calculation
        Positive: a response with positive karma
        Negative: a response with lesser karma or outside sample
    '''
    orphans = defaultdict(list) #parent_id => list
    for filename in filenames:
        for comment in comment_stream(filename):
            body, parent_id, _id = comment['body'], comment['parent_id'], comment['id']
            link_id = 't1_' + _id
            responses = orphans.pop(link_id, None)
            if body in EMPTY_BODY_TOKENS:
                pass
            else:
                if responses:
                    responses.sort(key=lambda c: c['score'], reverse=True)
                    for p, n in zip(responses, responses[1:]):
                        yield (comment, p, n)
                    #select outside comment
                    if len(orphans):
                        ro = random.choice(list(orphans.keys()))
                        rc = random.choice(orphans[ro])
                        yield (comment, responses[-1], rc)
                    #else:
                    #    assert False
                if parent_id.startswith('t1_') and comment['score'] > 0:
                    orphans[parent_id].append(comment)


def collect_filenames(directory):
    filenames = []
    for cur_path, directories, files in os.walk(directory):
        for file in files:
            if '.' in file or file == 'README':
                continue
            filenames.append(os.path.join(directory, cur_path, file))
    return filenames


def conditional_pad_tensor(s, length):
    if s.shape[0] > length:
        return s[:length]
    return pad_tensor(s, length)


def transformer(sequence_length, vectorizer=None):
    assert sequence_length > 0
    if vectorizer is None:
        embedder = BPEmb(lang='en', vs=50000, dim=300)
        embed = lambda s: torch.Tensor(embedder.embed(s))
        if sequence_length == 1:
            vectorizer = lambda sequence: torch.mean(embed(sequence), dim=0, keepdim=True)
        else:
            vectorizer = lambda sequence: conditional_pad_tensor(embed(sequence), sequence_length)
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


def reddit_triplet_datasource(batch_size=100, heads=100):
    body_t = transformer(100)
    subreddit_t = transformer(1)
    time_t = lambda x: torch.tensor(int(x), dtype=torch.float)
    filenames = collect_filenames(os.environ['DATA_DIR'])
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
        if len(batch['op_body']) == batch_size:
            yield {k: torch.stack(v) for k, v in batch.items()}
            batch = defaultdict(list)
    #yield batch


if __name__ == '__main__':
    import sys
    directory = sys.argv[-1]
    filenames = collect_filenames(directory)
    print(filenames)
    for t in triple_stream(filenames):
        print(t)
    