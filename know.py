from collections import defaultdict
from dataset import collect_filenames, transformer, get_subreddit_embedding, comment_stream
import torch
import os
import geojson
from tqdm import tqdm
import random
from torch import multiprocessing
import math


def get_projections_from_file(subreddits, data_dir, filename, batch_size=100):
    body_t = transformer(100)
    subreddit_t = get_subreddit_embedding(data_dir)
    queries = [subreddit_t(s) for s in subreddits]
    model = torch.load('model.pt').to('cpu').eval()
    lat_lng_buckets = {s: dict() for s in subreddits}
    batch = list()
    batch_bodies = list()
    batch_queries = list()
    batch_subreddits = list()
    def process_batch():
        q = torch.stack(batch_queries)
        b = torch.stack(batch_bodies)
        c_e = model.encode_post(q, b)
        lat = torch.atan2(c_e[:,2], (c_e[:,1]**2 + c_e[:,0]**2)**.5) / math.pi * 180
        lng = torch.atan2(c_e[:,1], c_e[:,0]) / math.pi * 180
        lat = lat.flatten().tolist()
        lng = lng.flatten().tolist()
        bucket_keys = zip(map(int, lat), map(int, lng))
        for subreddit, comment, key, lat_lng in zip(batch_subreddits, batch, bucket_keys, zip(lat, lng)):
            bucket = lat_lng_buckets[subreddit]
            if key in bucket and bucket[key][1]['score'] >= comment['score']:
                pass
            else:
                bucket[key] = (lat_lng, comment)
        batch.clear()
        batch_bodies.clear()
        batch_queries.clear()
        batch_subreddits.clear()

    comments = comment_stream(filename)
    for comment in comments:
        if comment['score'] < 10:
            continue
        body = body_t(comment['body'])
        for subreddit, q in zip(subreddits, queries):
            batch.append(comment)
            batch_bodies.append(body)
            batch_queries.append(q)
            batch_subreddits.append(subreddit)
            if len(batch) == batch_size:
                process_batch()
    if len(batch):
        process_batch()
    return lat_lng_buckets


def write_projections(subreddits, data_dir, num_processes=4):
    filenames = collect_filenames(data_dir)
    random.shuffle(filenames)
    # {subreddit: {lat,lng: comment}}
    lat_lng_buckets = {s: dict() for s in subreddits}
    
    work_pool = multiprocessing.Pool(processes=num_processes)
    output = [work_pool.apply_async(get_projections_from_file, args=(subreddits, data_dir, f)) for f in filenames]

    for top_month_task in tqdm(output):
        # {subreddit: {int(lat, lng): (top_comment, lat_lng)}}
        top_month = top_month_task.get()
        for subreddit, top_world in top_month.items():
            s_bucket = lat_lng_buckets[subreddit]
            for p, (k, c) in top_world.items():
                if p in s_bucket:
                    if s_bucket[p][1]['score'] < c['score']:
                        s_bucket[p] = (k,c)
                else:
                    s_bucket[p] = (k,c)
            #lat_lng_buckets[subreddit].update({k:c for k, c in top_world.values()})
    work_pool.close()
    work_pool.join()
        
    for subreddit, top_world in lat_lng_buckets.items():
        geos = list()
        for (lat, lng), comment  in top_world.values(): #top_world.items():
            geos.append(geojson.Feature(
                geometry=geojson.Point((lng, lat)),
                properties=comment))
        fc = geojson.FeatureCollection(geos)
        outfile = open(os.path.join(data_dir, '%s.geojson' % subreddit), 'w')
        geojson.dump(fc, outfile)
        outfile.close()


if __name__ == '__main__':
    import sys
    subreddits = sys.argv[1:]
    data_dir = os.environ.get('DATA_DIR', 'reddit_data')
    print('Writing projections for:', subreddits)
    write_projections(subreddits, data_dir)
