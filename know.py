from collections import defaultdict
from dataset import collect_filenames, transformer, get_subreddit_embedding, comment_stream
import torch
import os
import geojson


def write_projections(subreddits, data_dir):
    body_t = transformer(100)
    subreddit_t = get_subreddit_embedding(data_dir)
    queries = [subreddit_t(s) for s in subreddits]
    model = torch.load('model.pt').to('cpu')
    filenames = collect_filenames(data_dir)
    # {subreddit: {int(lat, lng): (top_comment, lat_lng)}}
    lat_lng_buckets = defaultdict(dict)
    batch_size = 100
    batch = list()
    batch_bodies = list()
    batch_queries = list()
    batch_subreddits = list()
    def process_batch():
        q = torch.stack(batch_queries)
        b = torch.stack(batch_bodies)
        c_e = model.encode_post(q, b)
        lat = torch.asin(c_e[:,2]).flatten().tolist()
        lng = torch.atan2(c_e[:,1], c_e[:,0]).flatten().tolist()
        bucket_keys = zip(map(int, lat), map(int, lng))
        for subreddit, comment, key, lat_lng in zip(batch_subreddits, batch, bucket_keys, zip(lat, lng)):
            bucket = lat_lng_buckets[subreddit]
            if key in bucket and bucket[key][0]['score'] >= comment['score']:
                pass
            else:
                bucket[key] = (comment, lat_lng)
        batch.clear()
        batch_bodies.clear()
        batch_queries.clear()
        batch_subreddits.clear()

    for filename in filenames:
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
    for subreddit, top_world in lat_lng_buckets.items():
        geos = list()
        for comment, (lat, lng) in top_world.values():
            geos.append(geojson.Feature(
                geometry=geojson.Point((lat, lng)),
                properties=comment))
        geos = geojson.FeatureCollection(geos)
        outfile = open(os.path.join(data_dir, '%s.geojson' % subreddit), 'w')
        geojson.dump(geos, outfile)
        outfile.close()


if __name__ == '__main__':
    import sys
    subreddits = sys.argv[1:]
    data_dir = os.environ.get('DATA_DIR', 'reddit_data')
    print('Writing projections for:', subreddits)
    write_projections(subreddits, data_dir)
