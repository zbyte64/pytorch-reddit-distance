Trains karmic based distance embedding using attention, and BPEmb.

You will need to download reddit comments in order to train: magnet:?xt=urn:btih:85a5bd50e4c365f8df70240ffd4ecc7dec59912b&dn=reddit%5Fdata&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce&ws=https%3A%2F%2Ffiles.pushshift.io%2Freddit%2Fcomments%2F

To train:
```
DATA_DIR=reddit_data python train.py
```

To compare a given comment to a user's set of comments relative to a given subreddit:
```
REDDIT_ID=yourappid
REDDIT_SECRET=yourappsecret
python bot.py ehew206 zbyte64 eli5
```

To use the model as a knowledge base where you can query based on a given statement and subreddit:
```
python know.py "Why is the sky blue?" askscience
```
