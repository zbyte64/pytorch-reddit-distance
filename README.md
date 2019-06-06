Trains karmic based distance embedding using attention, and GPT2.

You will need to download reddit comments in order to train: magnet:?xt=urn:btih:85a5bd50e4c365f8df70240ffd4ecc7dec59912b&dn=reddit%5Fdata&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce&ws=https%3A%2F%2Ffiles.pushshift.io%2Freddit%2Fcomments%2F

To train:
```
DATA_DIR=reddit_data python train.py
```

To query a comment against a user's comments:
```
REDDIT_ID=yourappid
REDDIT_SECRET=yourappsecret
python bot.py ehew206 zbyte64 askscience
```

To write a geospatial projection of a subreddit:
```
python know.py askscience
```


Credits:

- Attention Layer: pytorch-nlp
- GPT2 Model: https://github.com/huggingface/pytorch-pretrained-BERT