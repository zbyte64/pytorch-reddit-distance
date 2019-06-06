import importlib.util
import os
import sys


libpath = os.path.abspath('./pytorch-pretrained-bert')
if libpath not in sys.path:
    sys.path.append(libpath)

import pytorch_pretrained_bert

OpenAIAdam = pytorch_pretrained_bert.OpenAIAdam
GPT2Tokenizer = pytorch_pretrained_bert.GPT2Tokenizer
GPT2Config = pytorch_pretrained_bert.GPT2Config
GPT2Model = pytorch_pretrained_bert.GPT2Model
