from collections import defaultdict
from bert4keras.tokenizers import Tokenizer, load_vocab
import numpy as np
from .finetune_config import cfg
import mindspore.dataset.engine.datasets as de
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype

def create_tokenizer():
  dict_path = cfg.dict_path

  length = 128
  # disallowed words
  disallowed_words = cfg.disallowed_words
  # max sequence length
  max_len = cfg.max_len
  # min word frequency 
  min_word_frequency = cfg.min_word_frequency
  # mini batch 
  batch_size = cfg.batch_size

  # load dataset
  with open(cfg.dataset_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [line.replace('：', ':') for line in lines]

  # dataset list
  poetry = []
  # process by line
  for line in lines:
    if line.count(':') != 1:
        continue
    __, last_part = line.split(':')
    ignore_flag = False
    for dis_word in disallowed_words:
        if dis_word in last_part:
            ignore_flag = True
            break
    if ignore_flag:
        continue
    # sequence length limit
    if len(last_part) > max_len - 2:
        continue
    poetry.append(last_part)

  _token_dict = load_vocab(dict_path)
  _tokenizer = Tokenizer(dict_path, do_lower_case=True)

  # calculate word frequency
  word_frequency_count = defaultdict(int)
  for line in poetry:
    for t in _tokenizer.tokenize(line):
        word_frequency_count[t] += 1
  # filter low-frequency word
  tokens = [(token, count) for token, count in word_frequency_count.items() if count >= min_word_frequency]
  # sort by frequency
  tokens = sorted(tokens, key=lambda x: -x[1])

  tokens = [token for token, count in tokens]

  token_id_dict = {}
  keep_words = []

  for token in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
    token_id_dict[token] = len(token_id_dict)
    keep_words.append(_token_dict[token])
  
  for token in tokens:
    if token in _token_dict and token not in token_id_dict:
        token_id_dict[token] = len(token_id_dict)
        keep_words.append(_token_dict[token])

  # create tokenizer
  tokenizer = Tokenizer(token_id_dict, do_lower_case=True)
  # data shuffling
  np.random.shuffle(poetry)

  print(len(poetry))
  print(len(keep_words))
  return poetry, tokenizer, keep_words

def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs)]
    x = inputs
    x = x[:length]
    pad_width[0] = (0, length - len(x))
    x = np.pad(x, pad_width, 'constant', constant_values=padding)
    return x


class PoetryDataGenerator():
    """
    数据生成器
    """

    def __init__(self, batch_size, poetry, tokenizer, length=128):
        self.data = poetry
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.length = length

    def __getitem__(self, index, random=False):
        
        if random:
            np.random.shuffle(self.data)

        total = len(self.data)
        single_data = self.data[index]

        token_ids, segment_ids = self.tokenizer.encode(single_data)
        batch_token_ids = sequence_padding(token_ids, length=self.length)
        batch_segment_ids = sequence_padding(segment_ids, length=self.length)
        pad_mask = (batch_token_ids != 0).astype(np.float32)
        return (batch_token_ids, batch_segment_ids, pad_mask)

    def __len__(self):
        return len(self.data)

def create_poetry_dataset(batch_size, poetry, tokenizer):
    dt = PoetryDataGenerator(batch_size, poetry, tokenizer)
    ds = de.GeneratorDataset(dt, ["input_ids", "token_type_id", "pad_mask"])
    ds.set_dataset_size(dt.__len__())
    int_type_cast_op = C.TypeCast(mstype.int32)
    float_type_cast_op = C.TypeCast(mstype.float32)
    ds = ds.map(input_columns="input_ids", operations=int_type_cast_op)
    ds = ds.map(input_columns="token_type_id", operations=int_type_cast_op)
    ds = ds.map(input_columns="pad_mask", operations=float_type_cast_op)
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds
