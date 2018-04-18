#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:53:23 2018

@author: abhimanyu
"""

import numpy as np
from Model import Model
class Config(object):
    dir_output = "output/test/"
    dir_model  = dir_output + "model.weights/"
    dim_word = 50
    dim_char = 100
    filename_glove = "data//glove.6B.50d.txt"
    filename_trimmed = "data/glove.6B.50d.trimmed.npz"
    use_pretrained = True
    filename_dev  = filename_train = "data/ner.txt" 
    filename_test = "data/test.txt"
    max_iter = None 
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    train_embeddings = False
    nepochs          = 30
    dropout          = 0.5
    batch_size       = 20
    lr               = 0.001
    nepoch_no_imprv  = 3
    hidden_size_lstm = 300 
    vocab_words=[]
    vocab_tags = []
    nwords     = 0
    ntags      = 0
    unk = "<unk>"
    num = "<num>"
    none = "O"
    rewrite_embeddings = False


class Dataset(object):
    def __init__(self, filename, processing_word=None, processing_tag=None, max_iter=None):
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None
    def __iter__(self):
        niter = 0
        with open(self.filename, encoding='"ISO-8859-1') as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
#                print(line)
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0],ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]
    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length
    
#config = Config()


def load_vocab(filename):
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx
    return d


def get_processing_word(config, vocab_words=None,  lowercase=False, chars=False, allow_unk=True):
    def f(word):

        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = config.num

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[config.unk]
                else:
                    raise Exception("Unknow key is not allowed. Check that your vocab (tags?) is correct")

        return word

    return f


def get_trimmed_glove_vectors(filename):
    with np.load(filename) as data:
        return data["embeddings"]
    
def get_glove_vocab(filename):
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def get_vocabs(datasets):
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    f = open('test.txt','w')
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
            f.write(' '.join(words)+"\n")
            
    f.close()
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags
 

def write_vocab(vocab, filename):
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))
    

def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(trimmed_filename, embeddings=embeddings)

    
def align_data(data):
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned




config = Config()
processing_word = get_processing_word(config, lowercase=True)

dev   = Dataset(config.filename_dev, processing_word)
test  = Dataset(config.filename_test, processing_word)
train = Dataset(config.filename_train, processing_word)

config.vocab_words, config.vocab_tags = get_vocabs([train, dev, test])

vocab_glove = get_glove_vocab(config.filename_glove)
vocab = config.vocab_words & vocab_glove
vocab.add(config.unk)
vocab.add(config.num)
write_vocab(vocab, config.filename_words)
write_vocab(config.vocab_tags, config.filename_tags)
vocab = load_vocab(config.filename_words)
if config.rewrite_embeddings:
    export_trimmed_glove_vectors(vocab, config.filename_glove,config.filename_trimmed, config.dim_word)
train = Dataset(config.filename_train)
config.vocab_words = load_vocab(config.filename_words)
config.vocab_tags  = load_vocab(config.filename_tags)
config.nwords     = len(config.vocab_words)
config.ntags      = len(config.vocab_tags)
config.processing_word = get_processing_word(config, config.vocab_words,  lowercase=True)
config.processing_tag  = get_processing_word(config, config.vocab_tags,  lowercase=False, allow_unk=False)
config.embeddings = (get_trimmed_glove_vectors(config.filename_trimmed) if config.use_pretrained else None)

model = Model(config)
model.build()
dev   = Dataset(config.filename_dev, config.processing_word,
                     config.processing_tag, config.max_iter)
train = Dataset(config.filename_train, config.processing_word,
                     config.processing_tag, config.max_iter)

# train model
model.train(train, dev)


# testmodel
testmodel = Model(config)
testmodel.build()
testmodel.restore_session(config.dir_model)

test  = Dataset(config.filename_test, config.processing_word,
                     config.processing_tag, config.max_iter)

model.evaluate(test)


