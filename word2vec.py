# -*- coding: utf-8 -*-

import os
from gensim.models import Word2Vec

def read_file(filepath):
    if not os.path.isfile(filepath):
        return []
    sentences = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                sentences.append(line.split())
        return sentences
    except:
        return sentences

def read_dir(folder):
    if not os.path.isdir(folder):
        return None
    try:
        files = os.listdir(folder)
    except:
        return None

    sentences = None

    for file in files:
        filepath = os.path.join(folder, file)
        if not os.path.isfile(filepath):
            continue
        retval = read_file(filepath)
        if len(retval) == 0:
            continue
        if sentences is None:
            sentences = retval
        else:
            sentences += retval

    return sentences

def word2vec_train(data):
    if os.path.isdir(data):
        sentences = load_dir(data)
    else:
        sentences = load_file(data)

    if sentences is None or len(sentences) == 0:
        return None

    return Word2Vec(
        sentences,
        size=100,
        window=5,
        min_count=1,
        workers=4
    )

def word2vec_test(model, word):
    if nodel is None:
        return None
    word = word.strip()
    if not word:
        return None

    vector = model.wv[word]
    print(vector)
    words = model.wv.most_similar(word)
    print(words)
