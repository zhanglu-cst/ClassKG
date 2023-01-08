import json
import logging
import os

from gensim.models import word2vec

from PROJECT_ROOT import ROOT_DIR
from keyword_sentence.sentence_process import split_sentence_into_words

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)

task = 'SMS'

files = ['labeled.json', 'unlabeled.json', 'test.json', 'val.json']
target_files = 'word_to_emb.json'
target_path = os.path.join(ROOT_DIR, 'data', 'processed', task, target_files)

raw_sentence = []
for file in files:
    path = os.path.join(ROOT_DIR, 'data', 'processed', task, file)
    with open(path, 'r') as f:
        data = json.load(f)
        sentences, _ = map(list, zip(*data))
        raw_sentence += sentences

print('len sentence:{}'.format(len(raw_sentence)))
words_set = set()
sentences = []
for s in raw_sentence:
    words = split_sentence_into_words(s)
    sentences.append(words)
    words_set.update(words)

print('number words set:{}'.format(len(words_set)))
print('training')
model = word2vec.Word2Vec(sentences, min_count = 1)
print('finish training')

word_to_emb = {}
for word in words_set:
    emb = model[word].tolist()
    word_to_emb[word] = emb
# print(word_to_emb)
with open(target_path, 'w') as f:
    json.dump(word_to_emb, f)


print('1.50' in word_to_emb)

# print(model.similarity(''))


#
# raw_sentences = ["the quick brown fox jumps over the lazy dogs", "yoyoyo you go home now to sleep"]
#
# # 切分词汇
# sentences = [s.split() for s in raw_sentences]
# print(sentences)
# # 构建模型
# model = word2vec.Word2Vec(sentences, min_count = 1, )
#
# # 进行相关性比较
# print(model.similarity('dogs', 'you'))
# print(model['the'].shape)
