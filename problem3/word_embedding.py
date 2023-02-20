import numpy as np
from gensim import models
import pandas as pd
data=pd.read_excel('D:\美赛\data\Problem_C_Data_Wordle.xlsx')
word_list=np.array(data['Word'])
n=word_list.shape[0]
w = models.KeyedVectors.load_word2vec_format(
    'D:\美赛\problem3\GoogleNews-vectors-negative300.bin', binary=True)
def load_bin_vec(fname,vocab):
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())  # 3000000 300
        binary_len = np.dtype('float32').itemsize * layer1_size  #
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
vectors_file = 'D:\美赛\problem3\GoogleNews-vectors-negative300.bin'

vocab = ['I', 'can', 'do']
#
# vectors = load_bin_vec(vectors_file, vocab)  # pre-trained vectors
# add_unknown_words(vectors, vocab)
#
# print(vectors['I'])
#
# print('*' * 40)
#
# print(vectors['can'])
#
# print('*' * 40)
#
# print(vectors['do'])
output=np.array(w['eerie'])
output=pd.DataFrame(output)
output.to_csv('./eerie.csv')

# output=[]
# for word in word_list:
#     try:
#         output.append(w[word])
#     except KeyError:
#         output.append(np.random.uniform(-0.25, 0.25, 300))
# output=np.array(output).reshape(n,-1)
# output_=pd.DataFrame(output)
# output_.to_csv('./word_embedding.csv')
