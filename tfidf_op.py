import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import warnings
import jieba
import re

# 创建停用词列表
def get_stopwords_list():
    stopwords = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]
    stopwords.append('（')
    stopwords.append('）')
    return stopwords

# 对句子进行中文分词
def seg_depart(sentence):
    sentence_depart = jieba.lcut(sentence.strip())
    return sentence_depart

def move_stopwords(sentence_list, stopwords_list):
    # 去停用词
    out_list = []
    for word in sentence_list:
        if word not in stopwords_list:
            if word != '\t':
                out_list.append(word)
    return ' '.join(out_list)

def get_cut_list(x):
    sentence_depart = seg_depart(x)
    sentence_depart = move_stopwords(sentence_depart, stopwords)
    return sentence_depart

warnings.filterwarnings('ignore')
stopwords = get_stopwords_list()

base = pd.read_csv('./data/train/base_info.csv')
label = pd.read_csv('./data/train/entprise_info.csv')
base = pd.merge(base, label, on=['id'], how='left')

base['oploc_list'] = base['oploc'].apply(lambda x: ' '.join([x[16 * i:16 * (i + 1)] for i in range(int(len(x) / 16))]))
base['dom_list'] = base['dom'].apply(lambda x: ' '.join([x[16 * i:16 * (i + 1)] for i in range(int(len(x) / 16))]))
base['opscope_word_list'] = base['opscope'].apply(get_cut_list)

oploc__tfidf_vector = TfidfVectorizer(min_df=30).fit(
    base['oploc_list'].tolist())
dom__tfidf_vector = TfidfVectorizer(min_df=30).fit(
    base['dom_list'].tolist())
opscope_tfidf_vector = TfidfVectorizer(min_df=30).fit(
    base['opscope_word_list'].tolist())

data = base[['id', 'oploc_list', 'dom_list', 'opscope_word_list', 'label']]
def create_csr_mat_input(oploc_list, dom_list, opscope_word_list):
    return sp.hstack((oploc__tfidf_vector.transform(oploc_list),
                      dom__tfidf_vector.transform(dom_list),
                      opscope_tfidf_vector.transform(opscope_word_list)),
                     format='csr')

tfidf_input = create_csr_mat_input(data['oploc_list'], data['dom_list'], data['opscope_word_list'])
result = pd.DataFrame({'id': data['id']})

lda = LatentDirichletAllocation(n_jobs=-1,
                                random_state=2020,
                                n_components=16)
result[[
    f'lda_{i + 1}' for i in range(lda.n_components)
]] = pd.DataFrame(lda.fit_transform(
    tfidf_input), index=result.index)

nmf = NMF(random_state=2020, n_components=16)
result[[
    f'nmf_{i + 1}' for i in range(nmf.n_components)
]] = pd.DataFrame(nmf.fit_transform(
    tfidf_input),
    index=result.index)

svd = TruncatedSVD(random_state=2020,
                   n_components=32)
result[[
    f'svd_{i + 1}' for i in range(svd.n_components)
]] = pd.DataFrame(svd.fit_transform(
    tfidf_input),
    index=result.index)

result.to_csv('tfidf_decomposition.csv', index=False)
