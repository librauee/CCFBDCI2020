import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import warnings
import jieba
import nltk
from nltk.corpus import stopwords

warnings.filterwarnings('ignore')
nltk.download('stopwords')
stop_words = list(stopwords.words('chinese'))

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


def processed_text_list(x):
    sentence_depart = seg_depart(x)
    sentence_depart = move_stopwords(sentence_depart, stop_words)
    return sentence_depart


base = pd.read_csv('./train/base_info.csv')
label = pd.read_csv('./train/entprise_info.csv')
base = pd.merge(base, label, on=['id'], how='left')

base['oploc_list'] = base['oploc'].apply(lambda x: ' '.join([x[16 * i:16 * (i + 1)] for i in range(int(len(x) / 16))]))
base['dom_list'] = base['dom'].apply(lambda x: ' '.join([x[16 * i:16 * (i + 1)] for i in range(int(len(x) / 16))]))
base['opscope_word_list'] = base['opscope'].apply(processed_text_list)

oploc__tfidf_vector = TfidfVectorizer(min_df=40).fit(
    base['oploc_list'].tolist())
dom__tfidf_vector = TfidfVectorizer(min_df=40).fit(
    base['dom_list'].tolist())
opscope_tfidf_vector = TfidfVectorizer(min_df=40).fit(
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
                                n_components=18)
result[[
    f'lda_{i + 1}' for i in range(lda.n_components)
]] = pd.DataFrame(lda.fit_transform(
    tfidf_input), index=result.index)

nmf = NMF(random_state=2020, n_components=18)
result[[
    f'nmf_{i + 1}' for i in range(nmf.n_components)
]] = pd.DataFrame(nmf.fit_transform(
    tfidf_input),
    index=result.index)

svd = TruncatedSVD(random_state=2020,
                   n_components=36)
result[[
    f'svd_{i + 1}' for i in range(svd.n_components)
]] = pd.DataFrame(svd.fit_transform(
    tfidf_input),
    index=result.index)

result.to_csv('tfidf_decomposition.csv', index=False)
