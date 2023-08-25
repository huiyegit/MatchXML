import numpy as np
import re
from  tqdm import tqdm
from scipy.sparse import csr_matrix
import os
from multiprocessing import Pool, cpu_count,Manager
import multiprocessing
from sentence_transformers import SentenceTransformer
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from tqdm.contrib.concurrent import process_map

from sklearn.preprocessing import normalize as sk_normalize
import scipy.sparse as smat
import smat_util


def concat_features(X_feat, X_emb, normalize_emb=True):
    """Concatenate instance numerical features with transformer embeddings

    Args:
        X_feat (csr_matrix or ndarray): instance numerical features of shape (nr_inst, nr_features)
        X_emb (ndarray): instance embeddings of shape (nr_inst, hidden_dim)
        normalize_emb (bool, optional): if True, rowwise normalize X_emb before concatenate.
            Default False

    Returns:
        X_cat (csr_matrix or ndarray): the concatenated features
    """
    if normalize_emb:
        X_cat = sk_normalize(X_emb)
    else:
        X_cat = X_emb

    if isinstance(X_feat, smat.csr_matrix):
        X_cat = smat_util.dense_to_csr(X_cat)
        X_cat = smat_util.hstack_csr([X_feat, X_cat], dtype=np.float32)
    elif isinstance(X_feat, np.ndarray):
        X_cat = np.hstack([X_feat, X_cat])
    elif X_feat is None:
        pass
    else:
        raise TypeError(f"Expected CSR or ndarray, got {type(X_feat)}")
    return X_cat

class Embedding():
    def __init__(self):
        self.tokenizer = self.get_tokenizer(model)
        
    def get_tokenizer(self,model_name):
        """
        get tokenizer
        param:model_name
        """
        if 'T5' in model_name:
            print('loading T5 tokenizer')
            tokenizer = SentenceTransformer('sentence-transformers/sentence-t5-base')
        if 'MiniLM' in model_name:
            tokenizer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return tokenizer

    def read_dataset(self,dataset):  
        """
        param:dataset
        """
        train_texts, test_texts = [], []

        with open(f'./dataset/{dataset}/train_raw_texts.txt',encoding='utf-8') as f:
            for point in tqdm(f.readlines()):
                point = point.replace('\n', ' ')
                point = point.replace('_', ' ')
                point = re.sub(r"\s{2,}", " ", point)
                point = re.sub("/SEP/", "[SEP]", point)
                train_texts.append(point)

        with open(f'./dataset/{dataset}/test_raw_texts.txt',encoding='utf-8') as f:
            for point in tqdm(f.readlines()):
                point = point.replace('\n', ' ')
                point = point.replace('_', ' ')
                point = re.sub(r"\s{2,}", " ", point)
                point = re.sub("/SEP/", "[SEP]", point)
                test_texts.append(point)

        return train_texts, test_texts

    def make_csr_tfidf(self,dataset, LF_data):
        """
        read tfidf
        param:dataset
        """
        file_name_train = f'./dataset/{dataset}/X.trn.npz'
        file_name_test = f'./dataset/{dataset}/X.tst.npz'
        if os.path.exists(file_name_train):
            print(f"Loading {file_name_train}")
            tfidf_mat_train = sp.load_npz(file_name_train)
        if os.path.exists(file_name_test):
            print(f"Loading {file_name_test}")
            tfidf_mat_test = sp.load_npz(file_name_test)
        return tfidf_mat_train,tfidf_mat_test

    def encode(self,text): 
        """
        param:text
        """   
        encodings = self.tokenizer.encode(text)
        return encodings

    def create_data(self,dataset, model, LF_data=False):
        """
        param:dataset
        parm:model
        """   
        print(f"Creating new data for {model} model")
        train_texts, test_texts = self.read_dataset(dataset)
        print(f"Available CPU Count is: {cpu_count()}")
        #train_texts=train_texts[0:100]
        os.makedirs(f'{dataset}/{model}', exist_ok=True)
        # print(encode(train_texts[0:2]))
        with Pool(1) as p:
            encoded_train = process_map(self.encode, train_texts, max_workers=1, chunksize=500)
        with Pool(1) as p:
            encoded_test = process_map(self.encode, test_texts, max_workers=1, chunksize=500)
        return encoded_train,encoded_test
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    # model="MiniLM"
    model = "T5"
    # dataset="wiki-500k"
    dataset = "wiki10-31k"
    eb=Embedding()

    encoded_train,encoded_test=eb.create_data(dataset,model)
    encoded_train = np.stack(encoded_train, axis=0)
    encoded_test = np.stack(encoded_test, axis=0)

    encoded_train = sk_normalize(encoded_train)
    encoded_test = sk_normalize(encoded_test)

    file_name_train = f'./xmc-base/{dataset}/tfidf-attnxml/dense_train' + model + '.npy'
    np.save(file_name_train, encoded_train)
    file_name_test = f'./xmc-base/{dataset}/tfidf-attnxml/dense_test' + model + '.npy'
    np.save(file_name_test, encoded_test)

    # encoded_train = np.load(file_name_train)
    # encoded_test = np.load(file_name_test)

    tfidf_mat_train,tfidf_mat_test =eb.make_csr_tfidf(dataset,LF_data=True)
    tfidf_mat_train= normalize(tfidf_mat_train, norm='l2')
    tfidf_mat_test=normalize(tfidf_mat_test, norm='l2')
    print(tfidf_mat_train.shape)

    allmatrix_sp = concat_features(tfidf_mat_train, encoded_train, normalize_emb=True,)
    allmatrix_sp = sk_normalize(allmatrix_sp)
    print(allmatrix_sp.shape)
    file_name_train = f'./xmc-base/{dataset}/tfidf-attnxml/allmatrix_sparse_train' + model + '.npz'
    sp.save_npz(file_name_train, allmatrix_sp)

    allmatrix_sp = concat_features(tfidf_mat_test, encoded_test, normalize_emb=True, )
    allmatrix_sp = sk_normalize(allmatrix_sp)
    print(allmatrix_sp.shape)
    file_name_test = f'./xmc-base/{dataset}/tfidf-attnxml/allmatrix_sparse_test' + model + '.npz'
    sp.save_npz(file_name_test, allmatrix_sp)

