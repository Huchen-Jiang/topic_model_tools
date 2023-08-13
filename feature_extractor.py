import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class FeatureExtractor:
    '''
        Feature Extractor module
        
        extract features with methods like:
            CountVectorizer, TfidfVectorizer, Word2Vec
    '''
    
    def __init__(self, method='count', w2v_model=None):
        '''
            Initializing feature extractor instance
        '''
        self.method = method
        self.vectorizer = None
        self.w2v_model = w2v_model
        
        if self.method == 'count':
            print('Using CountVectorizer as feature extractor.')
            self.vectorizer = CountVectorizer()
        elif self.method == 'tfidf':
            print('Using TfidfVectorizer as feature extractor.')
            self.vectorizer = TfidfVectorizer()
        elif self.method == 'word2vec':
            print('Using Word2Vec as feature extractor.')
            if self.w2v_model is None:
                print('No w2v model provided')
                print('Please call FeatureExtractor.train_w2v_model to train a w2v model.')
        else:
            raise Exception(f'Unknown vectorizing_method: {vectorizing_method}')
        
    def train_w2v_model(self, documents=None, vector_size=100, window=5, min_count=1, workers=4):
        '''
            method to train a w2v model
        '''
        self.w2v_model = train_w2v_model_(documents, vector_size, window, min_count, workers)
    
    def fit_transform(self, documents):
        '''
            extract features
        '''
        if self.method == 'count' or self.method == 'tfidf':
            self.vecs = self.vectorizer.fit_transform(documents)
            self.feature_names = self.vectorizer.get_feature_names_out()
        elif self.method == 'word2vec':
            if self.w2v_model is None:
                print('a w2v model will be trained with default parameters.')
                self.w2v_model = self.train_w2v_model(documents)
            self.vecs = np.array(
                [np.mean([self.w2v_model[word] for word in doc.split() if word in self.w2v_model]
                                   or [np.zeros(self.w2v_model.vector_size)], axis=0)
                          for doc in preprocessed_documents]
            )
        return self.vecs
            

    
    
def train_w2v_model_(documents=None, vector_size=100, window=5, min_count=1, workers=4):
    '''
        helper function to train a w2v model
    '''
    if documents is None:
            raise ValueError("No documents provided for training Word2Vec.")
        
    sentences = [doc.split() for doc in documents]
    w2v_model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window,
                                min_count=min_count, workers=workers)
    
    return w2v_model
    
    
    
    