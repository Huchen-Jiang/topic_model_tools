import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import HdpModel
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from .feature_extractor import FeatureExtractor

class TopicModelTrainer:
    def __init__(self, topic_model_method='lda', feature_extract_method='count', num_topics=10, w2v_model=None, random_state=0):
        '''
            Initializing topic model trainer instance
            
            implemented topic model methods:
                LDA: Latent Dirichlet Allocation  (sklearn.decomposition.LatentDirichletAllocation)
                NMF: Non-negative Matrix Factorization  (sklearn.decomposition.NMF)
                LSA: Latent Semantic Analysis  (sklearn.decomposition.TruncatedSVD)
                HDP: Hierarchical Dirichlet Process  (gensim.models.HdpModel)
        '''
        if topic_model_method not in ['lda', 'nmf', 'lsa', 'hdp']:
            raise Exception(f'Unknown topic model method: {topic_model_method}')
        self.topic_model = topic_model_method
        self.feature_extractor_method = feature_extract_method
        self.feature_extractor = FeatureExtractor(method=feature_extract_method, w2v_model=w2v_model)
        self.num_topics = num_topics
        self.random_state = random_state
        
    def fit(self, documents):
        features = self.feature_extractor.fit_transform(documents)
        
        if self.topic_model == 'lda':
            self.model = LatentDirichletAllocation(n_components=self.num_topics, random_state=self.random_state)
            if self.feature_extractor_method == 'word2vec':
                features = minmax_scale(np.array(features))
            self.model.fit(features)
            
        elif self.topic_model == 'nmf':
            self.model = NMF(n_components=self.num_topics, random_state=self.random_state)
            self.model.fit(features)
            
        elif self.topic_model == 'lsa':
            self.model = TruncatedSVD(n_components=self.num_topics, random_state=self.random_state)
            self.model.fit(features)
            
        elif self.topic_model == 'hdp':
            tokenized_documents = [doc.split() for doc in documents]
            dictionary = corpora.Dictionary(tokenized_documents)
            corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]
            
            self.model = HdpModel(corpus=corpus, id2word=dictionary, random_state=self.random_state)
            
        if self.topic_model == 'hdp':
            self.document_topic_distribution = self.model.inference(corpus)[0]
        else:         
            self.document_topic_distribution = self.model.transform(features)
        
        print(f'{self.topic_model} model training finished.')
        
    def get_result(self, num_words=10):
        '''
            word2vec  WIP
        '''
        topic_words = []
        if self.topic_model in ['lda', 'nmf', 'lsa']:
            if self.feature_extractor_method != 'word2vec':
                feature_names = self.feature_extractor.vectorizer.get_feature_names_out()
                for topic_id in range(self.num_topics):
                    topic_words_probs = self.model.components_[topic_id].argsort()[-num_words:][::-1]
                    topic_words.append([feature_names[i] for i in topic_words_probs])
            else:
                raise Exception('Word2Vec methods not implemented')
        elif self.topic_model == 'hdp':
            for topic_id in range(len(self.model.print_topics())):
                topic_words_probs = self.model.show_topic(topic_id, topn=num_words)
                topic_words.append([word for word, _ in topic_words_probs])
                
        return topic_words
    