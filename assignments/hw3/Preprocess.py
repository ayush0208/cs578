import pandas as pd
import numpy as np
import string
import re
import nltk.stem
from nltk.tokenize import sent_tokenize, word_tokenize

def remove_between_square_brackets(text):
  return re.sub('\[[^]]*\]', '', text)

def remove_special_characters(text, remove_digits=True):
  pattern=r'[^a-zA-z0-9\s]'
  text=re.sub(pattern,'',text) 
  return text

def tokenize_text(text):
    return [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]

def replace_text_stemming(text):
    stemmer = nltk.stem.PorterStemmer()
    stems = [stemmer.stem(word) for word in tokenize_text(text)]
    return " ".join(stems)
    
def get_features_labels(df, label_column_name):
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', 1, inplace=True)
    labels = df[label_column_name]
    train_features = df.drop(label_column_name, 1)
    return train_features, labels.to_numpy()

def generate_N_grams(ngram, dataset):
    text_corpus = ""
    for i in range(0, dataset.shape[0]):
        text_corpus += " " + dataset.iloc[i]['Text']
    text_corpus = text_corpus.lower()
    text_corpus = remove_between_square_brackets(text_corpus)
    text_corpus = remove_special_characters(text_corpus, True)
    stemmed_corpus = replace_text_stemming(text_corpus)
    token=word_tokenize(stemmed_corpus)
    stopwords_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    token = [tok for tok in token if tok not in stopwords_list] 
    token = [tok for tok in token if tok not in string.punctuation]
    temp=zip(*[token[i:] for i in range(0,ngram)])
    ans=[' '.join(ngram) for ngram in temp]
    return ans

def get_word_count_n_gram(n_gram_token_list):
    word_count = {}
    for token in n_gram_token_list:
        if token in word_count:
            word_count[token]+=1
        elif token not in word_count:
            word_count[token]=1
    return word_count

def generate_N_grams_for_sent(ngram, sent):
    text_corpus = sent
    text_corpus = text_corpus.lower()
    text_corpus = remove_between_square_brackets(text_corpus)
    text_corpus = remove_special_characters(text_corpus, True)
    stemmed_corpus = replace_text_stemming(text_corpus)
    token=word_tokenize(stemmed_corpus)
    stopwords_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    token = [tok for tok in token if tok not in stopwords_list] 
    token = [tok for tok in token if tok not in string.punctuation]
    temp=zip(*[token[i:] for i in range(0,ngram)])
    ans=[' '.join(ngram) for ngram in temp]
    return ans

def get_vector_n_gram(text, vocabulary, n_gram_value):
    tokens = generate_N_grams_for_sent(n_gram_value, text)
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        if token in vocabulary:
            vector[list(vocabulary.keys()).index(token)]+= 1
    return vector

def build_vocab_n_gram(dataset, n_gram_value):
    n_gram_tokens = generate_N_grams(n_gram_value, dataset)
    vocab_n_gram_value = get_word_count_n_gram(n_gram_tokens)
    updated_vocab_n_gram_value = {}
    min_freq = 0
    max_freq = 0
    if(n_gram_value==1):
        min_freq=17
        max_freq=20000
    if(n_gram_value==2):
        min_freq=5
        max_freq=200
    if(n_gram_value==3):
        min_freq=2
        max_freq=200
    for key in vocab_n_gram_value.keys():
        if(vocab_n_gram_value[key]>min_freq and vocab_n_gram_value[key]<max_freq):
            updated_vocab_n_gram_value[key] = vocab_n_gram_value[key]
    return updated_vocab_n_gram_value