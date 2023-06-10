import pickle
import pureml
from pureml.decorators import dataset,load_data,transformer,model
import pandas as pd
import numpy as np
from nltk import WordNetLemmatizer, pos_tag
import re, string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
STOP_WORDS = stopwords.words('english')
from pureml import BasePredictor,Input,Output
import pureml
import dill
from sklearn.metrics import accuracy_score
import typing 
from nltk.tokenize import TweetTokenizer



class Predictor(BasePredictor):
    label = 'nlpexample_test:model:v3'
    input = Input(type = 'numpy ndarray')
    output = Output(type = 'numpy ndarray')
    word_to_vec_map:typing.Any = None
    words_to_index:typing.Any = None
    index_to_words:typing.Any = None
    max_len:typing.Any = None

    def load_pre_processor(self):

        with open('utils/read_glove_vecs.pickle','rb') as f:
            loaded_read_glove = dill.load(f)
            self.words_to_index,self.index_to_words,self.word_to_vec_map  = loaded_read_glove[0],loaded_read_glove[1],loaded_read_glove[2] 
     
        with open('utils/max_len.pickle','rb') as f:
            self.max_len = dill.load(f)

    def cleaned(self,token):
        if token == 'u':
            return 'you'
        if token == 'r':
            return 'are'
        if token == 'some1':
            return 'someone'
        if token == 'yrs':
            return 'years'
        if token == 'hrs':
            return 'hours'
        if token == 'mins':
            return 'minutes'
        if token == 'secs':
            return 'seconds'
        if token == 'pls' or token == 'plz':
            return 'please'
        if token == '2morow':
            return 'tomorrow'
        if token == '2day':
            return 'today'
        if token == '4got' or token == '4gotten':
            return 'forget'
        if token == 'amp' or token == 'quot' or token == 'lt' or token == 'gt' or token == '½25':
            return ''
        return token
 
    def list_to_dict(self,cleaned_tokens):
        return dict([token, True] for token in cleaned_tokens)

    def remove_noise(self,tweet_tokens):
        cleaned_tokens = []
        for token, tag in pos_tag(tweet_tokens):
            # Eliminating the token if it is a link
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            # Eliminating the token if it is a mention
            token = re.sub("(@[A-Za-z0-9_]+)","", token)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            cleaned_token = self.cleaned(token.lower())
            if cleaned_token not in string.punctuation and len(cleaned_token) > 2 and cleaned_token not in STOP_WORDS:
                cleaned_tokens.append(cleaned_token)
        return cleaned_tokens
    
    def cleaned_token_list(self,data):
        final_data = []
        for tokens, label in data:
            final_data.append((self.list_to_dict(tokens), label))
        print("Final Data created")
        cleaned_tokens_list = []
        for tokens, label in final_data:
            cleaned_tokens_list.append((self.remove_noise(tokens), label))
        return cleaned_tokens_list

    def read_data(self,x_train):
        tk = TweetTokenizer(reduce_len=True)
        da = []
        x_train = x_train.tolist()
        for i,x in enumerate(x_train):
            da.append((tk.tokenize(x),i))
        return da
    
    def cleared(self,word):
        res = ""
        prev = None
        for char in word:
            if char == prev: continue
            prev = char
            res += char
        return res
    
    def sentence_to_indices(self,sentence_words, word_to_index, max_len, i,X):
        unks = []
        UNKS = []
        sentence_indices = []
        for j, w in enumerate(sentence_words):
            try:
                index = self.words_to_index[w]
            except:
                UNKS.append(w)
                w = self.cleared(w)
                try:
                    index = self.words_to_index[w]
                except:
                    index = self.words_to_index['unk']
                    unks.append(w)
            X[i, j] = index
    
 

    def x_test_transformation(self,data):
        data = self.read_data(data)
        cleaned_token_list = self.cleaned_token_list(data)
        x = np.zeros((len(cleaned_token_list), self.max_len))
        for i, tk_lb in enumerate(cleaned_token_list):
            tokens, label = tk_lb
            self.sentence_to_indices(tokens, self.words_to_index, self.max_len, i,x)        
        return x
        

    def load_models(self):
        self.model = pureml.model.fetch(self.label)
        self.load_pre_processor()
        
    def predict(self, data):
        
        x_test = self.x_test_transformation(data)
        prediction = self.model.predict(x_test)
        threshold = 0.99995
        prediction = np.where(prediction > threshold,1,0)
        prediction = np.squeeze(prediction)
        return prediction


# p = Predictor()
# p.load_models()
# y_pred = p.predict(x_test)
# print(y_pred)

# y_test = d['y_test']

# union_data = set(y_pred).union(y_test)
# print(union_data)

# score = accuracy_score(y_pred,y_test)
# print(score)

# print(y_test.shape)
