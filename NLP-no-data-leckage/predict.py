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



with open('utils/read_data.pickle', 'rb') as f:
    loaded_data = dill.load(f)

with open('utils/cleaned_token_list.pickle','rb') as f:
    loaded_cleaned_token = dill.load(f)

with open('utils/list_to_dict.pickle','rb') as f:
    list_to_dict = dill.load(f)

with open('utils/remove_noise.pickle','rb') as f:
    remove_noise = dill.load(f)

with open('utils/cleared.pickle','rb') as f:
    cleared = dill.load(f)

with open('utils/sentence_to_indices.pickle','rb') as f:
    sentence_to_indices = dill.load(f)

with open('utils/read_glove_vecs.pickle','rb') as f:
    loaded_read_glove = dill.load(f)

words_to_index,index_to_words,word_to_vec_map  = loaded_read_glove[0],loaded_read_glove[1],loaded_read_glove[2] 

with open('utils/cleaned.pickle','rb') as f:
    cleaned = dill.load(f)

with open('utils/max_len.pickle','rb') as f:
    max_len = dill.load(f)

df = pureml.dataset.fetch(label='nlpexample_test:development:v1')
x_train = df['x_train']
y_train = df['y_train']
x_test = df['x_test']
y_test = df['y_test']
data = loaded_data(x_test,y_test)
cleaned_token_list = loaded_cleaned_token(data)
X = np.zeros((len(cleaned_token_list), max_len))
Y = np.zeros((len(cleaned_token_list), ))
print(f"X & Y Created. {len(X)} & {len(Y)}")
for i, tk_lb in enumerate(cleaned_token_list):
    tokens, label = tk_lb
    sentence_to_indices(tokens, words_to_index, max_len, i,X)
    Y[i] = label
print(f"{len(X)} & {len(Y)}")
x_test = X
y_test = Y

# data = np.core.records.fromarrays([x_test, y_test], names=' x_test, y_test')



# data = {
#     'x_test' : x_test,
#     'y_test': y_test
# }
# @dataset(label='nlpexample_test:development',upload = True)
# def create_dataset():
#     return {'x_train':x_train,'y_train':y_train,'x_test':x_test,'y_test':y_test}

# create_dataset()

# data = pureml.dataset.fetch(label='nlpexample_test:development:v2')

class Predictor(BasePredictor):
    label = 'nlpexample_test:model:v1'
    input = Input(type = 'numpy ndarray')
    output = Output(type = 'numpy ndarray')

    def load_models(self):
        self.model = pureml.model.fetch(self.label)
        
    def predict(self, data):
        
        prediction = self.model.predict(data)
        threshold = 0.4
        prediction = np.where(prediction > threshold,1,0)
        prediction = np.squeeze(prediction)
        return prediction

# p = Predictor()
# p.load_models()
# y_pred = p.predict(x_test)

# # y_test = data['y_test']

# union_data = set(y_pred).union(y_test)
# print(union_data)

# score = accuracy_score(y_pred,y_test)
# print(score)

# print(y_test.shape)
# print(x_test.shape)