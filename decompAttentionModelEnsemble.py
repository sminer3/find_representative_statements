from keras.models import load_model, Model
from keras.optimizers import Nadam, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.activations import softmax
import numpy as np
import sys
import pickle
import re
import falcon
#import requests
import ujson
matching_model1 = 'other_files/decomp_model1.h5'
matching_model2 = 'other_files/decomp_model2.h5'
matching_model3 = 'other_files/decomp_model3.h5'
matching_model4 = 'other_files/decomp_model4.h5'
matching_model5 = 'other_files/decomp_model5.h5'
with open('other_files/word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

stopwords = ['i', 'a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'what', 'when', 'where', 'who', 'will', 'with']
def preprocess(string):
    #Remove punctuation and make lower case
    string = string.lower()
    replace = str.maketrans('/\\"#$%&!()*+,-.:;<=>?@[]^_`{|}~', ' '*len('/\\"#$%&!()*+,-.:;<=>?@[]^_`{|}~'))
    string = string.translate(replace).strip()
    string = re.sub(' +',' ',string)
    return string

def is_numeric(s):
    return any(i.isdigit() for i in s)


def prepare2(s,top_words,stop_words,replace_word,max_length):
    new_s = []
    surplus_s = []
    numbers_s = []
    for w in s.split()[::-1]:
        if w in top_words:
            new_s = [w] + new_s
        else:
            if is_numeric(w):
                numbers_s = [w] + numbers_s
            else:
                surplus_s = [w] + surplus_s
        if len(new_s) == max_length:
            break
    new_s = " ".join(new_s)
    return new_s, set(surplus_s), set(numbers_s)


#NN Functions
def create_pretrained_embedding():
    "Create embedding layer from a pretrained weights array"
    in_dim, out_dim = (50747, 300)
    embedding = Embedding(in_dim, out_dim, trainable=False)
    return embedding


def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_= Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                        output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2),
                                output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def decomposable_attention(projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=500, compare_dropout=0.2,
                           dense_dim=300, dense_dropout=0.2,
                           lr=1e-3, activation='elu', maxlen=15):
    # Based on: https://arxiv.org/abs/1606.01933
    
    s1 = Input(name='s1',shape=(maxlen,))
    s2 = Input(name='s2',shape=(maxlen,))
    
    # Embedding
    embedding = create_pretrained_embedding()
    s1_embed = embedding(s1)
    s2_embed = embedding(s2)
    
    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
                Dense(projection_hidden, activation=activation),
                Dropout(rate=projection_dropout),
            ])
    projection_layers.extend([
            Dense(projection_dim, activation=None),
            Dropout(rate=projection_dropout),
        ])
    
    s1_encoded = time_distributed(s1_embed, projection_layers)
    s2_encoded = time_distributed(s2_embed, projection_layers)
    
    # Attention
    s1_aligned, s2_aligned = soft_attention_alignment(s1_encoded, s2_encoded)    
    
    # Compare
    s1_combined = Concatenate()([s1_encoded, s2_aligned, submult(s1_encoded, s2_aligned)])
    s2_combined = Concatenate()([s2_encoded, s1_aligned, submult(s2_encoded, s1_aligned)]) 
    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    s1_compare = time_distributed(s1_combined, compare_layers)
    s2_compare = time_distributed(s2_combined, compare_layers)
    
    # Aggregate
    s1_rep = apply_multiple(s1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    s2_rep = apply_multiple(s2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    
    #Features
    f_input = Input(name='features',shape=(4,))
    f_layer = BatchNormalization()(f_input)
    f_layer = Dense(50,activation=activation)(f_layer)
    f_layer = Dropout(dense_dropout)(f_layer)

    # Classifier
    merged = Concatenate()([s1_rep, s2_rep, f_layer])
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs=[s1, s2, f_input], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', 
                  metrics=['binary_crossentropy','accuracy'])
    return model

mod1 = decomposable_attention(projection_dim=200, projection_hidden=0, projection_dropout=0.2, compare_dim=200, compare_dropout=0.2, dense_dim=200, dense_dropout=0.2, lr=1e-3, activation='elu', maxlen=15)
mod1.load_weights(matching_model1)
mod2 = decomposable_attention(projection_dim=200, projection_hidden=0, projection_dropout=0.2, compare_dim=200, compare_dropout=0.2, dense_dim=200, dense_dropout=0.2, lr=1e-3, activation='elu', maxlen=15)
mod2.load_weights(matching_model2)  
mod3 = decomposable_attention(projection_dim=200, projection_hidden=0, projection_dropout=0.2, compare_dim=200, compare_dropout=0.2, dense_dim=200, dense_dropout=0.2, lr=1e-3, activation='elu', maxlen=15)
mod3.load_weights(matching_model3)  
mod4 = decomposable_attention(projection_dim=200, projection_hidden=0, projection_dropout=0.2, compare_dim=200, compare_dropout=0.2, dense_dim=200, dense_dropout=0.2, lr=1e-3, activation='elu', maxlen=15)
mod4.load_weights(matching_model4)  
mod5 = decomposable_attention(projection_dim=200, projection_hidden=0, projection_dropout=0.2, compare_dim=200, compare_dropout=0.2, dense_dim=200, dense_dropout=0.2, lr=1e-3, activation='elu', maxlen=15)
mod5.load_weights(matching_model5)    

form = {
    "statement1": "This is test big time yes",
    "statement2": "testing this statement"
}
s1 = preprocess(form['statement1'])
s2 = preprocess(form['statement2'])
w1 = s1.split()
w2 = s2.split()
w1 = [[word_index[w] for w in w1]]
w2 = [[word_index[w] for w in w2]]
w1 = np.asmatrix(pad_sequences(w1, maxlen = 15))
w2 = np.asmatrix(pad_sequences(w2, maxlen = 15))
emb = np.asmatrix([[0,0,0,0]])
p1 = mod1.predict([w1, w2, emb])[0,0]
p2 = mod2.predict([w1, w2, emb])[0,0]
p3 = mod3.predict([w1, w2, emb])[0,0]
p4 = mod4.predict([w1, w2, emb])[0,0]
p5 = mod5.predict([w1, w2, emb])[0,0]


class processMatch(object):
    def on_post(self, req, resp):
        form = ujson.loads(req.stream.read())
        if 'statement1' in form and 'statement2' in form:
            try:
                s1 = preprocess(form['statement1'])
                s2 = preprocess(form['statement2'])
                s1, surp1, num1 = prepare2(s1,word_index.keys(),stopwords, 'chebychev', 15)
                s2, surp2, num2 = prepare2(s2,word_index.keys(),stopwords, 'chebychev', 15)
                w1 = s1.split()
                w2 = s2.split()
                w1 = [[word_index[w] for w in w1]]
                w2 = [[word_index[w] for w in w2]]
                w1 = np.asmatrix(pad_sequences(w1, maxlen = 15))
                w2 = np.asmatrix(pad_sequences(w2, maxlen = 15))
                rwh = np.asmatrix([[len(surp1.intersection(surp2)), len(surp1.union(surp2)), len(num1.intersection(num2)), len(num1.union(num2))]])
                prob1 = mod1.predict([np.asmatrix(w1), np.asmatrix(w2), np.asmatrix(rwh)])[0,0]
                print(prob1)
                prob2 = mod2.predict([np.asmatrix(w1), np.asmatrix(w2), np.asmatrix(rwh)])[0,0]
                print(prob2)
                prob3 = mod3.predict([np.asmatrix(w1), np.asmatrix(w2), np.asmatrix(rwh)])[0,0]
                print(prob3)
                prob4 = mod4.predict([np.asmatrix(w1), np.asmatrix(w2), np.asmatrix(rwh)])[0,0]
                print(prob4)
                prob5 = mod5.predict([np.asmatrix(w1), np.asmatrix(w2), np.asmatrix(rwh)])[0,0]
                print(prob5)
                print(np.mean([prob1, prob2, prob3, prob4, prob5]))
                form['matchProbability'] = np.asscalar(np.mean([prob1, prob2, prob3, prob4, prob5]))
                resp.body = ujson.dumps(form)
                resp.status = falcon.HTTP_200
            except:
                resp.body = ujson.dumps({'Error': 'An internal server error has occurred'})
                resp.status = falcon.HTTP_500
        else:
            resp.body = ujson.dumps({'Error': 'Input is incorrect', 'Request': form})
            resp.status = falcon.HTTP_400
# instantiate a callable WSGI app
app = falcon.API()
# long-lived resource class instance
decompAttentionModelEnsemble = processMatch()
# handle all requests to the '/inferreview' URL path
app.req_options.auto_parse_form_urlencoded = False
app.add_route('/decompAttentionModelEnsemble', decompAttentionModelEnsemble)