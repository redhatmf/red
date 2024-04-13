import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

corpus_raw = corpus_raw = 'He is the king . The king is royal . She is the royal queen . The queen is beautiful . He is the royal prince '

# convert to lower case
corpus_raw = corpus_raw.lower()

words = []
for word in corpus_raw.split():
    if word != '.': # because we don't want to treat . as a word
        words.append(word)

words = set(words) # so that all duplicate words are removed
word2int = {}
int2word = {}
vocab_size = len(words) # gives the total number of unique words

for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

# raw sentences is a list of sentences.
raw_sentences = corpus_raw.split('.')
sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())

WINDOW_SIZE = 2

data = []
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                data.append([word, nb_word])

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

x_train = [] # input word
y_train = [] # output word

for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))

# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

EMBEDDING_DIM = 5 # you can choose your own number

n_iters=5000

model = Sequential([
    Dense(EMBEDDING_DIM, input_shape=(vocab_size,)),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=n_iters)
W,b=model.layers[0].get_weights()
vectors=W+b
print(vectors)

'''from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0,perplexity=5)
np.set_printoptions(suppress=True)
vectors_red = tsne.fit_transform(vectors) '''

from sklearn.decomposition import PCA
vectors_red=PCA(n_components=2).fit_transform(vectors)

from sklearn import preprocessing

normalizer = preprocessing.Normalizer()
vectors_red =  normalizer.fit_transform(vectors_red, 'l2')

fig, ax = plt.subplots()
ax.set_xlim(min([vectors[word2int[w]][0] for w in words])-1, max([vectors[word2int[w]][0] for w in words])+1)
ax.set_ylim(min([vectors[word2int[w]][1] for w in words])-1, max([vectors[word2int[w]][1] for w in words])+1)
for word in words:
    ax.annotate(word, (vectors_red[word2int[word]][0], vectors_red[word2int[word]][1]))
plt.show()

#plt.scatter(vectors_red[:,0],vectors_red[:,1])
def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index

print(int2word[find_closest(word2int['she'], vectors)])
print(int2word[find_closest(word2int['queen'], vectors)])
print(int2word[find_closest(word2int['royal'], vectors)])




"""------------------------------------------------------------------------------------------------------------"""

"""CBOW"""

import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

corpus_raw = corpus_raw = 'He is the king . The king is royal . She is the royal queen . The queen is beautiful . He is the royal prince '

# convert to lower case
corpus_raw = corpus_raw.lower()

words = []
for word in corpus_raw.split():
    if word != '.': # because we don't want to treat . as a word
        words.append(word)

words = set(words) # so that all duplicate words are removed
word2int = {}
int2word = {}
vocab_size = len(words) # gives the total number of unique words

for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

# raw sentences is a list of sentences.
raw_sentences = corpus_raw.split('.')
sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())

WINDOW_SIZE = 2

data = []
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                data.append([nb_word, word])  #only change from skipgram

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

x_train = [] # input word
y_train = [] # output word

for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))

# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

EMBEDDING_DIM = 5 # you can choose your own number

n_iters=5000

model = Sequential([
    Dense(EMBEDDING_DIM, input_shape=(vocab_size,)),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=n_iters)
W,b=model.layers[0].get_weights()
vectors=W+b
print(vectors)

'''from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0,perplexity=5)
np.set_printoptions(suppress=True)
vectors_red = tsne.fit_transform(vectors) '''

from sklearn.decomposition import PCA
vectors_red=PCA(n_components=2).fit_transform(vectors)

from sklearn import preprocessing

normalizer = preprocessing.Normalizer()
vectors_red =  normalizer.fit_transform(vectors_red, 'l2')

fig, ax = plt.subplots()
ax.set_xlim(min([vectors[word2int[w]][0] for w in words])-1, max([vectors[word2int[w]][0] for w in words])+1)
ax.set_ylim(min([vectors[word2int[w]][1] for w in words])-1, max([vectors[word2int[w]][1] for w in words])+1)
for word in words:
    ax.annotate(word, (vectors_red[word2int[word]][0], vectors_red[word2int[word]][1]))
plt.show()

#plt.scatter(vectors_red[:,0],vectors_red[:,1])
def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index

print(int2word[find_closest(word2int['king'], vectors)])
print(int2word[find_closest(word2int['queen'], vectors)])
print(int2word[find_closest(word2int['royal'], vectors)])
