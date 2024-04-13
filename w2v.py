
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

text = """Aligned to the vision of the department - Stay Ahead and Be Relevant, the Department of Applied Mathematics and Computational Sciences has introduced various programmes predicting the demands of this ever-changing world. It is also the largest department in PSG College of Technology and its breadth and scale bring unique advantage in terms of bridging the demand of industry ready professionals through state-of-the-art programmes run by the department. Graduate teaching provides a strong foundation in Mathematics and Computer science and also exposes students to the latest research and developments. The department offers 5 graduate programmes and one undergraduate programme.
        The five year integrated M.Sc Software Systems (erstwhile Software Engineering) programme, offered since 1997, caters to the human resource requirement of leading software industries across the globe. The programme has been designed to meet the challenging needs of the industry, by ensuring a good understanding of the software design process and to develop resilient applications using state-of-the art technologies.
        The five year integrated M.Sc Theoretical Computer Science is yet another innovative programme offered since 2007 has been well received by the R&D divisions of software industries and top notch research institutions for higher education across the globe.
        In the new era of Big Data, M.Sc Data Science was introduced during 2015 to solve the exponential growth and curse of dimensionality in giant databases accumulated by the industries. The programme has been designed to meet the current demands in the industry and to create pioneering experts in the field of data science.
        The five-year integrated M.Sc. Cyber Security programme was started in the year 2020, the first of its kind in India aims to prepare students with the technical knowledge and skills needed to protect and defend computer systems and networks. The programme has a strong and wide technical base and internship programs which are the most critical aspects to a good cyber security education.
        The M.Sc Applied Mathematics programme was offered by the department since 1975 to acquaint the students with various principles of Mathematics and apply to all relative fields of science, technology and management. This programme is also designed to expose the students to the development and applications of software, catering to the needs of the industries and R&D Sector.
        To meet the requirements of IT field, the department offers an undergraduate programme B.Sc Computer Systems & Design (erstwhile Computer Technology) since 1985. This programme emphasizes development of programming skills, understanding system design tools and technologies for effective problem solving.
        The Department has over 60 faculty members with a wide range of research specialties, spanning Mathematical Modelling, Topology, Epidemic Modelling, Graph Algorithms, Applied Machine learning to Cybersecurity. Their publications, fellowships and project funding strengthen the recognition of our department as a powerhouse of research as well as excellent teaching."""



"""## Building Dataset for CBOW context  words -> target word"""

vocab = list(set(text.split()))

one_hot_encoded_words = {}

c = 0
for i in vocab:
  temp = [0 for j in range(len(vocab))]
  temp[c] = 1
  one_hot_encoded_words[i] = temp
  c+=1

print("no of unique words:",len(vocab))

words = text.split()

context_words = []
target_word = []
window_size = 2
for wv in range(len(words)-2):
   if wv>window_size:
    context = [words[wv-1],words[wv-2],words[wv+1],words[wv+2]]
    context_words.append(context)
    target_word.append(words[wv])

cbow_X = []
cbow_Y = []
for record in range(len(context_words)):
  temp = []
  for i in range(len(context_words[record])):
    temp.extend(one_hot_encoded_words[context_words[record][i]])
  cbow_X.append(temp)
  cbow_Y.append(one_hot_encoded_words[target_word[i]])

cbow_X,cbow_Y = np.array(cbow_X),np.array(cbow_Y)

print(cbow_X.shape,cbow_Y.shape)

import keras
import tensorflow as tf

model = keras.Sequential([
    keras.layers.Input(shape=(992,)),
    keras.layers.Embedding(input_dim=248,output_dim=2),
    keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    keras.layers.Dense(248,activation='softmax')
])

model.summary()

model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer = keras.optimizers.SGD(),metrics = [keras.metrics.CategoricalAccuracy()])

model.fit(cbow_X,cbow_Y,epochs=10)

model.get_weights()[1].shape

embeddings = np.matmul(cbow_Y,model.get_weights()[1].T)
embeddings.shape

"""## SKip Gram"""

skip_X = []
skip_Y = []

for i in range(len(target_word)):
  for j in range(len(context_words[i])):
    skip_Y.append(one_hot_encoded_words[target_word[i]])
    skip_X.append(one_hot_encoded_words[context_words[i][j]])

skip_X,skip_Y = np.array(skip_X),np.array(skip_Y)

print(skip_X.shape,skip_Y.shape)

skip_model = keras.Sequential([
    keras.layers.Input(shape=(248,)),
    keras.layers.Embedding(input_dim=248,output_dim=2),
    keras.layers.Lambda(lambda x: tf.reduce_mean(x,axis=1)),
    keras.layers.Dense(248,activation='softmax')
])

skip_model.summary()

skip_model.compile(loss=keras.losses.CategoricalCrossentropy(),optimizer=keras.optimizers.Adam(),metrics=[keras.metrics.CategoricalAccuracy()])

skip_model.fit(skip_X,skip_Y,epochs=10)



skip_Y.shape

embeddings = np.matmul(skip_Y,skip_model.get_weights()[1].T)
embeddings.shape


# Predict word embeddings using CBOW model
cbow_embeddings = model.get_weights()[1].T  # Get the embedding weights
# Assuming 'word_index' is the index of the word whose embedding you want to retrieve
word_index = vocab.index('Department')  # Replace 'your_word_here' with the word you want to predict
cbow_embedding_for_word = cbow_embeddings[word_index]
print("CBOW Embedding for the word:", cbow_embedding_for_word)

# Predict word embeddings using Skip-gram model
skip_embeddings = skip_model.get_weights()[1].T  # Get the embedding weights
skip_embedding_for_word = skip_embeddings[word_index]
print("Skip-gram Embedding for the word:", skip_embedding_for_word)

from sklearn.metrics.pairwise import cosine_similarity

# Assume cbow_embeddings and skip_embeddings are already defined as in the previous code
target_embedding = cbow_embedding_for_word  # Use the embedding you want to find the word for

# Calculate cosine similarity between the target embedding and all other embeddings
cbow_similarities = cosine_similarity([target_embedding], cbow_embeddings)
skip_similarities = cosine_similarity([target_embedding], skip_embeddings)

# Find the index of the most similar word in the vocabulary
cbow_most_similar_idx = cbow_similarities.argmax()
skip_most_similar_idx = skip_similarities.argmax()

# Retrieve the word corresponding to the most similar index
cbow_most_similar_word = vocab[cbow_most_similar_idx]
skip_most_similar_word = vocab[skip_most_similar_idx]

print("Most similar word (CBOW):", cbow_most_similar_word)
print("Most similar word (Skip-gram):", skip_most_similar_word)

# import library
import pandas as pd
import numpy as np
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text = """
        Aligned to the vision of the department - Stay Ahead and Be Relevant, the Department of Applied Mathematics and Computational Sciences has introduced various programmes predicting the demands of this ever-changing world. It is also the largest department in PSG College of Technology and its breadth and scale bring unique advantage in terms of bridging the demand of industry ready professionals through state-of-the-art programmes run by the department. Graduate teaching provides a strong foundation in Mathematics and Computer science and also exposes students to the latest research and developments. The department offers 5 graduate programmes and one undergraduate programme.
        The five year integrated M.Sc Software Systems (erstwhile Software Engineering) programme, offered since 1997, caters to the human resource requirement of leading software industries across the globe. The programme has been designed to meet the challenging needs of the industry, by ensuring a good understanding of the software design process and to develop resilient applications using state-of-the art technologies.
        The five year integrated M.Sc Theoretical Computer Science is yet another innovative programme offered since 2007 has been well received by the R&D divisions of software industries and top notch research institutions for higher education across the globe.
        In the new era of Big Data, M.Sc Data Science was introduced during 2015 to solve the exponential growth and curse of dimensionality in giant databases accumulated by the industries. The programme has been designed to meet the current demands in the industry and to create pioneering experts in the field of data science.
        The five-year integrated M.Sc. Cyber Security programme was started in the year 2020, the first of its kind in India aims to prepare students with the technical knowledge and skills needed to protect and defend computer systems and networks. The programme has a strong and wide technical base and internship programs which are the most critical aspects to a good cyber security education.
        The M.Sc Applied Mathematics programme was offered by the department since 1975 to acquaint the students with various principles of Mathematics and apply to all relative fields of science, technology and management. This programme is also designed to expose the students to the development and applications of software, catering to the needs of the industries and R&D Sector.
        To meet the requirements of IT field, the department offers an undergraduate programme B.Sc Computer Systems & Design (erstwhile Computer Technology) since 1985. This programme emphasizes development of programming skills, understanding system design tools and technologies for effective problem solving.
        The Department has over 60 faculty members with a wide range of research specialties, spanning Mathematical Modelling, Topology, Epidemic Modelling, Graph Algorithms, Applied Machine learning to Cybersecurity. Their publications, fellowships and project funding strengthen the recognition of our department as a powerhouse of research as well as excellent teaching."""

words = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(words)

pos_tags

vocab = list(set(words))
unique_tags =  list(set([val for key,val in pos_tags]))
len(vocab),len(unique_tags)

V = len(vocab)
T = len(unique_tags)

V

T

def transition_prob(t1,t2,data=pos_tags):
    tags = [val for key,val in data]
    t1_count = tags.count(t1)
    t1_t2_count = 0
    for i in range(len(tags)-1):
        if tags[i] == t1 and tags[i+1]==t2:
            t1_t2_count += 1
    return t1_count,t1_t2_count

def emission_prob(t1,w1,data=pos_tags):
    tags = [pair for pair in data if pair[1]==t1]
    t1_count = len(tags)
    word = [key for key,val in tags if key==w1]
    w1_count = len(word)
    return w1_count,t1_count

def calculate_start_prob(data=pos_tags,tags=unique_tags):
    full_stop_index = []
    for i , pair in enumerate(pos_tags):
        if pair[0]=='.':
            full_stop_index.append(i)
    start_index = [0]
    for idx in full_stop_index:
        if idx+1 < len(data):
            start_index.append(idx+1)
    start_prob = {}
    start_tags = [data[idx][1] for idx in start_index]
    for tag in tags:
        start_prob[tag] = round(start_tags.count(tag)/len(start_index),4)

    return start_prob

start_prob = calculate_start_prob()

start_prob

emission_prob_mat = np.zeros((V,T))
transition_prob_mat = np.zeros((T,T))
for i,tag1 in enumerate(unique_tags):
    for j , tag2 in enumerate(unique_tags):
        t1_t2 , t1 = transition_prob(tag1,tag2)
        transition_prob_mat[i][j]= round((t1_t2+1)/(t1+T),4)

for i,word in enumerate(vocab):
    for j,tag in enumerate(unique_tags):
        w,t = emission_prob(tag,word)
        emission_prob_mat[i][j] = round((w+1)/(t+V),4)

emission_df = pd.DataFrame(emission_prob_mat,columns=unique_tags,index=vocab)
transition_df = pd.DataFrame(transition_prob_mat,columns=unique_tags,index=unique_tags)

emission_df.head()

transition_df.head()

emission_df.loc[vocab[0],unique_tags[0]]

def viterbi(start_prob,emission_prob,transition_prob,obs_states,hidden_states):
    v = [{}]
    for i in start_prob:
        v[0][i] = [round(start_prob[i]*emission_prob.loc[obs_states[0],i],4),None]
    for i in range(1,len(obs_states)):
        v.append({})
        for tag in hidden_states:
            prev_prob = [v[i-1][t][0] * transition_df.loc[t,tag] for t in hidden_states]
            max_prob = max(prev_prob)
            prev_tag = hidden_states[prev_prob.index(max_prob)]
            v[i][tag]=[max_prob*emission_df.loc[obs_states[i],tag],prev_tag]
    return v

obs_states = "Aligned to the vision of the department"
obs_states = obs_states.split()
v = viterbi(start_prob,emission_df,transition_df,obs_states,unique_tags)

path = []
i = len(obs_states)-1
curr_path = max(v[-1],key = lambda k : v[-1][k][0])
path.append(curr_path)
#v[word-1][curr_path][1]
while i >= 0 :
    curr_path = v[i][curr_path][1]
    path.append(curr_path)
    i -= 1

path = np.flip(path)
print(path)

