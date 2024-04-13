

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
