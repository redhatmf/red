
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

