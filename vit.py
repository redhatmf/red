import numpy as np

# Define the POS tags
POS_TAGS = {'N': 0, 'V': 1}

# Transition probabilities
transition_prob = np.array([[0.7, 0.3],  # Noun to Noun, Noun to Verb
                             [0.4, 0.6]]) # Verb to Noun, Verb to Verb

# Emission probabilities
emission_prob = {'N': {'the': 0.5, 'dog': 0.1, 'cat': 0.1, 'chases': 0.1, 'runs': 0.2},
                 'V': {'the': 0.1, 'dog': 0.1, 'cat': 0, 'chases': 0.7, 'runs': 0.1}}

def viterbi(sentence):
    # Convert sentence to lowercase and split into words
    words = sentence.lower().split()

    # Initialize viterbi matrix and backpointer matrix
    viterbi_mat = np.zeros((len(POS_TAGS), len(words)))
    backpointer = np.zeros((len(POS_TAGS), len(words)), dtype=int)

    # Initialize first column of viterbi matrix using initial probabilities
    for s in POS_TAGS.values():
        viterbi_mat[s][0] = transition_prob[s][s] * emission_prob[list(POS_TAGS.keys())[s]].get(words[0], 0)

    # Recursion step
    for t in range(1, len(words)):
        for s in POS_TAGS.values():
            probs = [viterbi_mat[s_prev][t - 1] * transition_prob[s_prev][s] * emission_prob[list(POS_TAGS.keys())[s]].get(words[t], 0) for s_prev in POS_TAGS.values()]
            viterbi_mat[s][t] = max(probs)
            backpointer[s][t] = np.argmax(probs)

    # Termination step
    best_path_prob = max(viterbi_mat[:, len(words) - 1])
    best_last_state = np.argmax(viterbi_mat[:, len(words) - 1])

    # Backtrack to find the best path
    best_path = [list(POS_TAGS.keys())[best_last_state]]
    for t in range(len(words) - 1, 0, -1):
        best_last_state = backpointer[best_last_state][t]
        best_path.append(list(POS_TAGS.keys())[best_last_state])
    best_path.reverse()

    return best_path, best_path_prob

def sequence_probability(sentence, pos_sequence):
    words = sentence.lower().split()
    prob = 1.0
    prev_pos = None
    for word, pos_tag in zip(words, pos_sequence):
        if prev_pos is not None:
            prob *= transition_prob[POS_TAGS[prev_pos]][POS_TAGS[pos_tag]]
        prob *= emission_prob[pos_tag].get(word, 0)
        prev_pos = pos_tag
    return prob

# Example usage
sentence = "The dog chases the cat"
best_path, best_path_prob = viterbi(sentence)
print("Most likely sequence of POS tags:", best_path)
print("Probability of the best path:", best_path_prob)

# Calculate the probability of a given sequence of POS tags
pos_sequence = ['N', 'V', 'V', 'N', 'N']
sequence_prob = sequence_probability(sentence, pos_sequence)
print("Probability of the given sequence:", sequence_prob)
