"""SKIPGRAM AND CBOW"""

import numpy as np

# Generate some sample data skipgram this is easy
corpus = ["I like playing football with my friends",
          "My friends enjoy playing football too",
          "Football is a fun game",
          "We play football every weekend"]

words = []
for sentence in corpus:
    words.extend(sentence.lower().split())
words

vocab = set(words)
word2id = {word: i for i, word in enumerate(vocab)}
id2word = {i: word for word, i in word2id.items()}
vocab_size = len(vocab)

word2id

window_size = 2
skipgrams = []
for sentence in corpus:
    words = sentence.lower().split()
    for i, target_word in enumerate(words):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                context_word = words[j]
                skipgrams.append((word2id[target_word], word2id[context_word]))

skipgrams

embedding_dim = 100
learning_rate = 0.01
num_epochs = 10  # Adjust this as needed

W = np.random.uniform(-1, 1, (vocab_size, embedding_dim))
W

import numpy as np

# Generate some sample data skipgram this is easy
corpus = ["I like playing football with my friends",
          "My friends enjoy playing football too",
          "Football is a fun game",
          "We play football every weekend"]

# Tokenize the corpus
words = []
for sentence in corpus:
    words.extend(sentence.lower().split())

vocab = set(words)
word2id = {word: i for i, word in enumerate(vocab)}
id2word = {i: word for word, i in word2id.items()}
vocab_size = len(vocab)

# Generate skip-grams
window_size = 2
skipgrams = []
for sentence in corpus:
    words = sentence.lower().split()
    for i, target_word in enumerate(words):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                context_word = words[j]
                skipgrams.append((word2id[target_word], word2id[context_word]))

# Define the model parameters
embedding_dim = 100
learning_rate = 0.01
num_epochs = 10  # Adjust this as needed

# Initialize word embeddings randomly
W = np.random.uniform(-1, 1, (vocab_size, embedding_dim))

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    np.random.shuffle(skipgrams)
    for target_id, context_id in skipgrams:
        target_embedding = W[target_id]
        context_embedding = W[context_id]

        # Calculate the dot product of target and context embeddings
        score = np.dot(target_embedding, context_embedding)

        # Apply sigmoid function to get probabilities
        pred = 1 / (1 + np.exp(-score))

        # Calculate the loss
        loss = -np.log(pred)

        # Update the embeddings using gradient descent
        grad = pred - 1
        W[target_id] -= learning_rate * grad * context_embedding
        W[context_id] -= learning_rate * grad * target_embedding

        total_loss += loss

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(skipgrams)}')

import numpy as np

def train_cbow(corpus, vocab_size, embedding_dim, window_size, learning_rate, num_epochs):
    W = np.random.uniform(-1, 1, (vocab_size, embedding_dim))
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for word, i in word_to_id.items()}

    for epoch in range(num_epochs):
        total_loss = 0
        np.random.shuffle(corpus)
        for i, target_word in enumerate(corpus):
            context = []
            for j in range(i - window_size, i + window_size + 1):
                if i != j and j >= 0 and j < len(corpus):
                    context.append(corpus[j])
            if context:
                target_id = word_to_id[target_word]
                context_ids = [word_to_id[word] for word in context]
                target_embedding = W[target_id]
                context_embeddings = [W[context_id] for context_id in context_ids]

                pred = np.mean(context_embeddings, axis=0)
                loss = -np.log(sigmoid(np.dot(pred, target_embedding)))
                total_loss += loss

                grad = sigmoid(np.dot(pred, target_embedding)) - 1
                for context_id in context_ids:
                    W[context_id] -= learning_rate * grad * pred
                W[target_id] -= learning_rate * grad * target_embedding

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(corpus)}')

    return W, word_to_id, id_to_word

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example usage:
corpus = ["I like playing football with my friends",
          "My friends enjoy playing football too",
          "Football is a fun game",
          "We play football every weekend"]

words = []
for sentence in corpus:
    words.extend(sentence.lower().split())

vocab = set(words)
vocab_size = len(vocab)
embedding_dim = 100
window_size = 2
learning_rate = 0.01
num_epochs = 10

W, word_to_id, id_to_word = train_cbow(words, vocab_size, embedding_dim, window_size, learning_rate, num_epochs)

print("Word Embeddings:")
print(W)

# Inspect the embedding for a specific word
word_to_inspect = "football"
if word_to_inspect in word_to_id:
    word_id = word_to_id[word_to_inspect]
    embedding = W[word_id]
    print(f"Embedding for '{word_to_inspect}':")
    print(embedding)
else:
    print(f"'{word_to_inspect}' not found in the vocabulary.")

# Save word embeddings to a file if needed
np.savetxt("word_embeddings.txt", W)
