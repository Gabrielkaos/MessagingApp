from django.shortcuts import render, redirect
from django.http import JsonResponse
import numpy as np
import json
from nltk.stem import PorterStemmer
import nltk
import random
import os
from django.conf import settings

nltk_data_dir = os.path.join(settings.BASE_DIR, 'chat_app', 'static', 'chat_app', 'julAi', 'nltk_data')


os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir) 

class TrainNet:
    def __init__(self, n_input, n_hidden, n_output, weight_scale=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.randn(n_input, n_hidden) * weight_scale
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_hidden) * weight_scale
        self.b2 = np.zeros((1, n_hidden))
        self.W3 = np.random.randn(n_hidden, n_output) * weight_scale
        self.b3 = np.zeros((1, n_output))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def forward(self, X):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        scores = self.a2.dot(self.W3) + self.b3
        return scores

    def backward(self, X, y, scores):
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        N = X.shape[0]

        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N

        self.dW3 = self.a2.T.dot(dscores)
        self.db3 = np.sum(dscores, axis=0, keepdims=True)

        da2 = dscores.dot(self.W3.T)
        dz2 = da2 * (self.z2 > 0)

        self.dW2 = self.a1.T.dot(dz2)
        self.db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2.dot(self.W2.T)
        dz1 = da1 * (self.z1 > 0)

        self.dW1 = X.T.dot(dz1)
        self.db1 = np.sum(dz1, axis=0, keepdims=True)

    def update_params(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3

class NumpyNeuralNet:
    def __init__(self, n_input, n_hidden, n_output, weight_scale=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.W1 = np.random.randn(n_input,  n_hidden)  * weight_scale
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden,  n_hidden)  * weight_scale
        self.b2 = np.zeros((1, n_hidden))
        self.W3 = np.random.randn(n_hidden,  n_output) * weight_scale
        self.b3 = np.zeros((1, n_output))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def forward(self, X):
        
        
        z1 = X.dot(self.W1) + self.b1
        a1 = self.relu(z1)

        
        z2 = a1.dot(self.W2) + self.b2
        a2 = self.relu(z2)

        
        out = a2.dot(self.W3) + self.b3
        return out


def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    stemmer=PorterStemmer()
    return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence,all_words):
    tokenized_sentence=[stem(w) for w in tokenized_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0
    return bag

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def index(request):
    intents_paths = os.path.join(settings.BASE_DIR,'chat_app','static','chat_app' ,'julAi', 'intents.json')

    with open(intents_paths, "r", encoding="UTF-8") as f:
        intents = json.load(f)

    all_words, tags, xy = [], [], []
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tokens = tokenize(pattern)
            all_words.extend(tokens)
            xy.append((tokens, tag))
    return render(request,"chat_app/chatbox.html",{"tags":tags})


def response(request):

    """
    init julAi
    """

    intents_paths = os.path.join(settings.BASE_DIR,'chat_app','static','chat_app' ,'julAi', 'intents.json')
    brain_path = os.path.join(settings.BASE_DIR,'chat_app','static','chat_app' ,'julAi', 'julAi_brain.npz')
    print(intents_paths)
    with open(intents_paths, 'r', encoding='UTF-8') as f:
        intents = json.load(f)

    data = np.load(brain_path, allow_pickle=True)
    all_words = data['all_words'].tolist()
    tags      = data['tags'].tolist()

    n_input  = data['W1'].shape[0]
    n_hidden = data['W1'].shape[1]
    n_output = data['W3'].shape[1]
    model    = NumpyNeuralNet(n_input, n_hidden, n_output)
    model.W1, model.b1 = data['W1'], data['b1']
    model.W2, model.b2 = data['W2'], data['b2']
    model.W3, model.b3 = data['W3'], data['b3']

    not_responses = [
        "I'm sorry, I didn't quite catch that. Could you please rephrase your question or provide more context?",
        "I'm still learning, and I might not fully understand what you're asking. Could you please try rephrasing or providing more details?",
        "It seems like I'm having trouble understanding your question. Can you please try phrasing it differently?",
    ]

    """
    generate response
    """
    data = json.loads(request.body)
    sentence = data.get('user_message')
    # print("sentence:",___sentence)
    tokens = tokenize(sentence)

    stems = [stem(w) for w in tokens]
    bow = bag_of_words(stems, all_words)
    X = bow.reshape(1, -1)

    
    scores = model.forward(X)
    probs  = softmax(scores)
    pred_i = np.argmax(probs, axis=1)[0]
    tag     = tags[pred_i]
    confidence = probs[0, pred_i]

    
    print(confidence)
    if confidence > 0.75:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                # print(f"\n{bot_name}: {response}\n")
                break
    else:
        response = random.choice(not_responses)
    
    if request.method=="POST":
        return JsonResponse({'response':response})
    return JsonResponse({'error':'Invalid method'})

def save_data(file_path, model, all_words, tags):
    np.savez(file_path,
             W1=model.W1, b1=model.b1,
             W2=model.W2, b2=model.b2,
             W3=model.W3, b3=model.b3,
             all_words=all_words, tags=tags)
    print(f"\nTRAINING COMPLETE, saved to: '{file_path}.npz'")

def get_tags():
    intents_paths = os.path.join(settings.BASE_DIR,'chat_app','static','chat_app' ,'julAi', 'intents.json')

    with open(intents_paths, "r", encoding="UTF-8") as f:
        intents = json.load(f)

    all_words, tags, xy = [], [], []
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tokens = tokenize(pattern)
            all_words.extend(tokens)
            xy.append((tokens, tag))
    return tags
def get_data_from_intents():
    intents_paths = os.path.join(settings.BASE_DIR,'chat_app','static','chat_app' ,'julAi', 'intents.json')

    with open(intents_paths, "r", encoding="UTF-8") as f:
        intents = json.load(f)

    all_words, tags, xy = [], [], []
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tokens = tokenize(pattern)
            all_words.extend(tokens)
            xy.append((tokens, tag))

    ignore = ["?", ",", "!", ".", "'"]
    all_words = sorted(set(stem(w) for w in all_words if w not in ignore))
    tags = sorted(set(tags))

    X_train, y_train = [], []
    for (tokens, tag) in xy:
        bow = bag_of_words(tokens, all_words)
        X_train.append(bow)
        y_train.append(tags.index(tag))

    return np.array(X_train), np.array(y_train), tags, all_words


def train(req):
    if req.method=="POST":
    
        batch_size = 4
        n_hidden = 100
        lr = 2e-1
        num_epochs =1000

        
        X_train, y_train, tags, all_words = get_data_from_intents()
        n_input = X_train.shape[1]
        n_output = len(tags)

        
        model = TrainNet(n_input, n_hidden, n_output, seed=42, weight_scale=0.1)

    
        for epoch in range(1, num_epochs + 1):
            perm     = np.random.permutation(len(X_train))
            X_shuf   = X_train[perm]
            y_shuf   = y_train[perm]

            for i in range(0, len(X_shuf), batch_size):
                X_batch = X_shuf[i : i + batch_size]
                y_batch = y_shuf[i : i + batch_size]
                N       = X_batch.shape[0]

                scores = model.forward(X_batch)

                exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probs      = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                loss       = -np.mean(np.log(probs[np.arange(N), y_batch]))

                model.backward(X_batch, y_batch, scores)
                model.update_params(lr)

                assert probs.shape == (N, n_output)
                assert y_batch.shape[0] == N

            if epoch % 5 == 0:
                print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss:.5f}")
            if loss < 1e-4:
                print(f"\n[TARGET REACHED] Epoch: {epoch}/{num_epochs}, Loss: {loss:.5f}\n")
                break


        print(f"Final Loss: {loss:.4f}")
        brain_path = os.path.join(settings.BASE_DIR,'chat_app','static','chat_app' ,'julAi', 'julAi_brain')

        save_data(brain_path, model, all_words, tags)
        return redirect("chat_app:index")
    

def add(req):
    if req.method=="POST":
        try:
            selected_tag = req.POST["tags"]
        except:
            selected_tag = ""
        tag = req.POST["tag"]
        pattern = req.POST["pattern"]
        response = req.POST["response"]

        real_tag = selected_tag if tag=="" else tag

        if not real_tag or not response or not pattern:
            return redirect("chat_app:index")
        
        intents_paths = os.path.join(settings.BASE_DIR,'chat_app','static','chat_app' ,'julAi', 'intents.json')
        with open(intents_paths, "r", encoding="utf-8") as file:
            data = json.load(file)

        tag_found = False
        for intent in data["intents"]:
            if intent["tag"] == real_tag:
                if pattern not in intent["patterns"]:
                    intent["patterns"].append(pattern)
                if response not in intent["responses"]:
                    intent["responses"].append(response)
                tag_found = True
                break

        if not tag_found:
            new_intent = {
                "tag": real_tag,
                "patterns": [pattern],
                "responses": [response]
            }
            data["intents"].append(new_intent)

        with open(intents_paths, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)


        return redirect("chat_app:index")