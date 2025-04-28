from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import json
from nltk.stem import PorterStemmer
import nltk
import random
import os
from django.conf import settings

# Create your views here.
class NumpyNeuralNet:
    def __init__(self, n_input, n_hidden, n_output, weight_scale=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize weights and biases
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
    return render(request,"chat_app/chatbox.html")


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
    # print("sentence:",sentence)
    tokens = tokenize(sentence)

    stems = [stem(w) for w in tokens]
    bow = bag_of_words(stems, all_words)
    X = bow.reshape(1, -1)

    # Forward pass
    scores = model.forward(X)
    probs  = softmax(scores)
    pred_i = np.argmax(probs, axis=1)[0]
    tag     = tags[pred_i]
    confidence = probs[0, pred_i]

    # Choose a response
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