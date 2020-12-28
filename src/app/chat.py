import random
import json
import torch
from src.features.build_features import bag_of_words, tokenize
from src.models.predict_model import NeuralNet
from os.path import dirname as dir
from datetime import datetime
import csv

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

root = dir(dir(dir(__file__)))
file = root + '/data/processed/intents.json'

with open(file, 'r') as f:
    intents = json.load(f)

FILE = root + '/data/interim/data.pth'
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

response_feedback = []
bot_name = "Helper"
print(f"{bot_name}: Hello, how can I help you?")
while True:
    sentence = input('You: ')
    if sentence.lower() == "quit":
        break

    question = sentence

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probabilities = torch.softmax(output, dim=1)
    probablility = probabilities[0][predicted.item()]

    if probablility.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f"{bot_name}: {response}")
        print(f"Feedback: Was this response helpful? 1 = 'Yes, this is what I was looking for!' 2 = 'Yes, but I don't like how this works.' 3 = 'Maybe, I have some instructions to follow.', 4 = 'No, but I found the answer somewhere else.', 5 = 'No, this bot has a lot to learn.'")
        feedback = input()
    else:
        response = 'Sorry, I do not understand. Is there another way you can ask the question?'
        print(f"{bot_name}: {response}")
        feedback = -1

    response_feedback.append([datetime.now(), question, response, feedback])

with open(root + '/data/interim/response_feedback.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in response_feedback:
        writer.writerow(row)
