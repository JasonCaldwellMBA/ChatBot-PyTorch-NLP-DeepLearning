import random
import json
import torch
from src.features.build_features import bag_of_words, tokenize
from src.models.predict_model import NeuralNet
from os.path import dirname as dir

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

bot_name = "Helper"
print("Let's chat! Type 'quit' to exit.")
while True:
    sentence = input('You: ')
    if sentence.lower() == "quit":
        break

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
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(
            f"{bot_name}: Sorry, I do not understand. Is there another way you can ask the question?")
