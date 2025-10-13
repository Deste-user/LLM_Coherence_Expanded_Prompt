from PIL import Image
import random as rdm
import re
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import os
import subprocess
import requests
import time

import env_for_chatting as efc
import grafic_analysis as ga
import analysis as a


NUM_CLASS = 1000
NUM_IMG_4_CLASS = 5
ITERATIONS = 10  
model_name = "openai/clip-vit-base-patch16"
device = None # Initialize device variable
img_emb = None

#Function to Download the dataset if it not present
def download_dataset():
    dataset_dir = "./Small-ImageNet-Validation-Dataset-1000-Classes"
    if not os.path.exists(dataset_dir):
        print(" Clonation of the dataset from GitHub...")
        subprocess.run(["git", "clone", "https://github.com/ndb796/Small-ImageNet-Validation-Dataset-1000-Classes.git"])
    else:
        print("Dataset is already present.")

# To initialize the ollama sw in a determinated port (11434)
def start_ollama():
    try:
        requests.get("http://localhost:11434")
        print("Ollama is already running.")
        return
    except requests.exceptions.ConnectionError:
        pass

    print("Run Ollama in 'serve' mode...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(10):
        try:
            r = requests.get("http://localhost:11434")
            if r.status_code in [200, 404]:
                print("Ollama serve is active.")
                return
        except requests.exceptions.ConnectionError:
            time.sleep(1)

    raise RuntimeError("Ollama is not running.")        


if __name__ == '__main__':
    models = ["mistral:latest", "phi3:latest","llama3.2:latest","gemma2:latest"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("I'm using:", device)
    
    # Download the dataset if it not present
    download_dataset()

    # Start the ollama server
    start_ollama()

    # Load the CLIP model and processor
    clip_model = CLIPModel.from_pretrained(model_name)

    #if I use use_fast=True i can't put image to make the embeedings 
    clip_processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
    clip_model.to(device)

    a.create_all_img_embedding(clip_processor, clip_model, device, NUM_CLASS, NUM_IMG_4_CLASS, model_name)
    embeddings_prompt, avg_time=a.create_prompt_embedding_for_all(clip_processor,clip_model,model_name,models, NUM_CLASS, device)

    print(embeddings_prompt[0])

    avg_clipscores=a.analyze(models, NUM_CLASS, embeddings_prompt)

    #TODO: try to parallelize the code (facoltative)
    ga.create_clipscore_table(avg_clipscores ,models, NUM_CLASS)
    ga.grafic_avg_time(models, avg_time)
    ga.grafic_distribution(models)
    
    
    # #To see if the avg clip score stabilize after a certain number of iterations
    # #It's not so useful for the paper
    # #TODO: Calculate for all class and do the avg of avg clip score for each model
    #ga.grafic_analysis(models, 10, ITERATIONS, NUM_CLASS, clip_processor, clip_model)

    


    
