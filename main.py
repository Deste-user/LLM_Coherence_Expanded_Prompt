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



# Function to compute CLIP score for an image and text
# The score is calculated as w * max(cosine_similarity, 0)
# if clamp_zero is False, the score is w * cosine_similarity
# This because if is True the average CLIP score is higher because negative scores are clamped to zero.
# We can use models with memory 
# And give to eat those examples for model.

# def compute_clip_score(image_path, text, w=2.5,clamp_zero=False):
#     global img_emb
#     image = Image.open(image_path).convert("RGB")
#     tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

#     # To include in the paper
#     # To count the number of tokens in the text (output of the model)
#     # tokens = tokenizer(text, return_tensors="pt", padding=True)
#     # num_tokens = tokens["input_ids"].shape[1]
#     # print(f"Number of tokens: {num_tokens}")
#     # If the image embedding is not computed yet, compute it
#     if img_emb is None:
#         tokenizer.use_fast = False
#         image_inputs = clip_processor(images=image, return_tensors="pt")
#         image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
#         with torch.no_grad():
#             img_emb = clip_model.get_image_features(**image_inputs)
#             tokenizer.use_fast = True   


#     # Preprocess the image and text
#     text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
#     text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
#     with torch.no_grad():
#         text_emb = clip_model.get_text_features(**text_inputs)


#     img_emb = img_emb.squeeze(0)   # diventa [D]
#     text_emb = text_emb.squeeze(0) 


#     # Normalize embeddings
#     img_emb = img_emb / img_emb.norm()
#     text_emb = text_emb / text_emb.norm()

#     # Cosine similarity
#     cosine_sim = torch.dot(img_emb, text_emb).item()
        
#     if clamp_zero:
#         # Apply max(., 0) e weights w
#         score = w * max(cosine_sim, 0.0)
#     else:
#         score = w * cosine_sim
#     return score

# Function to calculate average scores and response times for each model
# def avg_score(score_dict):
#     avg_scores = {}
#     for model, values in score_dict.items():
#         scores = values["clipscores"]
#         times = values["response_time"]

#         avg_clip = sum(scores) / len(scores) if scores else 0
#         avg_time = sum(times) / len(times) if times else 0

#         avg_scores[model] = {"avg_clipscore": avg_clip, "avg_time": avg_time}
#     return avg_scores


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
    #models = ["mistral:latest"] , "gemma2:latest","deepseek-r1:14b, , "llama3.2:latest""

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

    #Here didn't see the dataset, so there will be error #TODO: manage the error. 
    a.create_all_img_embedding(clip_processor, clip_model, device, NUM_CLASS, NUM_IMG_4_CLASS, model_name)
    embeddings_prompt=a.create_prompt_embedding_for_all(clip_processor,clip_model,model_name,models, NUM_CLASS, device)


    avg_clipscores=a.analyze(models, NUM_CLASS, embeddings_prompt)

    #TODO: try to parallelize the code (facoltative)
    ga.create_clipscore_table(avg_clipscores ,models, NUM_CLASS)

    #TODO: Calculate for all class and do the avg of avg clip score for each model
    #ga.grafic_analysis(models, 10, ITERATIONS, NUM_CLASS, clip_processor, clip_model)

    


    
