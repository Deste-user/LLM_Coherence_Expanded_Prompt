import os
from PIL import Image
from transformers import CLIPTokenizer
import torch
import numpy as np
import time
from tqdm import tqdm


import env_for_chatting as efc

#Create a file than calculates embeddings of images for all classes.
def create_all_img_embedding(clip_processor, clip_model,device,num_classes, num_img_4_class, name_model):
    tokenizer = CLIPTokenizer.from_pretrained(name_model)
    tokenizer.use_fast = False


    if os.path.exists('./images_embedding') == False:
        os.mkdir('./images_embedding')
        for i in tqdm(range(num_classes),desc="Calculating embeddings of all images"):
            class_embeddings =[]
            for j in range(num_img_4_class):
                img = efc.choose_class_and_img(i,j)
                image_path = img['path']
                image = Image.open(image_path).convert("RGB")
                image_inputs = clip_processor(images=image, return_tensors="pt")
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                with torch.no_grad():
                    img_emb = clip_model.get_image_features(**image_inputs)
                #I use squeeze to reduce the dimensionality of the tensor, then i save tensor copy in the CPU.
                #Compatibility with pcs without a GPU
                img_emb= img_emb.squeeze(0).cpu()
                class_embeddings.append(img_emb)

            # Stack the 5 embeddings in one tensor [5, D]
            class_embeddings = torch.stack(class_embeddings)

            # Save the tensor in a file .pt in a directory for the class
            class_dir = f'./images_embedding/class_{i}'
            if not efc.os.path.exists(class_dir):
                efc.os.mkdir(class_dir)

            save_path = f'{class_dir}/embeddings.pt'
            torch.save(class_embeddings, save_path)

            #print(f"Saved embeddings for class {i} at {save_path}")
    else:
        print('The Embedding Directory is already create, also the embedding of all image of Dataset.')     


#--------------------------------------------------------------------------------------------------------------


# Function to calculate embeddings of a single class -> put the prompt in the models.
# So if the class number is 1000 and we have 5 models we have to do 5000 prompt embeddings.        

def create_prompt_embedding_for_all(clip_processor, clip_model,model_name ,models,class_num, device,num_prompts=3):
    struct_of_text_embeddings = [
    [[] for _ in range(len(models))] 
    for _ in range(class_num)]  # class_num x num_models

    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    tokenizer.use_fast = True 
    message_prompt = efc.build_message_prompt()
    # To measure the average time to compute the embedding of a prompt for each model
    avg_time = [0.0]*len(models) 

    for i in range(len(models)):
            efc.setup_model(models[i])
            time_start = time.time()
            for j in tqdm(range(class_num),desc=f"Calculating prompt extension and text embeddings for model {models[i]}"):
                img=efc.choose_class_and_img(j,1)
                captions = []
                #This is the prompt extension of the class
                #print(img['class name'])
                for _ in range(num_prompts):  # different captions for each class
                    caption, t=efc.chat_with_model(img['class name'], models[i], message_prompt)
                    if models[i] == "deepseek-r1:14b":
                        caption = efc.extract_short_long(caption)
                    captions.append(caption)       
                text_inputs = clip_processor(text=captions, return_tensors="pt", padding=True, truncation=True, max_length=77)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                with torch.no_grad():
                    text_embs = clip_model.get_text_features(**text_inputs)                    
                for emb in text_embs:
                    struct_of_text_embeddings[j][i].append(emb.cpu())
                    #print(f"Saved embedding for class {j} and model {models[i]}")
            time_end = time.time()
            avg_time[i] = (time_end - time_start) / (class_num*num_prompts)    
    return struct_of_text_embeddings, avg_time


#-----------------------------------------------------------------------------------------------------------------------

# Embedding text is one vector of one class
# Models is the list of models
# Embeddings image is the tensor of all embeddings of images of one class [5,D]
def calculate_clip_score(embedding_text, models, embeddings_image ):
    average_clip_scores = [
    {
        "model":j ,
        "scores": [],
        "avg_score": 0.0,    # la media
        "deviation": 0.0  # la deviazione standard
    }
    for j in range(len(models))]
    
    for emb_img in embeddings_image:
        for i in range(len(models)):
            score = 0.0
            for k in range(len(embedding_text[i])):
                score += clip_score(emb_img, embedding_text[i][k])
            score /= len(embedding_text[i])      
            average_clip_scores[i]["scores"].append(score)

    # Calculate average scores
    for entry in average_clip_scores:
        if entry["scores"]:
            entry["avg_score"] = sum(entry["scores"]) / len(entry["scores"])
            entry["deviation"] = np.std(entry["scores"])
        else:
            entry["avg_score"] = 0.0

    return average_clip_scores                



# Calculate the clip score between one image embedding and one text embedding
def clip_score(emb_img, emb_text, w=2.5):
    # Normalize embeddings
    img_emb = emb_img / emb_img.norm()
    text_emb = emb_text / emb_text.norm()

    # Cosine similarity
    cosine_sim = torch.dot(img_emb, text_emb).item()
    
    #if the cosine similarity is negative we clip it to 0
    #We don't want to penalize the score if the cosine similarity is negative
    score = w * max(cosine_sim, 0.0)
    return score


# calculate clip scores for all classes 
def analyze(models, num_classes, embeddings_prompt):
    avg_clip_scores = []

    for i in range(num_classes):
        # Load image embeddings for class i
        embeddings_path = f'./images_embedding/class_{i}/embeddings.pt'
        embeddings_image = torch.load(embeddings_path)
        avg_clip_scores.append(calculate_clip_score(embeddings_prompt[i], models, embeddings_image))
        print(f"Class {i} - Average CLIP Scores: {avg_clip_scores[i]}")

    return avg_clip_scores


