from PIL import Image

import env_for_chatting as efc
import random as rdm
import re
import torch
from transformers import CLIPProcessor, CLIPModel

NUM_CLASS = 1000
NUM_IMG_4_CLASS = 5
LEN = 10  
model_name = "openai/clip-vit-base-patch16"

# Function to compute CLIP score for an image and text
# The score is calculated as w * max(cosine_similarity, 0)
# if clamp_zero is False, the score is w * cosine_similarity
# This because if is True the average CLIP score is higher because negative scores are clamped to zero.
def compute_clip_score(image_path, text, w=2.5,clamp_zero=False):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_emb = outputs.image_embeds[0]
        text_emb = outputs.text_embeds[0]

    # Normalization (cosine similarity)
    image_emb = image_emb / image_emb.norm()
    text_emb = text_emb / text_emb.norm()
    cosine_sim = torch.dot(image_emb, text_emb).item()

    if clamp_zero:
        # Apply max(., 0) e weights w
        score = w * max(cosine_sim, 0.0)
    else:
        score = w * cosine_sim
    return score

# Function to calculate average scores and response times for each model
def avg_score(score_dict):
    avg_scores = {}
    for model, values in score_dict.items():
        scores = values["clipscores"]
        times = values["response_time"]

        avg_clip = sum(scores) / len(scores) if scores else 0
        avg_time = sum(times) / len(times) if times else 0

        avg_scores[model] = {"avg_clipscore": avg_clip, "avg_time": avg_time}
    return avg_scores


def extract_short_long(response_text):
    short_match = re.search(r"Short:\s*(.*)", response_text)
    short = short_match.group(1).strip() if short_match else ""
    return short


if __name__ == '__main__':
    models = ["mistral:latest", "phi3:latest", "llama3.2:latest", "gemma2:latest","deepseek-r1:14b"]

    # Load the CLIP model and processor
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)

    #Initialize the envirionment for store clipscores and response times
    score_clip = {model: {"clipscores": [0] * LEN, "response_time": [0.0] * LEN} for model in models}

    #take two random number that corresponds to a class and a photo
    img_class = rdm.randrange(NUM_CLASS)
    img_num = rdm.randrange(NUM_IMG_4_CLASS)

    #In this structure we save the class and the photo number
    img = {"class": img_class, "photo": img_num}

    #Save the path of the choosen image
    choosen_img =efc.choose_img(img["class"], img["photo"])

    #Visualize the choosen image
    Image.open(choosen_img).show()

    #Calculate CLIP scores and response times for each model
    for i in range(LEN):
        print(f"ITERATION N°: {i}")
        captions = efc.make_all_conversation(models, img)
        score_clip["mistral:latest"]["clipscores"][i] = compute_clip_score(choosen_img, captions["mistral:latest"]["response"])
        score_clip["mistral:latest"]["response_time"][i] = captions["mistral:latest"]["response_time"]
        score_clip["phi3:latest"]["clipscores"][i] = compute_clip_score(choosen_img, captions["phi3:latest"]["response"])
        score_clip["phi3:latest"]["response_time"][i] = captions["phi3:latest"]["response_time"]
        score_clip["llama3.2:latest"]["clipscores"][i] = compute_clip_score(choosen_img, captions["llama3.2:latest"]["response"])
        score_clip["llama3.2:latest"]["response_time"][i] = captions["llama3.2:latest"]["response_time"]
        score_clip["gemma2:latest"]["clipscores"][i] = compute_clip_score(choosen_img, captions["gemma2:latest"]["response"])
        score_clip["gemma2:latest"]["response_time"][i] = captions["gemma2:latest"]["response_time"]
        short =extract_short_long(captions["deepseek-r1:14b"]["response"])
        string = f"Short: {short}"
        print(string)

        score_clip["deepseek-r1:14b"]["clipscores"][i] = compute_clip_score(choosen_img, string)
        score_clip["deepseek-r1:14b"]["response_time"][i] = captions["deepseek-r1:14b"]["response_time"]

    # Calculate average scores and response times
    average_scores = avg_score(score_clip)

    # Print the average scores and response times for each model
    print(f"[mistral] CLIP-S: {average_scores['mistral:latest']['avg_clipscore']:.3f}")
    print(f"[mistral] Average Time to Respond: {average_scores['mistral:latest']['avg_time']:.3f} seconds")
    print(f"[phi3] CLIP-S: {average_scores['phi3:latest']['avg_clipscore']:.3f}")
    print(f"[phi3] Average Time to Respond: {average_scores['phi3:latest']['avg_time']:.3f} seconds")
    print(f"[llama3.2] CLIP-S: {average_scores['llama3.2:latest']['avg_clipscore']:.3f}")
    print(f"[llama3.2] Average Time to Respond: {average_scores['llama3.2:latest']['avg_time']:.3f} seconds")
    print(f"[gemma2] CLIP-S: {average_scores['gemma2:latest']['avg_clipscore']:.3f}")
    print(f"[gemma2] Average Time to Respond: {average_scores['gemma2:latest']['avg_time']:.3f} seconds")
    print(f"[deepseek-r1] CLIP-S: {average_scores['deepseek-r1:14b']['avg_clipscore']:.3f}")
    print(f"[deepseek-r1] Average Time to Respond: {average_scores['deepseek-r1:14b']['avg_time']:.3f} seconds") 
