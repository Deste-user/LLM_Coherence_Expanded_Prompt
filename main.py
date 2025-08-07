from PIL import Image

import env_for_chatting as efc
import random as rdm
import torch
from transformers import CLIPProcessor, CLIPModel

NUM_CLASS = 1000
NUM_IMG_4_CLASS = 5
LEN = 10
model_name = "openai/clip-vit-base-patch16"

def compute_clip_score(image_path, text, w=2.5):
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

    # Normalizzazione (cosine similarity)
    image_emb = image_emb / image_emb.norm()
    text_emb = text_emb / text_emb.norm()
    cosine_sim = torch.dot(image_emb, text_emb).item()

    # Applica max(., 0) e peso w
    score = w * max(cosine_sim, 0.0)
    return score


def avg_score(score_dict):
    avg_scores = {}
    for model, values in score_dict.items():
        scores = values["clipscores"]
        times = values["response_time"]

        avg_clip = sum(scores) / len(scores) if scores else 0
        avg_time = sum(times) / len(times) if times else 0

        avg_scores[model] = {"avg_clipscore": avg_clip, "avg_time": avg_time}
    return avg_scores


if __name__ == '__main__':
    models = ["mistral", "phi3", "llama3.2"]
    #take two random number that corresponds to a class and a photo
    img_class = rdm.randrange(NUM_CLASS)
    img_num = rdm.randrange(NUM_IMG_4_CLASS)

    #
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)

    #Initialize the envirionment for store clipscores and response times
    score_clip = {model: {"clipscores": [0] * LEN, "response_time": [0.0] * LEN} for model in models}

    #In this structure we save the class and the photo number
    img = {"class": img_class, "photo": img_num}

    #Save the path of the choosen image
    choosen_img =efc.choose_img(img["class"], img["photo"])


    #Visualize the choosen image
    Image.open(choosen_img).show()

    for i in range(LEN):
        print(f"ITERATION N°: {i}")
        captions = efc.make_all_conversation(models, img)
        score_clip["mistral"]["clipscores"][i] = compute_clip_score(choosen_img, captions["mistral"]["response"])
        score_clip["mistral"]["response_time"][i] = captions["mistral"]["response_time"]
        score_clip["phi3"]["clipscores"][i] = compute_clip_score(choosen_img, captions["phi3"]["response"])
        score_clip["phi3"]["response_time"][i] = captions["phi3"]["response_time"]
        score_clip["llama3.2"]["clipscores"][i] = compute_clip_score(choosen_img, captions["llama3.2"]["response"])
        score_clip["llama3.2"]["response_time"][i] = captions["llama3.2"]["response_time"]

    average_scores = avg_score(score_clip)
    print(f"[mistral] CLIP-S: {average_scores['mistral']['avg_clipscore']:.3f}")
    print(f"[mistral] Average Time to Respond: {average_scores['mistral']['avg_time']:.3f} seconds")
    print(f"[phi3] CLIP-S: {average_scores['phi3']['avg_clipscore']:.3f}")
    print(f"[phi3] Average Time to Respond: {average_scores['phi3']['avg_time']:.3f} seconds")
    print(f"[llama3.2] CLIP-S: {average_scores['llama3.2']['avg_clipscore']:.3f}")
    print(f"[llama3.2] Average Time to Respond: {average_scores['llama3.2']['avg_time']:.3f} seconds")
