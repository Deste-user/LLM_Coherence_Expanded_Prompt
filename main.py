from PIL import Image

import env_for_chatting as efc
import random as rdm
import torch
from transformers import CLIPProcessor, CLIPModel

NUM_CLASS=1000
NUM_IMG_4_CLASS=5
model_name = "openai/clip-vit-base-patch16"

def compute_clip_score(image_path, text):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True,
    max_length=77)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_emb = outputs.image_embeds[0]
        text_emb = outputs.text_embeds[0]

    image_emb = image_emb / image_emb.norm()
    text_emb = text_emb / text_emb.norm()
    score = torch.dot(image_emb, text_emb).item()
    return score

def avg_score(score_dict):
        avg_scores = {}
        for model, scores in score_dict.items():
            avg_scores[model] = sum(scores) / len(scores) if scores else 0
        return avg_scores




if __name__ == '__main__':
    models=["llava","moondream","gemma3"]
    img_class=rdm.randrange(NUM_CLASS)
    img_num=rdm.randrange(NUM_IMG_4_CLASS)

    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    score_clip = {model: [0] * 10 for model in models}

    img={"class":img_class, "photo":img_num}
    Image.open(efc.choose_img(img["class"],img["photo"])).show()
    for i in range(10):
        captions=efc.make_all_conversation(models,img)

        score_clip["llava"][i] = compute_clip_score(efc.choose_img(img["class"],img["photo"]),captions["llava"])
        score_clip["moondream"][i] = compute_clip_score(efc.choose_img(img["class"], img["photo"]), captions["moondream"])
        score_clip["gemma3"][i] = compute_clip_score(efc.choose_img(img["class"], img["photo"]), captions["gemma3"])

    average_scores=avg_score(score_clip)
    print(f"[llava] CLIP-S: {average_scores['llava']:.3f}")
    print(f"[moondream] CLIP-S: {average_scores['moondream']:.3f}")
    print(f"[gemma3] CLIP-S: {average_scores['gemma3']:.3f}")






