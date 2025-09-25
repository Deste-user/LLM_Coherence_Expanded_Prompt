import matplotlib.pyplot as plt   
import numpy as np
import random as rdm
from PIL import Image
import json
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.styles import Border, Side, PatternFill
from openpyxl.formatting.rule import CellIsRule

import env_for_chatting as efc
import analysis as a

#Review this function
#This function create a grafic that show the stability of clip scores over iterations

def grafic_analysis( models, num_runs, num_iter, num_classes, clip_processor, clip_model):
    img_class = rdm.randrange(num_classes)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer.use_fast = True

    #In this structure we save the class and the photo number
    img = {"class": img_class, "photo": 1}

    #Save the path of the choosen image
    embedded_img_path = f"./images_embedding/class_{img_class}/embeddings.pt"

    emb_img= torch.load(embedded_img_path)

    #Visualize the choosen image
    choosen_img =efc.choose_img(img["class"], 1)
    Image.open(choosen_img).show()

    #Create a vector with the number of iterations for each run
    vect_iter = [j*num_iter for j in range(1,num_runs+1)]
    print(vect_iter)

    #Initialize a structure to save the average clip scores for each model at each run
    avg_clip_scores = {model: [0.0] * num_runs 
                       for model in models}

     #Initialize the envirionment for store clipscores and response times
    score_clip = {model: {"clipscores": [0] * num_iter, "response_time": [0.0] * num_iter} for model in models}


    #Iterate for the number of runs
    for i in range(1,num_runs+1):
        #Calculate CLIP scores and response times for each model
        for j in range(num_iter*i):
            print(f"ITERATION N°: {j}")
            #Comunicate with all models
            captions = efc.make_all_conversation(models, img)
            #Iterate over all models
            #make embeddings for all models

            for k in range(len(models)):
                #Check if the model is deepseek-r1:14b because I want only the foundamental part of the answer
                if models[k] == "deepseek-r1:14b":
                    captions[models[k]]["response"] = efc.extract_short_long(captions[models[k]]["response"])
                #Create the embedding of the prompt    
                text_inputs = clip_processor(text=[captions[models[k]]["response"]], return_tensors="pt", padding=True, truncation=True, max_length=77)
                text_inputs = {l: v.to(clip_model.device) for l, v in text_inputs.items()}
                with torch.no_grad():
                    text_emb = clip_model.get_text_features(**text_inputs)
                text_emb = text_emb.squeeze(0).cpu()
                #Calculate the clip score
                score = a.clip_score(emb_img[1], text_emb)
                #Accumulate the clip score
                score_clip[models[k]]["clipscores"][i-1] += score
        
        for model in models:
            #Calculate the average clip score for the model at the i-th run
            score_clip[model]["clipscores"][i-1] /= num_iter*i
            #Append the average clip score to the avg_clip_scores structure
            avg_clip_scores[model][i-1]=score_clip[model]["clipscores"][i-1]

    # So we have avg_clip_scores[model]= [0.0, score1, score2, ..., scoreN] where i-score is the average clip score at group of iterations i*num_iter
    # Function to create a bar graph comparing average CLIP scores across different models for each class
    # Plotting        
    plt.figure()
    for k in range(len(models)):
        plt.plot(vect_iter, avg_clip_scores[models[k]], marker='o', label=models[k])
    plt.xlabel('Number of Iterations')
    plt.ylabel('Average CLIP Score')
    plt.title(f'Average CLIP Scores over Iterations for Class {img_class}')
    plt.legend()
    plt.grid(True)
    plt.savefig("result.png")
        
#--------------------------------------------------------------------------------------------------------------

# Function to create a bar graph comparing average CLIP scores across different models for each class
def create_clipscore_table(avg_clip_scores, models, num_classes):
            
    wb= Workbook()
    ws= wb.active
    ws.title= "ClipScores"

    thin_border = Border(
        left=Side(style='medium'),
        right=Side(style='medium'),
        top=Side(style='medium'),
        bottom=Side(style='medium')
    )

    # Create header
    col_start = 2
    for model in models:
        col_end = col_start + 1  
        ws.merge_cells(start_row=1, start_column=col_start, end_row=1, end_column=col_end)
        cell = ws.cell(row=1, column=col_start, value=model)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
        cell.border = thin_border
        col_start += 2 

    class_cell=ws.cell(row=1, column=1, value="Class").font = Font(bold=True)
    class_cell.border = thin_border
    class_cell.alignment = Alignment(horizontal='center')

    col_start = 2
    for _ in range(len(models)):
        c1=ws.cell(row=2, column=col_start, value="avg")
        c1.font = Font(bold=True)
        c1.border = thin_border
        c1.alignment = Alignment(horizontal='center')
        
        c2=ws.cell(row=2, column=col_start+1, value="dev std")
        c2.font = Font(bold=True)
        c2.border = thin_border
        c2.alignment = Alignment(horizontal='center')

        col_start += 2

    
    with open('./Small-ImageNet-Validation-Dataset-1000-Classes/imagenet_class_index.json', 'r',
              encoding='utf-8') as file:
        map_class = json.load(file)

    map_class = list(map_class.items())


    sum_all_avg = 0.0
    for model in models:
        for class_idx in range(num_classes):
            sum_all_avg += avg_clip_scores[class_idx][models.index(model)]["avg_score"]
    avg_threshold = sum_all_avg / (num_classes * len(models))        
    
    # Fill in data

    for class_idx in range(num_classes):
        cell_class= ws.cell(row=class_idx+3, column=1, value=f"{class_idx} - {map_class[class_idx][1][1]}")
        cell_class.border = thin_border
        cell_class.alignment = Alignment(horizontal='center')
        col_start = 2
        for model in models:
            avg_score= avg_clip_scores[class_idx][models.index(model)]["avg_score"]
            deviation= avg_clip_scores[class_idx][models.index(model)]["deviation"]
            cell_avg= ws.cell(row=class_idx+3,column=col_start, value=round(avg_score,4))
            cell_avg.border = thin_border
            cell_avg.alignment = Alignment(horizontal='center')
            cell_dev= ws.cell(row=class_idx+3,column=col_start+1, value=round(deviation,4))
            cell_dev.border = thin_border
            cell_dev.alignment = Alignment(horizontal='center')
            col_start += 2

    red_fill = PatternFill(start_color="FF9999", end_color="FF9999", fill_type="solid")
    green_fill = PatternFill(start_color="99FF99", end_color="99FF99", fill_type="solid")


    for row in range(3, num_classes+3):
        col=2
        for model in models:
            ws.conditional_formatting.add(
                f"{ws.cell(row=row, column=col).coordinate}",
                CellIsRule(operator="lessThan", formula=[str(avg_threshold)], fill=red_fill)
            )
            ws.conditional_formatting.add(
                f"{ws.cell(row=row, column=col).coordinate}",
                CellIsRule(operator="greaterThanOrEqual", formula=[str(avg_threshold)], fill=green_fill)
            )
            col += 2

    wb.save("ClipScores.xlsx")




               

    
