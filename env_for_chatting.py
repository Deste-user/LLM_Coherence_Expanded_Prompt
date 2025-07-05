# we are using requests, that it has the same functionality of curl.
import json

import requests
import os
import subprocess
import time
import base64
from PIL import Image
from io import BytesIO

# Define the URL for the port
URL_OLLAMA = "http://localhost:11434"
# DEFINE string in input
question = "Describe the phisical appearence of this word in a flowing paragraph, avoiding bullet points: "


def image_to_base64(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # assicurati RGB
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


def choose_img(class_image, index):
    path = f"./Small-ImageNet-Validation-Dataset-1000-Classes/ILSVRC2012_img_val_subset/{class_image}/"
    files = sorted(os.listdir(path))
    count = 0
    find = False
    for f in files:
        if count == index:
            find = True
            break
        count = count + 1

    if find == False:
        print("Error, no file is present with those class and index")
        return None
    else:
        return path + f


def choose_class(class_image, index):
    with open('./Small-ImageNet-Validation-Dataset-1000-Classes/imagenet_class_index.json', 'r',
              encoding='utf-8') as file:
        map_class = json.load(file)

    map_class = list(map_class.items())
    path_img = choose_img(class_image, index)
    obj = {"class name": map_class[class_image][1][1], "path": path_img}
    return obj


def download_dataset():
    dataset_dir = "./Small-ImageNet-Validation-Dataset-1000-Classes"
    if not os.path.exists(dataset_dir):
        print("📦 Clonazione del dataset da GitHub...")
        subprocess.run(["git", "clone", "https://github.com/ndb796/Small-ImageNet-Validation-Dataset-1000-Classes.git"])
    else:
        print("✅ Dataset già presente.")


def chat_with_model(message_prompt, class_image, model_name):
    setup_model(model_name)
    input_string = message_prompt + class_image
    input_string = input_string.replace("_", " ")

    try:
        start = time.time()
        response = requests.post("http://localhost:11434/api/generate",
                                 json={"model": model_name, "prompt": input_string, "stream": False})
        response_from_chat = response.json()
        end = time.time()
        t = end - start
        # Verifica chiave 'response'
        if "response" in response_from_chat:
            return response_from_chat["response"], t
        else:
            raise KeyError(f"'response' non presente. Risposta ricevuta: {response_from_chat}")
    except requests.exceptions.ConnectionError:
        pass


# To modify the model name that ends always with :latest
def delete_last_part(string):
    return string.split(":")[0]


# To initialize the ollama sw in a determinated port (11434)
def start_ollama():
    try:
        requests.get("http://localhost:11434")
        print("Ollama è già in esecuzione.")
        return
    except requests.exceptions.ConnectionError:
        pass

    print("Avvio Ollama in modalità 'serve'...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(10):
        try:
            r = requests.get("http://localhost:11434")
            if r.status_code in [200, 404]:
                print("Ollama serve è attivo.")
                return
        except requests.exceptions.ConnectionError:
            time.sleep(1)

    raise RuntimeError("Ollama non si è avviato.")


# With this funct we pull some models, choosing the name of LLM
def setup_model(model_name):
    print(f"Check if  '{model_name}' is already present")
    response = requests.get("http://localhost:11434/api/tags")
    installed_models = []
    for m in response.json().get("models", []):
        installed_models.append(delete_last_part(m["name"]))
    checked = False
    for m in installed_models:
        if (m == model_name):
            checked = True
            print("Modello già presente")
            break
    if checked == False:
        print("Modello non presente")
        pull_response = requests.post("http://localhost:11434/api/pull", json={"name": model_name}, stream=True)

        if pull_response.status_code != 200:
            raise RuntimeError(
                f"Pull fallito per il modello '{model_name}'. Codice Errore: '{pull_response.status_code}'")
        else:
            for line in pull_response.iter_lines():
                if line:
                    print(line.decode('utf-8'))

            print(f" Modello '{model_name}' installato con successo.")


# This function is used to setup a chat and use it.
# It returns an array of caption in output from model.

def to_format_string(string):
    char = ["\n", "\\", "/", "**"]
    for c in char:
        string = string.replace(c, " ")
    return string


def make_all_conversation(models, img):
    download_dataset()
    start_ollama()
    # img_path = choose_img(img["class"], img["photo"])
    return_value = choose_class(img["class"], img["photo"])
    response = {model: {"response": "", "response_time": 0.0} for model in models}
    if return_value == None:
        print("Error")
    else:
        # Now we want to generate a expansion of an image.

        img64 = image_to_base64(return_value["path"])
        for m in models:
            resp, tm = chat_with_model(question, return_value["class name"], m)
            response[m]["response_time"] = tm
            response[m]["response"] = to_format_string(resp)

    print(f"Question: '{question}'")
    print(f"Response: '{response}'")
    return response
