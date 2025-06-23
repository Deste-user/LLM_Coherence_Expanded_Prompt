#we are using requests, that it has the same functionality of curl.
import requests
import os
import subprocess
import time
import base64
from PIL import Image

#Define the URL for the port
URL_OLLAMA = "http://localhost:11434"
#DEFINE string in input
question = "Expand the image"

def image_to_base64(image_path):
    with open(image_path,"rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def choose_img(class_image, index):
    path = f"./Small-ImageNet-Validation-Dataset-1000-Classes/ILSVRC2012_img_val_subset/{class_image}/"
    files = sorted(os.listdir(path))
    count=0
    find=False
    for f in files:
        if count==index:
            find=True
            break
        count=count+1

    if find == False:
        print("Error, no file is present with those class and index")
        return None
    else:
        return path + f


def download_dataset():
    dataset_dir = "./Small-ImageNet-Validation-Dataset-1000-Classes"
    if not os.path.exists(dataset_dir):
        print("📦 Clonazione del dataset da GitHub...")
        subprocess.run(["git", "clone", "https://github.com/ndb796/Small-ImageNet-Validation-Dataset-1000-Classes.git"])
    else:
        print("✅ Dataset già presente.")
def chat_with_model( message_prompt, data_image, model_name):
    setup_model(model_name)
    if(data_image == None):
        try:
            response = requests.post("http://localhost:11434/api/generate",
                                 json={ "model": model_name,"prompt": message_prompt,"stream": False,})
        except requests.exceptions.ConnectionError:
            pass

        response_from_chat= response.json()
        return response_from_chat["response"]
    else:
        try:
            response = requests.post("http://localhost:11434/api/generate",
                                     json={"model": model_name, "prompt": message_prompt, "stream": False,"images": [data_image]})
            #print(response.json())
            response_from_chat=response.json()
            return response_from_chat["response"]
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

#With this funct we pull some models, choosing the name of LLM
def setup_model(model_name):
    print(f"Check if  '{ model_name }' is already present")
    response= requests.get("http://localhost:11434/api/tags")
    installed_models = []
    for m in response.json().get("models", []):
        installed_models.append(delete_last_part(m["name"]))
    checked=False
    for m in installed_models:
        if (m == model_name):
            checked=True
            print("Modello già presente")
            break
    if checked == False:
        print("Modello non presente")
        pull_response= requests.post("http://localhost:11434/api/pull", json={"name": model_name}, stream=True)

        if pull_response.status_code != 200:
            raise RuntimeError(f"Pull fallito per il modello '{model_name}'. Codice Errore: '{pull_response.status_code}'")
        else:
            for line in pull_response.iter_lines():
                if line:
                    print(line.decode('utf-8'))

            print(f" Modello '{model_name}' installato con successo.")

# To stop the process that is on in the port 11434
def stop_ollama():
    subprocess.Popen(["pkill", "ollama"])
    response=requests.get("http://localhost:11434")
    if response.status_code!=200:
        print("Connection is close")

if __name__ == '__main__':
    download_dataset()
    start_ollama()
    img_path = choose_img(1,3)
    if img_path== None:
        print("Error")
    else:
        # Now we want to generate a expansion of an image.
        img=Image.open(img_path).show()
        img64=image_to_base64(img_path)
        response= chat_with_model(question, img64 , "llava")
        print(f"Question1: '{question}'")
        print(f"Response1: '{response}'")
        response = chat_with_model(question, img64, "moondream")
        print(f"Question2: '{question}'")
        print(f"Response2: '{response}'")
    stop_ollama()




