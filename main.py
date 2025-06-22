#we are using requests, that it has the same functionality of curl.
import requests
import os
import subprocess
import time
def download_dataset():
    dataset_dir = "imagenet_validation_dataset"
    if not os.path.exists(dataset_dir):
        print("📦 Clonazione del dataset da GitHub...")
        subprocess.run(["git", "clone", "https://github.com/ndb796/Small-ImageNet-Validation-Dataset-1000-Classes.git"])
    else:
        print("✅ Dataset già presente.")
def chat_with_model( message_prompt, data_image, model_name):
    if(data_image == None):
        try:
            response = requests.post("http://localhost:11434/api/generate",
                                 json={ "model": model_name,"prompt": message_prompt,"stream": False,})
        except requests.exceptions.ConnectionError:
            pass

        response_from_chat= response.json()
        return response_from_chat["response"]

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
    setup_model("mistral")
    question= "Define the Happiness"
    response=chat_with_model(question,None,"mistral")
    print(f"Question: '{question}'")
    print(f"Response: '{response}'")

    stop_ollama()




