# we are using requests, that it has the same functionality of curl.
import json
import requests
import os
import re
import time
import base64
from PIL import Image
from io import BytesIO

# Define the URL for the port
URL_OLLAMA = "http://localhost:11434"

# DEFINE string in input
question =  "\n Please improve this prompt: "

PREPARATED_PROMPT = "You are a helpful AI assistant designed to help the user produce prompts that will be used to generate images using Stable Diffusion." \
" Help the user by modifying a given prompt adding many details so that the generated images will look better. " \
"Generate only realistic prompts. Do not engage in artistic and unrealistic prompts."

EXAMPLES = [
    ('a portrait of a blonde woman', 'portrait of a pretty blonde woman, a flower crown, earthy makeup, flowing maxi dress with colorful patterns and fringe, a sunset or nature scene, green and gold color scheme'),

    ('a portrait of an old man', 'photorealistic, visionary portrait of a dignified older man with weather-worn features, digitally enhanced, high contrast, chiaroscuro lighting technique, intimate, close-up, detailed, steady gaze, rendered in sepia tones, evoking rembrandt, timeless, expressive, highly detailed, sharp focus, high resolution'),

    ('a realistic picture of a living room', 'a living room, bright modern Scandinavian style house, large windows, magazine photoshoot, 8k, studio lighting'),

    ('a closeup of a goth woman', 'closeup portrait photo of beautiful goth woman, makeup, 8k uhd, high quality, dramatic, cinematic'),

    ('a picture of a rabbit in a forest', 'close up photo of a rabbit, forest in spring, haze, halation, bloom, dramatic atmosphere, centred, rule of thirds, 200mm 1.4f macro shot'),

    ('a photo of an indian girl', 'happy indian girl, portrait photography, beautiful, morning sunlight, smooth light, shot on kodak portra 200, film grain, nostalgic mood'),

    ('a professional picture of a luxury bag', 'breathtaking shot of a bag, luxury product style, elegant, sophisticated, high-end, luxurious, professional, highly detailed'),

    ('johnny depp, noir style', 'johnny depp photo portrait, film noir style, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic'),

    ('a cinematic shot of a cat in the snow', 'a cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo, animal'),

    ('tokyo, long exposure', 'long exposure photo of tokyo street, blurred motion, streaks of light, surreal, dreamy, ghosting effect, highly detailed'),

    ('a photoshoot of a model, cyberpunk style ', 'a glamorous digital magazine photoshoot, a fashionable model wearing avant-garde clothing, set in a futuristic cyberpunk roof-top environment, with a neon-lit city background, intricate high fashion details, backlit by vibrant city glow, Vogue fashion photography'),

    ('floral tea', 'freshly made hot floral tea in glass kettle on the table, angled shot, midday warm, Nikon D850 105mm, close-up'),

    ('a picture of a smiling girl, red hair, upper body shot', 'masterpiece, best quality, girl, collarbone, wavy hair, looking at viewer, blurry foreground, upper body, necklace, contemporary, plain pants, intricate, print, pattern, ponytail, freckles, red hair, dappled sunlight, smile, happy'),
]

# Function to format examples for the prompt
# It returns a string with each example formatted
def format_examples(examples=EXAMPLES):
    return "\n".join([f"Short: {short}\n Long: {long}\n" for short, long in examples])

# Function to build the message prompt with examples
#TODO: Probably the prompt is too long and it superate the max token limit of the model.
def build_message_prompt():
    examples = format_examples()
    return (
        PREPARATED_PROMPT +
        "\n\n---\n\n"
        "Here are some EXAMPLES (for reference only, DO NOT copy them in the answer):\n\n" +
        examples +
         "\n\n---\n\n"
        "TASK:\n"
        "You must improve the given prompt.\n\n"
        "STRICT OUTPUT RULES:\n"
        "1. Do NOT include reasoning, explanations, meta-comments, or thought processes.\n"
        "2. Do NOT repeat the examples above.\n"
        "3. Your answer MUST follow EXACTLY this format:\n"
        "   Short: <short improved version>\n"
        "   Long: <long detailed version>\n"
        "4. Provide exactly ONE Short and ONE Long.\n"
        "5. Do not add any text before or after these two lines.\n"
        "\nPrompt to improve:\n"
    )


# Open the image and convert it in base64 format
# This is used to send the image to the model
def image_to_base64(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")  
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

#Choose the image based on the class and the photo number
def choose_img(class_image, index):
    path = f"./Small-ImageNet-Validation-Dataset-1000-Classes/ILSVRC2012_img_val_subset/{class_image}/"
    # Order the files in the directory
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


def choose_class_and_img(class_image, index):
    with open('./Small-ImageNet-Validation-Dataset-1000-Classes/imagenet_class_index.json', 'r',
              encoding='utf-8') as file:
        map_class = json.load(file)

    map_class = list(map_class.items())
    path_img = choose_img(class_image, index)
    obj = {"class name": map_class[class_image][1][1], "path": path_img}
    return obj

# #Function to Download the dataset if it not present
# def download_dataset():
#     dataset_dir = "./Small-ImageNet-Validation-Dataset-1000-Classes"
#     if not os.path.exists(dataset_dir):
#         print(" Clonation of the dataset from GitHub...")
#         subprocess.run(["git", "clone", "https://github.com/ndb796/Small-ImageNet-Validation-Dataset-1000-Classes.git"])
#     else:
#         print("Dataset is already present.")

# Function used to chat with the model
# It sends a request to the model with the image and the message prompt
def chat_with_model(class_image, model_name, message_prompt):
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
        elif "error" in response_from_chat:
            print(f"Error from model {model_name}: {response_from_chat['error']}")
            return "", t
        else:
            raise KeyError(f"Unexpected response: {response_from_chat}")
    except requests.exceptions.ConnectionError:
        pass


# To modify the model name that ends always with :latest
def delete_last_part(string):
    return string.split(":")[0]


# # To initialize the ollama sw in a determinated port (11434)
# def start_ollama():
#     try:
#         requests.get("http://localhost:11434")
#         print("Ollama is already running.")
#         return
#     except requests.exceptions.ConnectionError:
#         pass

#     print("Run Ollama in 'serve' mode...")
#     subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     for _ in range(10):
#         try:
#             r = requests.get("http://localhost:11434")
#             if r.status_code in [200, 404]:
#                 print("Ollama serve is active.")
#                 return
#         except requests.exceptions.ConnectionError:
#             time.sleep(1)

#     raise RuntimeError("Ollama is not running.")


# With this funct we pull some models, choosing the name of LLM
def setup_model(model_name):
    print(f"Check if  '{model_name}' is already present")
    response = requests.get("http://localhost:11434/api/tags")
    installed_models = []
    for m in response.json().get("models", []):
        # We delete the last part of the model name to append the model name without the version
        installed_models.append(m["name"])
    checked = False
    for m in installed_models:
        if (m == model_name):
            checked = True
            print("Model is already present")
            break
    if checked == False:
        print("Model is not already present, we pull it...")
        pull_response = requests.post("http://localhost:11434/api/pull", json={"name": model_name}, stream=True)

        if pull_response.status_code != 200:
            raise RuntimeError(
                f"Failed Pull  for the model '{model_name}'. Error Code: '{pull_response.status_code}'")
        else:
            for line in pull_response.iter_lines():
                if line:
                    print(line.decode('utf-8'))

            print(f" Model '{model_name}' is successfully installated.")


# This function is used to setup a chat and use it.
# It returns an array of caption in output from model.

def to_format_string(string):
    char = ["\n", "\\", "/", "**"]
    for c in char:
        string = string.replace(c, " ")
    return string

# Function to make a conversation based on a image with a list of models
# It returns a dictionary with the response and the response time for each model
def make_all_conversation(models, img):
    
    # # Download the dataset if it not present
    # download_dataset()

    # # Start the ollama server
    # start_ollama()
    
    # Prepare the message prompt)
    message_prompt = build_message_prompt()

    # Choose the class and the photo number, Return a dictinonary with the class name and the path of the image
    return_value = choose_class_and_img(img["class"], img["photo"])

    #Initialize the response dictionary
    response = {model: {"response": "", "response_time": 0.0} for model in models}
    # If the return value is None, it means that there was an error in choosing the class and the photo
    if return_value == None:
        print("Error, there is no image with that class and index.")
    else:
        # Now we want to generate a expansion of an image.
        # img64 = image_to_base64(return_value["path"])
        for m in models:
            #Store the response and the time of response for each model
            resp, tm = chat_with_model(return_value["class name"], m, message_prompt)
            response[m]["response_time"] = tm
            response[m]["response"] = to_format_string(resp)

    print(f"Question: '{message_prompt + return_value['class name']}'")
    print(f"Response: '{response}'")
    return response

def extract_short_long(response_text):
    short_match = re.search(r"Short:\s*(.*)", response_text)
    short = short_match.group(1).strip() if short_match else ""
    return short
