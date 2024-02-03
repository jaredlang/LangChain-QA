import os
import time
from dotenv import load_dotenv
import requests
import io
from PIL import Image

from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
import openai

load_dotenv()
HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"]

def generate_image_hf(image_description: str): 
    # Huggingface model: 
    # API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
    headers = {"Authorization": F"Bearer {HUGGINGFACE_API_KEY}"}
    
    response = requests.post(API_URL, headers=headers, json={
        "inputs": image_description,
    })
    
    return response.content


def generate_image_dalle(image_description: str): 
    # DALL-E-3 allows up to 4000 charaters
    dalle_model = "dall-e-3"
    if len(image_description_text) < 1000: 
        # DALL-E2 doesn't accept a prompt longer than 1000 characters. 
        dalle_model = "dall-e-2"

    print("USE MODEL: ", dalle_model)

    image_url = DallEAPIWrapper(model=dalle_model).run(image_description_text)
    print("IMAGE URL: ", image_url)

    # # Another way to import a class with the same name 
    # openAIClient = openai.OpenAI()
    # image2_url = openAIClient.images.generate(prompt=image_description_text)
    # print("IMAGE2 URL: ", image2_url)

    response = requests.get(image_url)

    return response.content 


llm = OpenAI(temperature=0.9)

prompt = PromptTemplate(
    input_variables=["image_desc"],
    #     The prompt must be less than 150 words long. 
    template="""Generate a detailed prompt to generate an image based on the following description. 
    DESCRIPTION:
    {image_desc}
    """,
)

chain = LLMChain(llm=llm, prompt=prompt)

image_description = chain.invoke(
#    "A sunlit indoor lounge area with a pool containing a flamingo"
#    "sunny sky with plenty of sunshine and no cloud"
#    "cloud sky"
#    "Astronaut riding a horse"
#    "halloween night at a haunted museum"
    "beautiful girl lying on beach"
#    "In her flaming red dress, she sat amidst the emerald field, a beacon of solitude, silently battling her unseen demons under the open sky."
#    "A beleaguered man sat, phone in hand, anxiously awaiting a call that could change his life forever."
)

print("DESCRIPTION: ", image_description)

image_description_text = image_description["text"]

# a poor man's fallback 
# try:
#     image_bytes = generate_image_dalle(image_description_text)
# except Exception as e: 
#     print(f"Error: {e}")
#     image_bytes = generate_image_hf(image_description_text)

# swtich from OpenAI to Stable Diffusion on HuggingFace. 
# Free and much better quality 
image_bytes = generate_image_hf(image_description_text)

# You can access the image with PIL.Image for example
image = Image.open(io.BytesIO(image_bytes))

ts = time.time()
image_format = "png"
image.save(f"./output/image-{ts}.{image_format}", format=image_format)
