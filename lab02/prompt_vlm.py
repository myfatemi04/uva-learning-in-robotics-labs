import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
from io import BytesIO

model = AutoModelForCausalLM.from_pretrained(
    "qresearch/llama-3.1-8B-vision-378",
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    "qresearch/llama-3.1-8B-vision-378",
    use_fast=True,
)

prompt = """
You are a judge in a competition to move red blocks to green circles.
A robot is competing. This image shows the state of the robot. Please
describe the state of the robot's attempt.
""".strip()

for image_name in ['start', 'end']:
    image = Image.open(f"start_{image_name}.png")
    print("Input image:", image_name.title())
    print("Response:",
        model.answer_question(
            image,
            prompt,
            tokenizer,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.3,
        ),
    )
    print()
