import base64
from pathlib import Path

from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = "https://foundryagentproject-resource.services.ai.azure.com/openai/v1"
deployment_name = "gpt-5.6-terra"
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")
image_path = Path(__file__).resolve().parents[1] / "Misc" / "500KNormal.png"
image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

client = OpenAI(
    base_url=endpoint,
    api_key=token_provider
)

response = client.responses.create(
    model=deployment_name,
    instructions="You are a helpful assistant that always answers like a sarcastic but still helpful AI.",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in this picture?"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_data}",
                },
            ],
        }
    ],
)
print(response.output_text)
print(f"ID - {response.id}\n")