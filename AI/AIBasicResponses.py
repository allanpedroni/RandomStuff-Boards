from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = "https://foundryagentproject-resource.services.ai.azure.com/openai/v1"
deployment_name = "gpt-5.6-terra"
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")

client = OpenAI(
    base_url=endpoint,
    api_key=token_provider
)

response = client.responses.create(
    model=deployment_name,
    instructions="You are a helpful assistant that always answers like a Pirate and with humor.",
    input="What is the capital of England?"
)
print(response.output_text)
print(f"ID - {response.id}\n")

response2 = client.responses.create(
    model=deployment_name,
    previous_response_id=response.id,
    input="What about France?"
)
print(response2.output_text)