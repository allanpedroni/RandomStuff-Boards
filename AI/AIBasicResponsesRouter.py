from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ModelDeployment
from azure.identity import DefaultAzureCredential

project_endpoint = "https://foundryagentproject-resource.services.ai.azure.com/api/projects/foundryagentproject"
deployment_name = "model-router"

project_client = AIProjectClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(),
)
client = project_client.get_openai_client()

response = client.responses.create(
    model=deployment_name,
    instructions="You are a helpful assistant that always answers like a Pirate and with humor.",
    input="What is the capital of England?"
)
print(f"Requested deployment: {deployment_name}")
print(f"Selected model: {response.model}\n")
print(response.output_text)
