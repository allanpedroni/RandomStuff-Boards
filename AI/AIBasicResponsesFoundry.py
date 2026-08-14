from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ModelDeployment
from azure.identity import DefaultAzureCredential

project_endpoint = "https://foundryagentproject-resource.services.ai.azure.com/api/projects/foundryagentproject"
deployment_name = "gpt-5.6-terra"

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
print(response.output_text)
print(f"ID - {response.id}\n")

response2 = client.responses.create(
    model=deployment_name,
    previous_response_id=response.id,
    input="What about France?"
)
print(response2.output_text)

print("\nModel deployments:")
for deployment in project_client.deployments.list():
    if isinstance(deployment, ModelDeployment):
        print(
            f"- {deployment.name}: {deployment.model_publisher}/"
            f"{deployment.model_name} ({deployment.model_version})"
        )
