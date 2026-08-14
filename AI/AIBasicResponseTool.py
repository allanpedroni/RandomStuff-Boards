from openai import OpenAI
from openai.types.responses import ResponseCodeInterpreterToolCall
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
    instructions="You are a helpful assistant that always answers like a Pirate and with humor."
            + "Use the python tool to run code for any math problems.",
    input="What is the square root of 1764?",
    tools=[{"type": "code_interpreter", "container": {"type" : "auto"}}]
)

for item in response.output:
    if isinstance(item, ResponseCodeInterpreterToolCall):
        print("Tool called: Code Interpreter")
        print(f"Status: {item.status}")
        print(f"Code:\n{item.code}")
        for output in item.outputs or []:
            if output.type == "logs":
                print(f"Output:\n{output.logs}")

print("\nAssistant response:")
print(response.output_text)