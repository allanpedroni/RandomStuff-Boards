from azure.ai.textanalytics import TextAnalyticsClient
from azure.identity import DefaultAzureCredential

language_endpoint = "https://foundryagentproject-resource.services.ai.azure.com/"

language_client = TextAnalyticsClient(
    endpoint=language_endpoint,
    credential=DefaultAzureCredential(),
)

documents = [
    "Azure AI Language can detect the language used in a sentence.",
    "Azure AI Language peut detecter la langue utilisee dans une phrase.",
]

results = language_client.detect_language(documents)

for document, result in zip(documents, results):
    print(f"Text: {document}")
    if result.is_error:
        print(f"Error: {result.error.code} - {result.error.message}\n")
    else:
        language = result.primary_language
        print(
            f"Detected language: {language.name} ({language.iso6391_name})\n"
            f"Confidence score: {language.confidence_score:.2f}\n"
        )
