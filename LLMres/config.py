import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

class AzureOpenAIConfig:
    
    
    def __init__(self):
        load_dotenv()
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("MODEL_NAME")
        self.api_version = "2024-05-01-preview"

    def create_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_endpoint=self.endpoint,
            azure_deployment=self.deployment,
            openai_api_version=self.api_version,
        )