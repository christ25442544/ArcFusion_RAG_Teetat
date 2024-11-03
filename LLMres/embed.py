import os
import logging
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from openai import AzureOpenAI


load_dotenv()

class EmbeddingsService:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("API_VERSION")
        )
        self.model_name = os.getenv("DEPLOYMENT_NAME", "embedding")

    def get_embeddings_sync(self, text: str):
        try:
            response = self.client.embeddings.create(
                input=text, 
                model=self.model_name
            )
            result = response.data[0].embedding
            logging.info(f"dimension: {len(result)}")
            return result
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise

class CustomAzureOpenAIEmbeddings(Embeddings):
    def __init__(self, embedding_service: EmbeddingsService):
        self.embedding_service = embedding_service

    def embed_query(self, text: str) -> list:
        return self.embedding_service.get_embeddings_sync(text)

    def embed_documents(self, texts: list) -> list:
        return [self.embedding_service.get_embeddings_sync(text) for text in texts]


embedding_service = EmbeddingsService()
custom_embeddings = CustomAzureOpenAIEmbeddings(embedding_service)