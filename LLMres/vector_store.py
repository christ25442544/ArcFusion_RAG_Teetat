import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

class PineconeManager:
    
    def __init__(self, api_key: str, environment: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = None
        self.vector_store = None
        self.pc = Pinecone(api_key=api_key)

    async def create_vectorstore(self, 
                               documents: List[Document], 
                               embedding_model, 
                               index_name: str,
                               dimension: int = 1536) -> PineconeVectorStore:
        self.index_name = index_name

        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region='us-east-1'
                )
            )

        self.vector_store = await asyncio.to_thread(
            PineconeVectorStore.from_documents,
            documents=documents,
            embedding=embedding_model,
            index_name=index_name
        )
        
        return self.vector_store

    def get_retriever(self):
        
        if not self.vector_store:
            raise ValueError("Vector store has not been initialized")
        return self.vector_store.as_retriever()

    def similarity_search(self, query: str, k: int = 4):
        if not self.vector_store:
            raise ValueError("Vector store has not been initialized")
        return self.vector_store.similarity_search(query, k=k)

    def delete_index(self):
        if self.index_name and self.index_name in self.pc.list_indexes().names():
            self.pc.delete_index(self.index_name)