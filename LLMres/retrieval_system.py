import json
import os
import asyncio
from typing import List, Dict, Any, Optional
from langchain.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import AzureOpenAIConfig
from document_processor import DocumentProcessor
from vector_store import PineconeManager
from conversation import ConversationalAgent

class RetrievalSystem:
    
    def __init__(self, 
                 embedding_model,
                 pinecone_api_key: str,
                 pinecone_environment: str,
                 index_name: str):
        self.embedding_model = embedding_model
        self.doc_processor = DocumentProcessor()
        self.pinecone_manager = PineconeManager(
            api_key=pinecone_api_key,
            environment=pinecone_environment
        )
        self.index_name = index_name
        self.azure_config = AzureOpenAIConfig()
        self.memory = MemorySaver()
        self.agent = None
        self.conv_agent = None
        self.metadata_file = "embedding_metadata.json"
        self.processed_files = self._load_metadata()
        self.pc = Pinecone(api_key=pinecone_api_key)

    def _load_metadata(self) -> Dict[str, Dict]:
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.processed_files, f, indent=2)

    def _update_file_metadata(self, file_path: str):
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            self.processed_files[file_path] = {
                "last_modified": stat.st_mtime,
                "size": stat.st_size
            }
            self._save_metadata()

    def _check_file_changed(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            if file_path in self.processed_files:
                del self.processed_files[file_path]
                self._save_metadata()
                return False
            return False

        stat = os.stat(file_path)
        if file_path not in self.processed_files:
            return True

        metadata = self.processed_files[file_path]
        return (stat.st_mtime != metadata["last_modified"] or 
                stat.st_size != metadata["size"])

    def initialize_document_client(self, endpoint: str, key: str):
        self.doc_processor.initialize_document_client(endpoint, key)

    async def initialize_chat_system(self):
        try:
            if self.index_name not in self.pc.list_indexes().names():
                raise ValueError(f"Index '{self.index_name}' not found. Please run generate_embeddings.py first.")

            print("Found existing Pinecone index.")
            
            self.pinecone_manager.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embedding_model
            )
            
            retriever = self.pinecone_manager.get_retriever()
            
            llm = self.azure_config.create_llm()
            
            tool = create_retriever_tool(
                retriever,
                "document_retriever",
                "Searches the database for information about Chris (Teetat Karuhawanit). Always use this tool first before answering any question."
            )
            
            self.agent = create_react_agent(
                llm,
                [tool],
                checkpointer=self.memory,
            )
            
            self.conv_agent = ConversationalAgent(self.agent)
            
            print("Successfully initialized chat system with existing embeddings.")
            return self.conv_agent
            
        except Exception as e:
            print(f"Error initializing chat system: {str(e)}")
            raise

    async def setup_from_web(self, url: str):
        documents = self.doc_processor.load_web_content(url)
        await self._setup_system(documents)

    async def setup_from_files(self, file_paths: List[str]):
        all_documents = []
        new_files = False

        for file_path in file_paths:
            if self._check_file_changed(file_path):
                print(f"Processing new/modified file: {file_path}")
                documents = self.doc_processor.load_text_files([file_path])
                all_documents.extend(documents)
                self._update_file_metadata(file_path)
                new_files = True
            else:
                print(f"Skipping unchanged file: {file_path}")

        if new_files:
            await self._setup_system(all_documents, use_char_splitter=True)
        else:
            await self._initialize_existing_system()

    async def setup_from_pdf_file(self, file_path: str):
        if self._check_file_changed(file_path):
            print(f"Processing new/modified PDF: {file_path}")
            documents = await self.doc_processor.load_pdf_file(file_path)
            if documents:  
                self._update_file_metadata(file_path)
                await self._setup_system(documents)
        else:
            print(f"Skipping unchanged PDF: {file_path}")
            await self._initialize_existing_system()

    async def _initialize_existing_system(self):
        try:
            retriever = self.pinecone_manager.get_retriever()
            
            llm = self.azure_config.create_llm()
            
            tool = create_retriever_tool(
                retriever,
                "document_retriever",
                "Searches the database for information about Chris (Teetat Karuhawanit). Always use this tool first before answering any question."
            )
            
            self.agent = create_react_agent(
                llm,
                [tool],
                checkpointer=self.memory,
            )
            
            self.conv_agent = ConversationalAgent(self.agent)
            
            return self.conv_agent
        except Exception as e:
            print(f"Error initializing existing system: {str(e)}")
            raise

    async def _setup_system(self, documents: List[Document], use_char_splitter: bool = False):
    
        if not documents:
            print("No new documents to process. Using existing system...")
            await self._initialize_existing_system()
            return

        try:
            splits = self.doc_processor.split_documents(documents, use_char_splitter)
            
            if not self.index_name in self.pc.list_indexes().names():
                print(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                await asyncio.sleep(10)
            
            await self.pinecone_manager.create_vectorstore(
                documents=splits,
                embedding_model=self.embedding_model,
                index_name=self.index_name
            )
            
            await self._initialize_existing_system()
        except Exception as e:
            print(f"Error in system setup: {str(e)}")
            raise

    async def chat(self, thread_id: str, message: str):
        if not self.conv_agent:
            raise ValueError("System not initialized. Call setup first.")
        print(f"\nYou: {message}")
        response = await self.conv_agent.stream_response(thread_id, message)
        return response

    def get_conversation_history(self, thread_id: str) -> List[Dict[str, Any]]:
        if not self.conv_agent:
            raise ValueError("System not initialized. Call setup first.")
        return self.conv_agent.get_conversation(thread_id).messages

    def similarity_search(self, query: str, k: int = 4):
        return self.pinecone_manager.similarity_search(query, k=k)

    def cleanup_removed_files(self):
        removed_files = []
        for file_path in list(self.processed_files.keys()):
            if not os.path.exists(file_path):
                print(f"File no longer exists: {file_path}")
                removed_files.append(file_path)
                del self.processed_files[file_path]
        
        if removed_files:
            print(f"Removed metadata for deleted files: {removed_files}")
            self._save_metadata()