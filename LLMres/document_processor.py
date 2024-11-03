from typing import List
import bs4
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

class DocumentProcessor:
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.char_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separator="\n"
        )
        self.document_client = None

    def initialize_document_client(self, endpoint: str, key: str):
        self.document_client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )

    def load_web_content(self, url: str) -> List[Document]:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        return loader.load()

    def load_text_files(self, file_paths: List[str]) -> List[Document]:
        documents = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    if text.strip():
                        documents.append(Document(page_content=text))
                        print(f"Successfully loaded {file_path}: {len(text)} characters")
                    else:
                        print(f"Warning: {file_path} is empty")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        return documents

    async def load_pdf_url(self, url: str) -> List[Document]:
        if not self.document_client:
            raise ValueError("Document client not initialized. Call initialize_document_client first.")
        
        try:
            poller = self.document_client.begin_analyze_document_from_url(
                "prebuilt-read", url
            )
            result = poller.result()
            
            documents = []
            for page in result.pages:
                page_text = "\n".join([line.content for line in page.lines])
                metadata = {
                    "source": url,
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "unit": page.unit
                }
                documents.append(Document(
                    page_content=page_text,
                    metadata=metadata
                ))
            
            print(f"Successfully loaded PDF from URL: {url}")
            return documents
            
        except Exception as e:
            print(f"Error loading PDF from URL {url}: {str(e)}")
            return []

    async def load_pdf_file(self, file_path: str) -> List[Document]:
        if not self.document_client:
            raise ValueError("Document client not initialized. Call initialize_document_client first.")
        
        try:
            with open(file_path, "rb") as file:
                poller = self.document_client.begin_analyze_document(
                    "prebuilt-read", file
                )
                result = poller.result()

            documents = []
            for page in result.pages:
                page_text = "\n".join([line.content for line in page.lines])
                metadata = {
                    "source": file_path,
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "unit": page.unit
                }
                documents.append(Document(
                    page_content=page_text,
                    metadata=metadata
                ))
            
            print(f"Successfully loaded PDF file: {file_path}")
            return documents
            
        except Exception as e:
            print(f"Error loading PDF file {file_path}: {str(e)}")
            return []

    def split_documents(self, documents: List[Document], use_char_splitter: bool = False) -> List[Document]:
        if not documents:
            print("Warning: No documents to split")
            return []
            
        splitter = self.char_splitter if use_char_splitter else self.text_splitter
        splits = splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(splits)} chunks")
        return splits