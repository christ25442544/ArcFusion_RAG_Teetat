import os
import asyncio
from dotenv import load_dotenv
from embed import EmbeddingsService, CustomAzureOpenAIEmbeddings
from retrieval_system import RetrievalSystem

async def process_docs_directory(system: RetrievalSystem, docs_dir: str = "docs"):
    """Process all documents in the docs directory and generate embeddings."""
    # Create docs directory if it doesn't exist
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"Created {docs_dir} directory")
        return

    # Get all files in docs directory
    files = os.listdir(docs_dir)
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    text_files = [f for f in files if f.lower().endswith(('.txt', '.md'))]
    
    if not (pdf_files or text_files):
        print(f"No PDF or text files found in {docs_dir} directory")
        return

    print("\nChecking for files that need updating...")
    
    # Process PDF files
    for pdf_file in pdf_files:
        file_path = os.path.join(docs_dir, pdf_file)
        if system._check_file_changed(file_path):
            try:
                print(f"\nGenerating embeddings for PDF file: {pdf_file}")
                await system.setup_from_pdf_file(file_path)
            except Exception as e:
                print(f"Error processing PDF file {pdf_file}: {str(e)}")
        else:
            print(f"Skipping unchanged PDF: {pdf_file}")

    # Process text files
    if text_files:
        changed_text_files = []
        for text_file in text_files:
            file_path = os.path.join(docs_dir, text_file)
            if system._check_file_changed(file_path):
                changed_text_files.append(file_path)
                print(f"Will process modified text file: {text_file}")
                
        if changed_text_files:
            try:
                print("\nGenerating embeddings for modified text files...")
                await system.setup_from_files(changed_text_files)
            except Exception as e:
                print(f"Error processing text files: {str(e)}")
        else:
            print("No text files need updating")

async def main():
    """Main function to generate embeddings."""
    # Load environment variables
    load_dotenv()
    
    # Set USER_AGENT
    os.environ['USER_AGENT'] = 'ChrisBot/1.0'
    
    print("\nInitializing embedding system...")
    
    # Initialize embedding service
    embedding_service = EmbeddingsService()
    custom_embeddings = CustomAzureOpenAIEmbeddings(embedding_service)
    
    # Initialize the system
    system = RetrievalSystem(
        embedding_model=custom_embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
        index_name="chris-data"
    )
    
    # Initialize document client
    system.initialize_document_client(
        endpoint=os.getenv("AZURE_DOCUMENT_ENDPOINT"),
        key=os.getenv("AZURE_DOCUMENT_KEY")
    )
    
    try:
        # Check for removed files and clean up if necessary
        system.cleanup_removed_files()
        
        # Process all documents in the docs directory
        await process_docs_directory(system)
        
        print("\nEmbedding generation complete!")
        
    except Exception as e:
        print(f"An error occurred during embedding generation: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"Fatal error: {str(e)}")