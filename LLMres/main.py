import os
import asyncio
import uuid
from dotenv import load_dotenv
from embed import EmbeddingsService, CustomAzureOpenAIEmbeddings
from retrieval_system import RetrievalSystem

async def run_chat_session(system: RetrievalSystem, thread_id: str):
    """Run an interactive chat session."""
    print("\n=== Interactive Chat Session Started ===")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'search: your query' to perform similarity search")
    print("Type your questions for normal chat mode")
    print("==========================================\n")
    
    while True:
        try:
            user_input = input().strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nEnding chat session...")
                break
                
            elif user_input.lower().startswith('search:'):
                search_query = user_input[7:].strip()
                print("\nPerforming similarity search...")
                try:
                    results = system.similarity_search(search_query, k=2)
                    if results:
                        for i, doc in enumerate(results, 1):
                            print(f"\nResult {i}:")
                            print(f"{doc.page_content[:800]}...")
                    else:
                        print("No results found")
                except Exception as e:
                    print(f"Error during search: {str(e)}")
                    continue
            
            else:
                if not user_input:
                    continue
                    
                try:
                    await system.chat(thread_id, user_input)
                except Exception as e:
                    print(f"Error in chat processing: {str(e)}")
                    print("Please try again with a different question.")
                    continue
        
        except KeyboardInterrupt:
            print("\nDetected keyboard interrupt. Ending session...")
            break
            
        except EOFError:
            print("\nDetected EOF. Ending session...")
            break
            
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            print("The system will continue running. You can keep chatting.")
            continue

async def main():
    """Main function to initialize and run the chat system."""
    # Load environment variables
    load_dotenv()
    
    # Set USER_AGENT
    os.environ['USER_AGENT'] = 'ChrisBot/1.0'
    
    print("\nInitializing chat system...")
    
    try:
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
        
        # Initialize the chat system with existing embeddings
        await system.initialize_chat_system()
        
        print("Chat system ready!")
        
        # Start chat session
        thread_id = str(uuid.uuid4())
        await run_chat_session(system, thread_id)
        
    except ValueError as e:
        print(f"\nError: {str(e)}")
        print("Please make sure to run generate_embeddings.py first to create the embeddings.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        print("\nThank you for using the chat system!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"Fatal error: {str(e)}")