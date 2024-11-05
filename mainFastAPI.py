# mainFastAPI.py
import json
from fastapi import FastAPI, Request
import os
import logging
import asyncio
from dotenv import load_dotenv
import requests
import sys
from pathlib import Path

# Add LLMres directory to Python path
LLMRES_PATH = Path(__file__).parent / "LLMres"
sys.path.append(str(LLMRES_PATH))

# Import LLMres modules
from LLMres.embed import EmbeddingsService, CustomAzureOpenAIEmbeddings
from LLMres.retrieval_system import RetrievalSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set USER_AGENT
os.environ['USER_AGENT'] = 'ChrisBot/1.0'

app = FastAPI(title="Chatbot API")

class ChatbotService:
    def __init__(self):
        self.retrieval_system = None
        self.initialized = False
    
    async def initialize(self):
        if not self.initialized:
            try:
                # Initialize embedding service
                embedding_service = EmbeddingsService()
                custom_embeddings = CustomAzureOpenAIEmbeddings(embedding_service)
                
                # Initialize the retrieval system
                self.retrieval_system = RetrievalSystem(
                    embedding_model=custom_embeddings,
                    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                    pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
                    index_name="chris-data"
                )
                
                # Initialize the chat system
                await self.retrieval_system.initialize_chat_system()
                self.initialized = True
                logger.info("Chatbot system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize chatbot system: {str(e)}")
                raise

    def extract_assistant_response(self, full_response: str) -> str:
        """Extract only the final assistant response from the full conversation."""
        try:
            # First, try to split by 'Assistant:' and get the last part
            parts = full_response.split('Assistant:')
            if len(parts) > 1:
                final_response = parts[-1].strip()
                
                # Remove any 'You:' or user message that might be included
                if 'You:' in final_response:
                    final_response = final_response.split('You:')[0].strip()
                
                # Remove any repeated text at the start
                if final_response.startswith(final_response[:20]) and len(final_response) > 40:
                    unique_part = final_response[len(final_response)//2:].strip()
                    return unique_part
                
                return final_response
            
            # If no 'Assistant:' found, return the cleaned original response
            return full_response.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning response: {str(e)}")
            return full_response.strip()

    async def get_response(self, user_id: str, message: str) -> str:
        if not self.initialized:
            await self.initialize()
        
        try:
            # Handle search queries
            if message.lower().startswith('search:'):
                search_query = message[7:].strip()
                results = self.retrieval_system.similarity_search(search_query, k=2)
                if results:
                    response_parts = []
                    for i, doc in enumerate(results, 1):
                        response_parts.append(f"Result {i}:\n{doc.page_content[:800]}...")
                    return "\n\n".join(response_parts)
                return "No results found"
            
            # Handle regular chat messages
            full_response = await self.retrieval_system.chat(user_id, message)
            
            # Extract only the final assistant response
            clean_response = self.extract_assistant_response(full_response)
            
            # Add logging to debug response cleaning
            logger.debug(f"Original response: {full_response}")
            logger.debug(f"Cleaned response: {clean_response}")
            
            return clean_response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return f"I apologize, but I encountered an error processing your message. Please try again."

# Initialize the chatbot service
chatbot_service = ChatbotService()

@app.on_event("startup")
async def startup_event():
    try:
        await chatbot_service.initialize()
        logger.info("Chatbot service initialized successfully on startup")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot service on startup: {str(e)}")
        # You might want to exit the application here if initialization is critical
        # sys.exit(1)

@app.post("/webhook")
async def receive_webhook(request: Request):
    try:
        payload = await request.json()
        logger.info(f"Received payload: {json.dumps(payload, indent=4, ensure_ascii=False)}")
        
        events = payload.get("events", [])
        for event in events:
            if event["type"] == "message" and event["message"]["type"] == "text":
                user_message = event["message"]["text"]
                reply_token = event["replyToken"]
                user_id = event["source"]["userId"]
                
                # Get response from chatbot
                bot_reply = await chatbot_service.get_response(user_id, user_message)
                
                # Send response back to LINE
                line_response = send_line_message(reply_token, bot_reply)
                logger.info(f"LINE API response: {line_response}")
        
        return {"status": "received", "message": "Webhook processed successfully"}
    
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return {"status": "error", "message": str(e)}

def send_line_message(reply_token: str, message: str):
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('LINE_CHANNEL_ACCESS_TOKEN')}"
    }
    data = {
        "replyToken": reply_token,
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        logger.error(f"Error sending LINE message: {response.status_code} {response.text}")
    return response.json()

# Add health check endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "initialized": chatbot_service.initialized,
        "llmres_path": str(LLMRES_PATH)
    }

@app.get("/")
async def root():
    return {
        "message": "Chatbot API is running",
        "status": "active",
        "initialized": chatbot_service.initialized
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)