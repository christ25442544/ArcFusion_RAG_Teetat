from dataclasses import dataclass
from typing import List, Dict, Any
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from prompt import PromptBot

@dataclass
class Conversation:
    thread_id: str
    messages: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        self.messages = []
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def get_context(self, last_n: int = 3) -> List[Dict[str, Any]]:
        return self.messages[-last_n:] if len(self.messages) > 0 else []
    
    def clear_history(self):
        self.messages = []

class ConversationalAgent:
    
    def __init__(self, agent):
        self.agent = agent
        self.conversations: Dict[str, Conversation] = {}
        self.system_prompt = PromptBot
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("MODEL_NAME"),
            openai_api_version="2024-05-01-preview",
        )
    
    def create_conversation(self, thread_id: str) -> Conversation:
        conversation = Conversation(thread_id=thread_id)
        self.conversations[thread_id] = conversation
        return conversation
    
    def get_conversation(self, thread_id: str) -> Conversation:
        if thread_id not in self.conversations:
            return self.create_conversation(thread_id)
        return self.conversations[thread_id]
    
    def clear_conversation(self, thread_id: str) -> bool:
        if thread_id in self.conversations:
            del self.conversations[thread_id]
            self.create_conversation(thread_id)
            return True
        return False
    
    def prepare_messages(self, conversation: Conversation, query: str) -> List[dict]:
        messages = [
            SystemMessage(content=self.system_prompt),
        ]
        
        for msg in conversation.get_context():
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=query))
        return messages
    
    async def detect_clear_intent(self, message: str) -> bool:
        system_message = """You are a message intent classifier. 
        Your task is to determine if a message expresses an intent to clear chat history, memory, or start a new conversation.
        Respond with just 'true' if the intent is to clear memory/chat, or 'false' otherwise.
        This should work across all languages."""
        
        human_message = f"""Determine if this message expresses an intent to clear chat history or memory: "{message}"
        Remember to respond with just 'true' or 'false'."""
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content.strip().lower() == 'true'

    async def stream_response(self, thread_id: str, query: str):
        if await self.detect_clear_intent(query):
            if self.clear_conversation(thread_id):
                print("Chat history cleared successfully")
                return "Memory cleared. Starting a new conversation."
            return "No conversation history found to clear."
        
        if thread_id not in self.conversations:
            self.create_conversation(thread_id)
        conversation = self.conversations[thread_id]
        
        messages = self.prepare_messages(conversation, query)
        config = {"configurable": {"thread_id": thread_id}}
        
        response_content = []
        for event in self.agent.stream(
            {"messages": messages},
            config=config,
            stream_mode="values"
        ):
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                print(f"Assistant: {last_message.content}")
                response_content.append(last_message.content)
        
        full_response = " ".join(response_content)
        conversation.add_message("human", query)
        conversation.add_message("assistant", full_response)
        
        return full_response