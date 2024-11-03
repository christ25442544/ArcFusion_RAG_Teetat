import json
from fastapi import FastAPI, Request
import os
import logging
from dotenv import load_dotenv
import requests
from LLM.llm_agent import LLMAgentService  

load_dotenv()

app = FastAPI()
llm_service = LLMAgentService()  # Instantiate the agent service

@app.post("/webhook")
async def receive_webhook(request: Request):
    payload = await request.json()
    logging.info(f"Received payload: {json.dumps(payload, indent=4, ensure_ascii=False)}")

    events = payload.get("events", [])
    for event in events:
        if event["type"] == "message" and event["message"]["type"] == "text":
            user_message = event["message"]["text"]
            reply_token = event["replyToken"]

            # Get the response from LLMAgentService
            user_id = event["source"]["userId"]
            bot_reply = await llm_service.get_response(user_id, user_message)

            # Send the response back to the user on LINE
            line_response = send_line_message(reply_token, bot_reply)
            
            logging.info(f"LINE API response: {line_response}")

    return {"status": "received", "message": "Webhook processed successfully"}

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
        logging.error(f"Error sending LINE message: {response.status_code} {response.text}")
    return response.json()