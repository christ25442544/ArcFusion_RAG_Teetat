# ChrisBot - Personalized Document QA System

A conversational AI system that allows users to chat with their documents using Azure OpenAI and Pinecone for vector storage.

## How It Works

The system works in two main modes:

1. **Embedding Generation Mode**
   - Processes documents from the `docs/` directory
   - Generates embeddings using Azure OpenAI
   - Stores embeddings in Pinecone vector database
   - Tracks document changes and updates
   - Only processes new or modified files

2. **Chat Mode**
   - Loads existing embeddings from Pinecone
   - Uses RAG (Retrieval Augmented Generation) to answer questions
   - Maintains conversation history
   - Provides similarity search functionality

### System Architecture

```mermaid
graph TB
    subgraph Document Processing
        A[Documents in docs/] --> B[Azure Document Intelligence]
        B --> C[Text Chunks]
        C --> D[Azure OpenAI Embeddings]
        D --> E[Pinecone Vector Store]
    end

    subgraph Chat System
        F[User Query] --> G[LangChain Agent]
        G --> H[Pinecone Retriever]
        H --> I[Azure OpenAI LLM]
        I --> J[Response]
    end
```

## Running Locally with Docker Compose

### Prerequisites

1. Create `.env` file with the variables needed:

2. Place your documents in the `docs/` directory (In PDF, txt, or even Website for scraping)

### Running the System

1. Build and start the containers:
```bash
docker-compose up --build
```

2. Generate embeddings (first time or after document changes):
```bash
docker-compose run app python generate_embeddings.py
```

3. Start chatting:
```bash
docker-compose run app python main.py
```

### Chat Commands

- Regular questions: Type your question and press Enter
- Test similarity search: Type `search: your query`
- Exit: Type `quit` or `exit`

## Future Improvements

1. **System Enhancements**
   - Add support for more document types (e.g., DOCX, PPT, md, etc)
   - Add support for multi-language documents (But I'm pretty sure document intelligence can support multiple languages).
   - Implement streaming responses for Beautify. (Look at Langchain documentation).

2. **Performance Optimizations**
   - Implement caching for frequently accessed embeddings.
   - Optimize document chunking size for better context retention.

3. **Features**
   - I’m working on integrating with LINE chat using mainFastAPI.py. Right now, the code in that file isn’t fully refined; it’s mostly a carryover from my previous chatbot project. The idea is to receive messages from LINE via the Messaging API, then pass them to a backend GPT-based Langchain for response generation. Once the response is ready, it will be sent back to LINE using the Messaging API.

    - In addition, I plan to enhance the UX/UI by leveraging LINE’s flex messages. This will allow me to implement interactive carousel messages that showcase my CV and LinkedIn profile, providing a more engaging experience.
   - Add support for document history and versioning
   - Add document metadata extraction and filtering

4. **Infrastructure**
   - Add monitoring and logging system
   - Implement automatic scaling (Kubernetes, or even EC2 with load balancing technique) when deploying on FastAPI application framework.
   - Implement CI/CD pipeline


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.