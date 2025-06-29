# ARYA - AI Agent (RAG + OpenChat)

This is a terminal-based AI agent built with Python, Ollama, and OpenChat. It uses Retrieval-Augmented Generation (RAG) to improve answers by referencing a small knowledge base.

## How to Run

1. Install [Ollama](https://ollama.com)
2. Pull required models:
    ```bash
    ollama pull openchat
    ollama pull nomic-embed-text
    ```
3. Install Python packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the agent:
    ```bash
    python RAG_AGENT.py
    ```

> Developed by Sankeerth C V
