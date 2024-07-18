# CLI Ollama RAG Intructions to Run: 

Make sure you have Visual Build Tools and gcc installed beforehand.

1) Install Python (3.10.11 or below).
2) Install ChromaDB - 
   for python client: pip install chromadb
   // for javascript: npm install chromadb
   // for client-server mode: chroma run --path /chroma_db_path
3) Install and set up Ollama from this link: https://ollama.com/download
4) Install llama2 by running this command: ollama run llama2 and then check http://localhost:11434/ for whether Ollama is running.
5) CD into the local directory and run: pip install -r requirements.txt for dependencies.
6) Edit the DB_PATH and DATA_PATH variables in the .env file accordingly.
7) Run preprocess.py and wait until vectorstore is persisted. This might take 4-10 minutes depending on your processor.
8) Run chatbot.py and ask questions. Type 'exit' to end conversation.

To avoid ChromaDB error ---> delete /vectorstore/ contents and run preprocess.py to vectorize your data.
