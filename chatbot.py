import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

def set_custom_prompt():
    prompt_template = """
        Use the following pieces of context to answer the question at the end in one sentence.
        If you don't know the answer, don't try to make up an answer.

        {context}

        Question: {question}
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt

def load_llm():
    llm = Ollama(
        model="llama2", # tried out mistral-7b, llama2, tinyllama
        verbose=True,
        temperature=0.2,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm

# Set up the QA chain
def retrieval_qa_chain(llm, prompt, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain

# Create the retrieval QA bot
def create_retrieval_qa_bot():
    db_path = os.getenv('DB_PATH')
    vectorstore = Chroma(persist_directory=db_path, embedding_function=HuggingFaceEmbeddings())
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, vectorstore)
    return qa

# Get input payload from UI and generate response text
def get_response_from_bot(question, context=""):
    qa_bot = create_retrieval_qa_bot()
    query = f"Question: {question}"
    response = qa_bot.invoke({"query": query})
    return response['result']

# Store context in the DB for further conversation
def store_context(context):
    db_path = os.getenv('DB_PATH')
    vectorstore = Chroma(persist_directory=db_path, embedding_function=HuggingFaceEmbeddings())
    vectorstore.add_documents([Document(page_content=context)])
    vectorstore.persist()

# Push output payload (response text) to CMD --- simulates sending a response to the UI by printing it to CMD
def push_output_to_cmd(response_text):
    print("Bot:", response_text)

if __name__ == "__main__":
    conversation_context = ""
    
    while True:
        question = input("You: ")
        
        if question.lower() in ["exit", "quit"]:
            print("Ending conversation.")
            break
        
        response = get_response_from_bot(question, conversation_context)
        conversation_context += f"Question: {question}\nAnswer: {response}\n"
        push_output_to_cmd(response)
        store_context(conversation_context)
