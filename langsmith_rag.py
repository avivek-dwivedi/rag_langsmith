import os
from apikeys import LangSmith_API  # Ensure apikeys.py contains LangSmith_API = "your_api_key_here"

# ✅ Set up LangSmith API environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LangSmith_API
os.environ["LANGCHAIN_PROJECT"] = "LangChain RAG Linux"

# ✅ Define Ollama model name & server details
ollama_model_name = "llama2"  # Ensure this matches the model name in Ollama
ollama_base_url = "http://127.0.0.1:11501"  # Connects to your Ollama serve instance

# ✅ Load the LLM with Ollama
from langchain_community.llms import Ollama
llm = Ollama(model=ollama_model_name, base_url=ollama_base_url)

# ✅ Test the LLM connection
response = llm.invoke("Why is the sky blue?")
print("LLM Response:", response)

# ✅ Load embeddings using Ollama
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model=ollama_model_name, base_url=ollama_base_url)

# ✅ Load documents from the 'Data' directory
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("Data", glob="**/*.docx")  # Change extension if needed
docs = loader.load()

# ✅ Split documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# ✅ Create FAISS vector database with embeddings
from langchain_community.vectorstores import FAISS
vector = FAISS.from_documents(documents, embedding=embeddings)

# ✅ Prepare the prompt template
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt_template = """You are an AI assistant providing insightful answers.
Use ONLY the provided context:

<context>
{context}
</context>

Question: {input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
document_chain = create_stuff_documents_chain(llm, prompt)

# ✅ Create the retriever and LangChain retrieval chain
from langchain.chains import create_retrieval_chain
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# ✅ Test Questions
TestQuestions = [
    "Summarize the story for me",
    "Who was the main protagonist?",
    "Did they have any children? If so, what were their names?",
    "Did anything eventful happen?",
    "Who are the main characters?",
    "What do you think happens next in the story?"
]

qa_pairs = []

for index, question in enumerate(TestQuestions, start=1):
    question = question.strip()
    print(f"\n{index}/{len(TestQuestions)}: {question}")
    
    response = retrieval_chain.invoke({"input": question})
    qa_pairs.append((question, response["answer"].strip()))  # Add to output array

# ✅ Print out the questions and answers
for index, (question, answer) in enumerate(qa_pairs, start=1):
    print(f"{index}/{len(qa_pairs)} {question}\n\n{answer}\n\n--------\n")
