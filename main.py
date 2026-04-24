import os
from glob import glob
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv() 

# ---------------------------
# 1. CONFIG & PERFORMANCE FIXES
# ---------------------------
persist_directory = "./db/chroma_db"
pdf_dir = "./pdf"
os.makedirs(pdf_dir, exist_ok=True)

# ---------------------------
# 2. LOAD & 3. SPLIT (Optimized for better retrieval)
# ---------------------------
pdf_files = glob(os.path.join(pdf_dir, "*.pdf"))
if not pdf_files:
    print("❌ No PDFs found.")
    exit()

print(f"📄 Processing {len(pdf_files)} PDFs...")
all_docs = []
for file_path in pdf_files:
    loader = PyPDFLoader(file_path)
    all_docs.extend(loader.load())

# FINE-TUNE: Smaller chunks often lead to better retrieval for specific names
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, 
    chunk_overlap=100
)
docs = splitter.split_documents(all_docs)

# ---------------------------
# 4. EMBEDDINGS & 5. VECTOR DB
# ---------------------------
# Using a slightly better model for local embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FINE-TUNE: Resetting the DB on each run for testing (Optional)
# If you want to keep the DB, remove the next two lines
if os.path.exists(persist_directory):
    import shutil
    shutil.rmtree(persist_directory)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

# FINE-TUNE: Increased 'k' to 5 to give the LLM more context
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ---------------------------
# 7. GROQ LLM (Security & Model Fix)
# ---------------------------
API = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 


llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.1, # Slight temperature for better natural language
    api_key=SecretStr(API) 
)

# ---------------------------
# 8. PROMPT (Fine-tuned for strictness)
# ---------------------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the user's question.
If the answer is not contained in the context, strictly say "Not found in documents".

Context:
{context}

Question:
{question}

Answer:""")

# ---------------------------
# 9. RAG PIPELINE
# ---------------------------
qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}

    | prompt 
    | llm 
    | StrOutputParser()
)

if __name__ == "__main__":
    print()
    print('loading documents.................')
    print(f"\n✅ Indexing complete. {len(docs)} chunks created.")
    print("🚀 GROQ RAG SYSTEM READY\n")
    
    while True:
        query = input("Ask ➜ ").strip()
        if query.lower() in ["exit", "quit"]: break
        if not query: continue

        try:
            # Check if the retriever actually gets anything before sending to LLM
            # (Debugging step)
            test_docs = retriever.invoke(query)
            if not test_docs:
                print("⚠️  Retriever found 0 matching chunks in your PDFs.")
            
            response = qa_chain.invoke(query)
            print(f"\n🧠 Answer:\n{response}\n")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
