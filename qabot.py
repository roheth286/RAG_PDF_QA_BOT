import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_core.prompts import PromptTemplate
# pyrefly: ignore [missing-import]
from langchain_classic.chains import ConversationalRetrievalChain

load_dotenv()

def get_llm():

    model_id = "gemini-2.5-flash"

    gemini_llm = ChatGoogleGenerativeAI(
        model=model_id,
        temperature=0.5,
        max_tokens=256,
    )

    return gemini_llm


def document_loader(file):

    pdf_loader = PyPDFLoader(file)

    pdf_document = pdf_loader.load()

    return pdf_document

def text_splitter(data):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )

    chunks = splitter.split_documents(data)

    return chunks

def huggingface_embedding():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model


def vector_database(chunks):

    embedding_model = huggingface_embedding()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vectordb


prompt_template = """
You are a document-based question answering assistant.

Answer the question using ONLY the information contained in the document context below.

Rules:
1. Do not use any external knowledge.
2. Do not make assumptions or infer missing information.
3. If the answer is not explicitly present in the document, respond exactly with:
   I DON'T KNOW
4. If the question is unrelated to the document, respond exactly with:
   My knowledge is limited to the document provided.
5. Keep your answer concise and factual.

Document Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)



history = []

retriever_object = None

qa_chain = None

def initialize_rag(file):

    global retriever_object
    global qa_chain

    print("Loading PDF...")

    pdf_document = document_loader(file)

    print("Splitting document...")

    chunks = text_splitter(pdf_document)

    print(f"Created {len(chunks)} chunks")

    print("Creating vector database...")

    vector_db = vector_database(chunks)

    retriever_object = vector_db.as_retriever()

    print("Creating QA chain...")

    llm = get_llm()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_object,
        combine_docs_chain_kwargs={
            "prompt": prompt
        },
        return_source_documents=False,
        verbose=True
    )

    print("RAG initialization complete!")



def ask_question(query):

    global history
    global qa_chain

    if qa_chain is None:
        raise ValueError(
            "RAG pipeline not initialized. Call initialize_rag() first."
        )

    result = qa_chain.invoke(
        {
            "question": query,
            "chat_history": history
        }
    )

    history.append(
        (
            query,
            result["answer"]
        )
    )

    return result["answer"]

