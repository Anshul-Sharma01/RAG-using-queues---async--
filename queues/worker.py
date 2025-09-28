from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)



embedding_model = HuggingFaceEmbeddings(
    model_name ="all-MiniLM-L6-v2"
)

vector_db = QdrantVectorStore.from_existing_collection(
    embedding = embedding_model,
    url = "http://localhost:6333",
    collection_name = "First-RAG-Pipeline"
)




def process_query(query : str):
    print("Searching chunks for query !!")
    search_results = vector_db.similarity_search(query=query)

    context = "\n\n\n".join([f"Page Content : {result.page_content}\nPage Number : {result.metadata['page_label']}\nFile Location : {result.metadata['source']}"
    for result in search_results])


    SYSTEM_PROMPT = f"""
        You are a helpful AI Assistant that answers user queries based solely on the provided context retrieved from a PDF file, including the page contents and their page numbers.

        You must answer only using the given context and, whenever applicable, guide the user to the specific page number(s) where they can find more detailed information.

        Context:
        {context}
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]

    response = llm.invoke(messages)
    print(f"The response is : {response.content}")
    return response.content;
