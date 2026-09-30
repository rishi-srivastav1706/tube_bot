import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")



def get_transcript(video_id):

    api = YouTubeTranscriptApi()

    transcript_list = api.fetch(
        video_id,
        languages=["en"]
    )

    transcript = " ".join(
        snippet.text
        for snippet in transcript_list
    )

    return transcript

def create_vector_store(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def create_llm():
    llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.2,
    )
    return ChatHuggingFace(llm=llm)

def ask_question(vector_store, question):
    print("QUESTION RECEIVED:", question)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    retrieved_docs    = retriever.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables =['context','question']
    
    )
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    llm = create_llm()
    answer = llm.invoke(final_prompt)

    return answer.content