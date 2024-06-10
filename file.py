import os
import bs4
from langchain import hub

from langchain_community.document_loaders import WebBaseLoader

from langchain.vectorstores import Chroma

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv



# Get API key

load_dotenv()


api_key = os.environ.get("API_KEY")


## Parameters

website = "https://johannesphd.de"
question = "What is the controler's basic concept?"
question = "Who is Johannes Köppern"


## Application

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=api_key)


loader = WebBaseLoader(

    web_paths=(website,),

    bs_kwargs={}

)


docs = loader.load()


# Take a look into the downloaded website

print(docs[0].page_content[:500])


# Split text in documents into chunks

# Neccessary since text can be too long for llm's context window

# Overlaping chunks mitigstes the danger of loosing context within chunks

text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=1500, chunk_overlap=200, add_start_index=True

)


splits = text_splitter.split_documents(docs)


print(f"Text was split into {len(splits)} chunks")


for this_split in splits:

    print(f"- {len(this_split.page_content)} characters")


# Actual indexing
# Prepare ChronmaDB as vector store
# Actual indexing
# Prepare ChronmaDB as vector store
create_db = False

def create_and_persist_vectorstore(splits, persist_directory="chroma_db"):
    """Creates a Chroma vectorstore from documents and persists it."""

    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(api_key=api_key), 
        persist_directory=persist_directory
    )
    return vectorstore

def load_vectorstore(persist_directory="chroma_db"):
    """Loads a Chroma vectorstore from a persisted directory."""

    # This line is the key to loading the persisted vectorstore
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=OpenAIEmbeddings(api_key=api_key)
    )
    return vectorstore

if create_db:
    vectorstore = create_and_persist_vectorstore(splits=splits, persist_directory="chroma_db")
else:
    vectorstore = load_vectorstore()



## Retrieve and generate using the relevant snippets of the blog.

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})


## Ask store for relevant chunks

retrieved_docs = retriever.invoke(question)


print(f"Found {len(retrieved_docs)} relevant chunks.")


## Entire retrieval chain

# chain that takes a question, retrieves relevant documents, constructs a prompt, passes that to a model, and parses the output


prompt = hub.pull("rlm/rag-prompt")

"""

You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.


Question: {question} 


Context: {context} 


Answer:

"""


example_messages = prompt.invoke(

    {"context": "Insert the found chunks here...", "question": question}

).to_messages()


print(example_messages)



# Merge all cxhunks to one single string

def format_docs(docs):

    return "\n\n".join(doc.page_content for doc in docs)



rag_chain = (

    {"context": retriever | format_docs, "question": RunnablePassthrough()}

    | prompt

    | llm

    | StrOutputParser()

)


print(rag_chain.invoke(question))
