## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

## PROBLEM STATEMENT:
Design an LCEL pipeline using LangChain with at least two dynamic prompt parameters. Integrate prompt, model, and output parser components to form a complete expression. Evaluate its functionality through real-world query-response scenarios.

## DESIGN STEPS:

**STEP 1:** Import the required LangChain libraries, OpenAI modules, embedding models, prompt templates, output parsers, and environment configuration packages.

**STEP 2:** Configure the OpenAI API key and initialize the ChatOpenAI model for generating responses from user inputs.

**STEP 3:** Create simple and complex chains by connecting prompt templates, LLM models, output parsers, and multiple processing steps using LangChain Expression Language (LCEL).

**STEP 4:** Create a vector store using document embeddings and store the given information using DocArrayInMemorySearch for efficient retrieval.

**STEP 5:** Configure a retriever and integrate it with prompt templates and the LLM to create a Retrieval-Augmented Generation (RAG) chain.

**STEP 6:** Execute the chains with different inputs, retrieve relevant information, process the queries using the LLM, and display the generated responses.
### PROGRAM:
````
import os
import openai
import sys
sys.path.append('../..')


from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("harrypotter.pdf")

pages = loader.load()

len(pages)
page = pages[0]
print(page.page_content[0:500])
page.metadata


import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)
llm.predict("Hello world!")

from langchain.prompts import PromptTemplate
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
from langchain.chains import RetrievalQA
question = "Is my resume contain my contant information "
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


result = qa_chain({"query": question})
result["result"]

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

question = " who is harrypotter book author"

result = qa({"question": question})

result['answer']

question = "how many parts has released in harrypotter series"
result = qa({"question": question})
result['answer']
``````````
### OUTPUT:
<img width="673" height="346" alt="image" src="https://github.com/user-attachments/assets/539e159b-3e0e-41ad-abc4-f66987eb8ddd" />
<img width="899" height="144" alt="image" src="https://github.com/user-attachments/assets/b36ef4f5-1303-4ae7-8fd2-2cf6c4f33c72" />
<img width="870" height="84" alt="image" src="https://github.com/user-attachments/assets/d5c08a76-7554-44c4-8467-f5fbafd03ade" />
<img width="890" height="346" alt="image" src="https://github.com/user-attachments/assets/9dbd285f-1527-403f-b4a4-69459d18e0a2" />


### RESULT:
The implemented LCEL expression takes at least two prompt parameters, processes them using a model, and formats the output with a parser, demonstrating its effectiveness through real-world examples.
