#from customllm import CustomLLM
from customllm import default_llm
#from custom_embedders import CustomChromaEmbedder
from custom_embedders import default_embedder
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

question = "What is the CHIPS Act?"
template = """Instructions: Use only the following context to answer the question.

Context: {context}
Question: {question}
"""

# embedder = CustomChromaEmbedder()
embedder = default_embedder
db = Chroma(persist_directory="./chromadb", embedding_function=embedder)
retriever = db.as_retriever(search_kwargs={"k": 1})
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# llm = CustomLLM()
llm = default_llm
# chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(chain.invoke(question))   


# import time
# from customllm import CustomLLM

# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain_community.vectorstores import Chroma
# from embedder import CustomEmbedder


# # load the document 
# loader = TextLoader("./state_of_the_union.txt")
# documents = loader.load()

# # split it into chunks
# text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# # create the open-source embedding function
# #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# embedding_function = CustomEmbedder()

# # load it into Chroma
# db = Chroma.from_documents(docs, embedding_function)

# # query it
# query = "What did the president say about Ketanji Brown Jackson"
# docs = db.similarity_search(query)

# # print results
# print("State union content : ",docs[0].page_content)

# time.sleep(20)

# # question = "Who won the FIFA World Cup in the year 1994?"

# # template = """Question: {question}

# # Answer: Let's think step by step."""

# # prompt = PromptTemplate(template=template, input_variables=["question"])

# # llm = CustomLLM()
# # llm_chain = LLMChain(prompt=prompt, llm=llm)
# # print(llm_chain.invoke(question))

# question = "Who won the FIFA World Cup in the year 1994? "
# context = "Portugal won the World Cup in 1994."

# template = """Instructions: Use only the following context to answer the question.

# Context: {context}
# Question: {question}

# Answer: The FIFA World Cup winner in 1994 is"""

# prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# llm = CustomLLM()
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# print(llm_chain.invoke({"context": context, "question": question}))