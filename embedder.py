from custom_embedders import CustomChromaEmbedder
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

if __name__ == "__main__":
    # load the document 
    loader = TextLoader("./state_of_the_union.txt")
    documents = loader.load()

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # create the open-source embedding function
    embedding_function = CustomChromaEmbedder()

    # load it into Chroma
    db = Chroma.from_documents(documents=docs, embedding=embedding_function, persist_directory="./chromadb")
    print("embedding complete")  



# import os
# from typing import List
# import chromadb
# import requests
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# from langchain_community.vectorstores import Chroma

# from chromadb.api.types import (
#     Documents,
#     EmbeddingFunction,
#     Embeddings
# )

# class CustomEmbedder(EmbeddingFunction):
#     def __init__(self) -> None:
#         self.API_TOKEN = os.environ["API_TOKEN"]
#         #self.API_TOKEN = "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"
#         self.API_URL = "https://c9ejquzh6yum3xqf.us-east-1.aws.endpoints.huggingface.cloud/"

#     def __call__(self, input: Documents) -> Embeddings:
#         rest_client = requests.Session()
#         response = rest_client.post(
#             self.API_URL, json={"inputs": input}, headers={"Authorization": f"Bearer {self.API_TOKEN}"}
#         ).json()
#         print("Embedder response:", response)
#         return response
    
#     def embed_documents(self, texts) -> Embeddings:
#         return self(texts)
    
#     # added to resolve AttributeError: 'CustomEmbedder' object has no attribute 'embed_query'
#     def embed_query(self, text: str) -> List[float]:
#         #print(f"CustomEmbedder.embed_query called with query {text}")
#         return self.embed_documents([text])[0]

# def main():

#     # load the document 
#     loader = TextLoader("./state_of_the_union.txt")
#     documents = loader.load()

#     # split it into chunks
#     text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
#     docs = text_splitter.split_documents(documents)

#     # create the open-source embedding function
#     embedding_function = CustomEmbedder()

#     # load it into Chroma
#     db = Chroma.from_documents(documents=docs, embedding=embedding_function, persist_directory="./chroma.db")

#     print("Embedding complete")


# if __name__ == "__main__":
#     main()
    
