import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceTextGenInference
from langchain.chains import APIChain
from langchain.llms import OpenAI

load_dotenv()
# llm = HuggingFaceTextGenInference(
#     inference_server_url=os.getenv("LLM_API"),
#     max_new_tokens=10,
#     top_p=0.9,
#     server_kwargs={
#         "headers": {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {os.getenv('API_TOKEN')}"
#         }
#     }
# )



llm = HuggingFaceTextGenInference(
    #inference_server_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    inference_server_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=10,
    top_p=0.9,
    server_kwargs={
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('API_TOKEN')}"
        }
    }
)

# #llm = OpenAI(temperature=0)
# api_docs = """

# BASE URL: https://restcountries.com/

# API Documentation:

# The API endpoint /v3.1/name/{name} Used to find informatin about a country. All URL parameters are listed below:
#     - name: Name of country - Ex: italy, france
    
# The API endpoint /v3.1/currency/{currency} Uesd to find information about a region. All URL parameters are listed below:
#     - currency: 3 letter currency. Example: USD, COP
    
# Woo! This is my documentation
# """

# chain_new = APIChain.from_llm_and_api_docs(llm, api_docs, verbose=True, limit_to_domains=["https://restcountries.com/"])


# chain_new.run('Can you tell me information about france?')


from langchain.chains import APIChain
from langchain.chains.api import open_meteo_docs
from langchain_openai import OpenAI

#llm = OpenAI(temperature=0)
chain = APIChain.from_llm_and_api_docs(
    llm,
    open_meteo_docs.OPEN_METEO_DOCS,
    verbose=True,
    limit_to_domains=["https://api.open-meteo.com/"],
)
chain.run(
    "What is the weather like right now in Munich, Germany in degrees Fahrenheit?"
)


# os.environ["TMDB_BEARER_TOKEN"] = ""
# from langchain.chains.api import tmdb_docs

# headers = {"Authorization": f"Bearer {os.environ['TMDB_BEARER_TOKEN']}"}
# chain = APIChain.from_llm_and_api_docs(
#     llm,
#     tmdb_docs.TMDB_DOCS,
#     headers=headers,
#     verbose=True,
#     limit_to_domains=["https://api.themoviedb.org/"],
# )
# chain.run("Search for 'Avatar'")