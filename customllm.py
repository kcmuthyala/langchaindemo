import os
import requests
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_community.llms import HuggingFaceTextGenInference
from dotenv import load_dotenv

load_dotenv()
default_llm = HuggingFaceTextGenInference(
    inference_server_url=os.getenv("LLM_API"),
    max_new_tokens=1024,
    top_p=0.9,
    server_kwargs={
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('API_TOKEN')}"
        }
    }
)

class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        load_dotenv()
        API_KEY = os.getenv("API_TOKEN")
        headers = {"Authorization": f"Bearer {API_KEY}"}
        API_URL = os.getenv("LLM_API")
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.9,
                "do_sample": False,
                "return_full_text": False
            }
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()[0]['generated_text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"prompt": "Input query to the LLM for answers."}

# import os
# import requests

# from typing import Any, List, Mapping, Optional

# from langchain_core.callbacks.manager import CallbackManagerForLLMRun
# from langchain_core.language_models.llms import LLM
# from dotenv import load_dotenv

# class CustomLLM(LLM):

#     @property
#     def _llm_type(self) -> str:
#         return "custom"

#     def _call(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> str:
#         #import requests
#         #make a API call to model on AWS
#         #Retrieve JSON
#         #Return string of output
#         #API_URL = "https://c9ejquzh6yum3xqf.us-east-1.aws.endpoints.huggingface.cloud/" # Embeddings model LLM 
#         API_URL = "https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud/"  # Test LLM
#         API_TOKEN = os.environ["API_TOKEN"] 
#         headers = {"Authorization": f"Bearer {API_TOKEN}"}
#         payload = {
#             "inputs": prompt,
#             "parameters": { #Try and experiment with the parameters
#                 "max_new_tokens": 1024,
#                 "temperature": 0.6,
#                 "top_p": 0.9,
#                 "do_sample": False,
#                 "return_full_text": False
#             }
#         }        
#         response = requests.post(API_URL, headers=headers, json=payload)
#         output=response.json()[0]['generated_text'] 
#         return output

#     @property
#     def _identifying_params(self) -> Mapping[str, Any]:
#         """Get the identifying parameters."""
#         return {"promt":"Input query to the LLM for answer"}