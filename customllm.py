import os
import requests

from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

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
        #import requests
        #make a API call to model on AWS
        #Retrieve JSON
        #Return string of output
        API_URL = "https://c9ejquzh6yum3xqf.us-east-1.aws.endpoints.huggingface.cloud/"
        API_TOKEN = os.environ["API_TOKEN"] 
        print(API_TOKEN)
        #API_TOKEN = "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"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        payload = {
            "inputs": str,
            "parameters": { #Try and experiment with the parameters
                "max_new_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.9,
                "do_sample": False,
                "return_full_text": False
            }
        }        
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()[0]['generated_text']

    # @property
    # def _identifying_params(self) -> Mapping[str, Any]:
    #     """Get the identifying parameters."""
    #     return {"n": self.n}