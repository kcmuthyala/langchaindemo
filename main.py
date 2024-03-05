import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceTextGenInference
from langchain.agents import create_sql_agent
#from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
#from langchain.chat_models import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAI
#from langchain_community import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain.prompts.chat import ChatPromptTemplate
from sqlalchemy import create_engine

cs="mysql+pymysql://root:chandu@localhost:3306/test1"
db_engine=create_engine(cs)
db=SQLDatabase(db_engine)
print(db.get_usable_table_names())

load_dotenv()
default_llm = HuggingFaceTextGenInference(
    inference_server_url=os.getenv("LLM_API"),
    max_new_tokens=50,
    top_p=0.9,
    server_kwargs={
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('API_TOKEN')}"
        }
    }
)

#llm=ChatOpenAI(temperature=0.0, api_key="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")
#llm=ChatOpenAI(temperature=0.0,base_url="https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud/")
#llm=ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", api_key="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")
#print(llm.invoke("what is your name"))
#llm=OpenAI(temperature=0.0,base_url="https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud/", model="zephyer-7b-beta")
print(default_llm.invoke("what is your name"))
sql_toolkit=SQLDatabaseToolkit(db=db,llm=default_llm)
sql_toolkit.get_tools()
prompt=ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        you are a very intelligent AI assitasnt who is expert in identifying relevant questions from user and converting into sql queries to generate correcrt answer.
        Please use the below context to write the mysql queries.
       context:
       you must query against the connected database, information is in 'mebers' table. It coulmns name, sex, dob. show me member details based on question.
        """
        ),
        ("user","{question}\ ai: ")
    ]
)
agent=create_sql_agent(llm=default_llm,toolkit=sql_toolkit,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
#agent=create_sql_agent(llm=llm,verbose=True)
#agent = create_sql_agent(llm=llm, db=db, agent_type="openai-tools", verbose=True)
#agent.run(prompt.format_prompt(question="What is John's date of birth?"))
agent.invoke("What is John's sex")
#gent.invoke("In what movie JOHNNY was acted ?") 
#agent.invoke("")

#db_chain = SQLDatabaseChain(llm=default_llm, database=db, verbose=True)
