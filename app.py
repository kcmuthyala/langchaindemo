from customllm import CustomLLM

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

question = "Who won the FIFA World Cup in the year 1994?"

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = CustomLLM()
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.invoke(question))

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