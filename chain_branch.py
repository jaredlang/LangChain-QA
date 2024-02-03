from operator import itemgetter

from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_core.runnables import RunnablePassthrough

planner = (
    ChatPromptTemplate.from_template("Generate an argument about: {input}") ## input for planner
    | ChatOpenAI()
    | StrOutputParser()
    | {"base_response": RunnablePassthrough()}  ## output from planner 
)

arguments_for = (
    ChatPromptTemplate.from_template(
        "List the pros or positive aspects of {base_response}"  ## input for arguments_for
    )
    | ChatOpenAI()
    | StrOutputParser()
)
arguments_against = (
    ChatPromptTemplate.from_template(
        "List the cons or negative aspects of {base_response}"  ## input for arguments_against
    )
    | ChatOpenAI()
    | StrOutputParser()
)

# magic of chain
final_responder = (
    ChatPromptTemplate.from_messages(
        [
            ("ai", "{original_response}"),  ## one call to planner
            ("human", "Pros:\n{results_1}\n\nCons:\n{results_2}"),  ## two parallel calls
            ("system", "Generate a final response given the critique"),
        ]
    )
    | ChatOpenAI()
    | StrOutputParser()
)

chain = (
    planner
    | {
        "results_1": arguments_for,
        "results_2": arguments_against,
        "original_response": itemgetter("base_response"),
    }
    | final_responder
)

response = chain.invoke({"input": "cheating"})

print("RESPONSE: ", response)
