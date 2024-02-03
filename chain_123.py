from operator import itemgetter

from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt1 = ChatPromptTemplate.from_template("what is the city {person} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what country is the city {city} in? respond in {language}"
)

model = ChatOpenAI()

chain1 = prompt1 | model | StrOutputParser()

chain2 = (
    {"city": chain1, "language": itemgetter("language")}
    | prompt2
    | model
    | StrOutputParser()
)

final_answer = chain2.invoke({"person": "obama", "language": "Chinese"})

print("ANSWER: ", final_answer)


from langchain_core.runnables import RunnablePassthrough

prompt1 = ChatPromptTemplate.from_template(
    "generate a {attribute} color. Return the name of the color and nothing else:"
)
prompt2 = ChatPromptTemplate.from_template(
    "what is a fruit of color: {color}. Return the name of the fruit and nothing else:"
)
prompt3 = ChatPromptTemplate.from_template(
    "what is a country with a flag that has the color: {color}. Return the name of the country and nothing else:"
)
prompt4 = ChatPromptTemplate.from_template(
    "What is the color of {fruit} and the flag of {country}?"
)

model_parser = model | StrOutputParser()

color_generator = (
    {"attribute": RunnablePassthrough()} | prompt1 | {"color": model_parser}
)

color_to_fruit = prompt2 | model_parser

color_to_country = prompt3 | model_parser

# magic of chain
question_generator = (
    color_generator | {"fruit": color_to_fruit, "country": color_to_country} | prompt4 
)

final_question = question_generator.invoke("warm")

print("FINAL QUESTION: ", final_question)
