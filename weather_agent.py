import math
from langchain_openai import ChatOpenAI
from langchain.agents import tool 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages
)
from langchain.agents.output_parsers.openai_tools import (
    OpenAIToolsAgentOutputParser
)
from langchain.agents import AgentExecutor

import os
from dotenv import load_dotenv
import requests

load_dotenv()
openWeatherApiKey = os.environ["openWeatherApiKey"]


# Define a function as a tool with decorator 
@tool
def celsius_to_fahrenheit(celsius) -> float:
  """
  Converts a temperature in Celsius to Fahrenheit.

  Args:
      celsius: The temperature in degrees Celsius.

  Returns:
      The temperature in degrees Fahrenheit.
  """
  fahrenheit = math.floor((celsius * 9/5) + 32)
  return fahrenheit


@tool 
def get_current_temperature(city: str) -> int:
    """Retrieves the current temperature for a given city using the OpenWeatherMap API.

    Args:
        city (str): The name of the city to get the temperature for.

    Returns:
        float: The current temperature in degrees Celsius. Returns None if an error occurs.
    """

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": openWeatherApiKey, "units": "metric"}  # Use metric for Celsius

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for error status codes

        data = response.json()
        temperature_kelvin = data["main"]["temp"]
        return temperature_kelvin

    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


tools = [get_current_temperature, celsius_to_fahrenheit]

# Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a very powerful assistant, but don't know current events"), 
    ("user", "{input}"), 
    MessagesPlaceholder(variable_name="agent_scratchpad"), 
])

# Define a factual LLM 
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Bind LLM with tools
llm_with_tools = llm.bind_tools(tools)

# Create an agent 
agent = (
    {
        "input": lambda x: x["input"], 
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        )
    }
    | prompt 
    | llm_with_tools
    #| llm
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#answer = agent_executor.stream({"input": "what's the current temperature in London?"})

#answer = agent_executor.stream({"input": "what's the current temperature in London? Use Fahrenheit instead Celsius. "})

#answer = agent_executor.stream({"input": "what's the current temperature in Houston?"})

answer = agent_executor.stream({"input": "what's the current temperature in London? Use Fahrenheit instead Celsius. "})

print("ANSWER: ", list(answer))