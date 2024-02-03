#from pydantic import BaseModel, Field # pydantic v2 not compatible with langchain
from pydantic.v1 import BaseModel, Field # pydantic v1 compatible with langchain

from langchain.tools import StructuredTool

class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="multiply numbers",
    args_schema=CalculatorInput,
    return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well
)

print(calculator.name)
print(calculator.description)
print(calculator.args)