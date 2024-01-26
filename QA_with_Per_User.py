from pinecone import Pinecone
pc = Pinecone(api_key="bf4db278-4787-40c8-95d4-9e798f4727e1", environment="gcp-starter")
index = pc.Index("test-example")


# It appears Pinecone from langchain_community.vectorstores is conflicting with Pinecone from pinecone
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    RunnableBinding,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# The environment should be the one specified next to the API key
# in your Pinecone console
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone(index, embeddings, "text")

vectorstore.add_texts(["i worked at kensho"], namespace="harrison")
vectorstore.add_texts(["i worked at facebook"], namespace="ankush")

# This will only get documents for Harrison
vectorstore.as_retriever(
    search_kwargs={"namespace": "harrison"}
).get_relevant_documents("where did i work?")


template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

retriever = vectorstore.as_retriever()

configurable_retriever = retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)

chain = (
    {"context": configurable_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# # This will only get documents for Ankush
# vectorstore.as_retriever(
#     search_kwargs={"namespace": "ankush"}
# ).get_relevant_documents(
#     "where did i work?"
# )

answer = chain.invoke(
    "where did the user work?",
    config={"configurable": {"search_kwargs": {"namespace": "harrison"}}},
)
print(answer)

# # This will only get documents for Harrison
# vectorstore.as_retriever(
#     search_kwargs={"namespace": "harrison"}
# ).get_relevant_documents(
#     "where did i work?"
# )

answer = chain.invoke(
    "where did the user work?",
    config={"configurable": {"search_kwargs": {"namespace": "ankush"}}},
)
print(answer)
