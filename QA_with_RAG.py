import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()
#os.environ["OPENAI_API_KEY"]

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# add the chat history into the structured prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

# format the question with the chat history
from langchain_core.messages import AIMessage, HumanMessage

contextualize_q_chain.invoke(
    {
        "chat_history": [
            HumanMessage(content="What does LLM stand for?"),
            AIMessage(content="Large language model"),
        ],
        "question": "What is meant by large",
    }
)

# format the question with the context

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever | format_docs
    )
    | qa_prompt
    | llm
)

chat_history = []

# Ask a question 
question = "What is Task Decomposition?"
ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg])

second_question = "What are common ways of doing it?"
answer = rag_chain.invoke({"question": second_question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=second_question), answer])

print(chat_history)

# cleanup
vectorstore.delete_collection()
