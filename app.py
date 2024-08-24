from openai import AsyncAzureOpenAI, AsyncOpenAI
from dotenv import load_dotenv
import chainlit as cl
import os
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
import chainlit as cl
from typing import Optional
import auth_db
    
load_dotenv()

client = AsyncOpenAI()

authdb = auth_db.AuthDB()
# Instrument the OpenAI client
#cl.instrument_openai()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in Korean.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    '''auth_callback with username and password'''
    match = authdb.authenticate( username, password)
    if match: # match ( userid, password, display_name, role) 
        return cl.User(identifier=match[2], metadata={"role": "USER"})
    else:
        return None
    '''
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "ADMIN"})
    '''

@tool
def check_post_delivery (reg_no: str) -> str:
    """등기번호 등기 배송 상태  조회 """
    print(f'등기번호: {reg_no}')
    return "LangChain"

@tool 
def lookup_user_request( user_id: str) -> str:
    '''사용자ID를 기반으로 사용자가 신청한 거래 내역서 조회'''
    print(f'거래 내역 조회 사용자 정보: {user_id}')
    return "거래 내역서"

tools = [ check_post_delivery, lookup_user_request]
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
llm = llm.bind_tools(tools)

tools = [TavilySearchResults(max_results=1)]
# Get the prompt to use - you can modify this!
#agent = create_openai_tools_agent(llm, tools, prompt)
#agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = prompt | llm

settings = {
    "temperature": 0,
    # ... more settings
}
config = {"configurable": {"session_id": "charles"}}

with_message_history = RunnableWithMessageHistory(chain, get_session_history)


@cl.on_message
async def on_message(message: cl.Message):
    print( message)
    response = with_message_history.invoke( 
        {"messages": [HumanMessage(content=message.content)]},
        config=config,
    )
    print( response) 
    await cl.Message(content=response.content).send()
