from openai import AsyncAzureOpenAI, AsyncOpenAI
from dotenv import load_dotenv
import chainlit as cl
import os
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from fastapi import Request, Response
from langchain_core.runnables.history import RunnableWithMessageHistory
import chainlit as cl
from typing import Optional
import transactiondb
import documentqa

load_dotenv()

client = AsyncOpenAI()
localdb = transactiondb.TransactionDB()

@tool
def check_post_delivery (reg_no: str) -> str:
    """등기번호 등기 배송 상태  조회 """
    print(f'등기번호: {reg_no}')
    return "LangChain"

@tool 
def lookup_user_request() -> str:
    '''사용자ID를 기반으로 사용자가 신청한 거래 내역서 조회'''
    print(f'거래 내역 조회 사용자 정보: ')
    return localdb.list_requests('charles')

tools = [ check_post_delivery, lookup_user_request]
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, streaming=True)
llm = llm.bind_tools(tools)


query = "거래 내역서를 요청했는데 언제 받을 수 있어요?"
messages = [HumanMessage(query)]
ai_msg= llm.invoke( messages)
messages.append(ai_msg)
print(messages)
for tool_call in ai_msg.tool_calls:
    selected_tool = {"lookup_user_request": lookup_user_request}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)
for msg in messages:
    print(msg)
print(llm.invoke( messages))