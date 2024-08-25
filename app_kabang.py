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
pdfqa = documentqa.PdfQA()
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
    match = localdb.authenticate( username, password)
    if match: # match ( userid, password, display_name, role) 
        config['configurable']['session_id'] = match[0] # user_id display_name
        return cl.User(identifier=match[1], metadata={"role": "USER"})
    else:
        return None

@tool
def check_post_delivery (reg_no: str) -> str:
    """등기번호 등기 배송 상태  조회 """
    print(f'등기번호: {reg_no}')
    return "LangChain"

@tool 
def lookup_user_request( userid: str ) -> str:
    '''사용자ID를 기반으로 사용자가 신청한 거래 내역서 조회'''
    print(f'거래 내역 조회 사용자 정보: {userid}')
    return localdb.list_requests(userid)

tools = [ check_post_delivery, lookup_user_request]
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, streaming=True)
llm = llm.bind_tools(tools)

# Get the prompt to use - you can modify this!
#agent = create_openai_tools_agent(llm, tools, prompt)
#agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = prompt | llm

config = {"configurable": {"session_id": None}}

with_message_history = RunnableWithMessageHistory(chain, get_session_history)


message_history = ChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    chat_memory=message_history,
    return_messages=True,
)

    
@cl.on_message
async def on_message(message: cl.Message):
    mycontent= message.content
    mymsgs = [HumanMessage(content=mycontent)]
    while True:
        response = with_message_history.invoke( 
            {"messages": mymsgs},
            config=config,
        )
        print( response.response_metadata, response.tool_calls) 
        if response.tool_calls:
            mymsgs.append( response)
            for tcall in response.tool_calls:
                if tcall['name'] == 'lookup_user_request':
                    tcall['args']['userid'] = config['configurable']['session_id']
                    result = lookup_user_request.invoke( tcall)
                    print(result)
                    #toolmsg = [ ToolMessage(content=result, name=tcall['name'],tool_call_id=tcall['id'] )]
                    mymsgs.append(result)
                    response = with_message_history.invoke( 
                        {"messages": mymsgs},
                        config=config,
                    )
        else:
            await cl.Message(content=response.content).send()
            break



@cl.on_chat_start
async def on_chat_start():
    print('on_chat_start')
    cl.user_session.set("doc", None)
    if config['configurable']['session_id']: 
        await cl.Message(
            content=f"안녕하세요 {config['configurable']['session_id']}님 ",
        ).send()
    else:
        await cl.Message( 
            content="안녕하세요"
        ).send()
        
    
@cl.on_logout
def on_logout(request: Request, response: Response):
    print('on_logout') 
    print(response.raw_headers)
    config['configurable']['session_id'] = None
    response.delete_cookie("my_cookie")