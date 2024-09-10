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
import uipathlib
#import documentqa
    
load_dotenv()

client = AsyncOpenAI()

localdb = transactiondb.TransactionDB()
#pdfqa = documentqa.PdfQA()
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
    print( f'userid: {username} and password: {password}')
    match = localdb.authenticate( username, password)
    if match: # match ( userid, password, display_name, role) 
        config['configurable']['session_id'] = match[0] # user_id display_name
        '''
        with_message_history.invoke( 
                        {"messages": [HumanMessage(f"저의 사용자 ID는 {username} 입니다.")]},
                        config=config,
                    )
        '''
        return cl.User(identifier=match[1], metadata={"role": "USER"})
    else:
        return None

@tool
def check_post_delivery (postNum: str) -> str:
    """등기번호 등기 배송 상태  조회 """
    print(f'등기번호: {postNum}')
    args = {
        'name': 'ToolCallingQ',
        'folder': { 'Id': 1493557, 'Name': 'Shared'},
        'reference': 'PostOffice',
        'item': {
            'postNum': postNum
        }
    }
    tracker = uipathlib.UiPathQueueTracker(kwargs=args)
    tracker.start()
    return tracker.join()

@tool 
def lookup_user_request( userid: str ) -> str:
    '''사용자ID를 기반으로 사용자가 신청한 거래 내역서 조회'''
    print(f'거래 내역 조회 사용자 정보: {userid}')
    return localdb.list_requests(userid)

tools = [ check_post_delivery, lookup_user_request]
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
llm = llm.bind_tools(tools)

# Get the prompt to use - you can modify this!
#agent = create_openai_tools_agent(llm, tools, prompt)
#agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = prompt | llm

config = {"configurable": {"session_id": None}}

with_message_history = RunnableWithMessageHistory(chain, get_session_history)

mymsgs= []
    
@cl.on_message
async def on_message(message: cl.Message):
    continue_flag = True
    mycontent= message.content
    mymsgs.append(HumanMessage(content=mycontent))
    while continue_flag:
        response = chain.invoke( mymsgs)
        print( "Response: " , response.response_metadata, response.tool_calls) 
        if response.tool_calls:
            mymsgs.append( response)
            for tcall in response.tool_calls:
                if tcall['name'] == 'lookup_user_request':
                    tcall['args']['userid'] = config['configurable']['session_id']
                    result = lookup_user_request.invoke(tcall)
                    print("tool_calls_response: ", result)
                    #toolmsg = [ ToolMessage(content=result, name=tcall['name'],tool_call_id=tcall['id'] )]
                    mymsgs.append(result)
                    response = chain.invoke( mymsgs)
                    await cl.Message( content=response.content).send()
                    #mymsgs.remove( result)
                elif tcall['name'] == 'check_post_delivery':
                    result = check_post_delivery.invoke( tcall)
                    print("tool_calls_response: ", result)
                    print(mymsgs)
                    mymsgs.append( result)
                    response = chain.invoke( mymsgs)
                    await cl.Message( content=response.content).send()
                    #mymsgs.remove(result)
            continue_flag = False 
        else:
            await cl.Message(content=response.content).send()
            continue_flag = False 


@cl.on_chat_start
async def on_chat_start():
    print('on_chat_start')
    mymsgs.clear()
    cl.user_session.set("doc", None)
    if config['configurable']['session_id']: 
        await cl.Message(
            content=f"안녕하세요 {config['configurable']['session_id']}님 ",
        ).send()
    else:
        await cl.Message( 
            content="안녕하세요. 로그 아웃후 다시 로그인 부탁드립니다."
        ).send()
        
    
@cl.on_logout
def on_logout(request: Request, response: Response):
    print('on_logout') 
    mymsgs.clear()
    print(response.raw_headers)
    config['configurable']['session_id'] = None
    response.delete_cookie("my_cookie")
    
    
    
if __name__ == '__main__':
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)