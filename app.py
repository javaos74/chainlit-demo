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
        config['configurable']['session_id'] = match[1] # display_name
        return cl.User(identifier=match[1], metadata={"role": "USER"})
    else:
        return None

@tool
def check_post_delivery (reg_no: str) -> str:
    """등기번호 등기 배송 상태  조회 """
    print(f'등기번호: {reg_no}')
    return "LangChain"

@tool 
def lookup_user_request( user_id: str ) -> str:
    '''사용자ID를 기반으로 사용자가 신청한 거래 내역서 조회'''
    print(f'거래 내역 조회 사용자 정보: {user_id}')
    return "거래 내역서"

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
    print( message.elements)
    docsearch = cl.user_session.get("doc")
    if message.elements:
        rchain = cl.user_session.get("retrieval.chain")  # type: ConversationalRetrievalChain
        file = message.elements[0]
        msg = cl.Message(content=f"Processing `{file.name}`...")
        await msg.send()
        docsearch = await cl.make_async(pdfqa.get_docsearch)(file)
        if not rchain:
            rchain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
                chain_type="stuff",
                retriever=docsearch.as_retriever(),
                memory=memory,
                return_source_documents=True,
            )
            cl.user_session.set("retrieval.chain", rchain)
        cb = cl.AsyncLangchainCallbackHandler()
        res = await chain.acall(message.content, callbacks=[cb])
        cl.user_session.set("doc", docsearch)
        print(res)
        answer = res["answer"]
        source_documents = res["source_documents"]  # type: List[Document] 
        text_elements = []  # type: List[cl.Text]

        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                # Create the text element referenced in the message
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            source_names = [text_el.name for text_el in text_elements]

            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found"
        await cl.Message(content=answer, elements=text_elements).send()
    elif docsearch:
        rchain = cl.user_session.get("retrieval.chain")  # type: ConversationalRetrievalChain
        cb = cl.AsyncLangchainCallbackHandler()
        res = await rchain.acall(message.content, callbacks=[cb])
        answer = res["answer"]
        source_documents = res["source_documents"]  # type: List[Document] 
        text_elements = []  # type: List[cl.Text]

        if answer and source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                # Create the text element referenced in the message
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            source_names = [text_el.name for text_el in text_elements]

            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found"
            await cl.Message(content=answer, elements=text_elements).send()
        else:
            response = with_message_history.invoke( 
                {"messages": [HumanMessage(content=message.content)]},
                config=config,
            )
            print( response) 
    else:
        response = with_message_history.invoke( 
            {"messages": [HumanMessage(content=message.content)]},
            config=config,
        )
        print( response.response_metadata, response.tool_calls) 
        if response.tool_calls:
            for tcall in response.tool_calls:
                if tcall.name == 'lookup_user_request':
                    lookup_user_request( config['configurable']['session_id'])
        await cl.Message(content=response.content).send()



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