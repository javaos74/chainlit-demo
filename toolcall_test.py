from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import transactiondb
import os

localdb = transactiondb.TransactionDB()

@tool 
def lookup_user_request( userid: str ) -> str:
    '''사용자ID를 기반으로 사용자가 신청한 거래 내역서 조회'''
    print(f'거래 내역 조회 사용자 정보: {userid}')
    return localdb.list_requests(userid)

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [lookup_user_request]
llm_with_tools = llm.bind_tools(tools)
query="거래내역서 조회?"
messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
print(ai_msg)
messages.append( ai_msg)
messages.append( HumanMessage("charles"))

ai_msg = llm_with_tools.invoke(messages)
print(ai_msg)
messages.append( ai_msg)
if ai_msg.tool_calls:
    for tcall in ai_msg.tool_calls:
        if tcall['name'] == 'lookup_user_request':
            tool_resp = lookup_user_request.invoke( tcall)
            messages.append( tool_resp)
            ai_msg = llm_with_tools.invoke( messages)
            messages.append( ai_msg)
            print(messages)
            

'''
kabang 데모 아이디어 정리 

1. 로그인시  LLM에서 현재 사용자ID 역할을 알려준다.  역할은  user, admin 2개로 구분된다. 
   로그인시 역할에 맞는 내용을 LLM에게 알려준다. ( chat history에 기록 )
   1.1. user는 본인의 신청 정보를 조회할때 미리 알려준 사용자ID를 사용한다 
   1.2. admin은 신청정보를 조회시 사용자 ID가 없어 설정
2. 답변에서 tool_calls 부분을 찾아보고 있으면 해당 tool을 찾아서 호출하고 없으면 응답 메세지를 출력한다. 


3. UiPath Process를 호출시 우선 Queue를 사용한다. 
  3.1 큐를 추가하고 나면 큐가 처리되는 상태를 모니터링 하다가 진행사항 및 결과를 LLM으로 전달하고 최종 메세지를 출력 한다. 
  3.2 상태를 모니터링할 때는 orchestrator webhook을 사용해서 상태를 감지한다. 요청한 qitem id를 트래킹한다. 
  3.3 FastAPI를 이용해서 chainlit과 연동한다. 
  
'''