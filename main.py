import streamlit as st

from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.tracers.langchain import LangChainTracer
tracer = LangChainTracer(project_name="CRA")

from load_prompts import load_prompt
from major_selection import major_selection
from rag_process import rag_process

from dotenv import load_dotenv
load_dotenv()


#---------------------------------#
#-------- Deploy Settings --------#
#---------------------------------#
# __import__('pysqlite3')
# import sys
# import pysqlite3
# sys.modules['sqlite3'] = sys.modules["pysqlite3"]

# from google.oauth2 import service_account
# import google.generativeai as genai  # genai import 추가

# # Create API client.
# credentials = service_account.Credentials.from_service_account_info(
#     st.secrets["gcp_service_account"],
# )

# # Gemini 구성
# genai.configure(
#     credentials=credentials,
#)
###################################


#---------------------------------#
#---------- UI Settings ----------#
#---------------------------------#
st.title("📌수강편람 도우미✨")

# sidebar
with st.sidebar :
    # clear dialouge
    clear_btn = st.button("대화 초기화")
    # 학번 입력
    id = st.number_input(
        "학번", step=1, max_value=25, min_value=17
    )
    # 소속 단과대 입력
    dept = st.selectbox(
        "소속 대학",
        ("인문과학대",
         "사회과학대",
         "경영경제대",
         "호텔관광대",
         "자연과학대",
         "생명과학대",
         "인공지능융합대",
         "공과대",
         "예체능대")
    )
    # 전공 입력
    major = st.selectbox("전공", options=major_selection(dept))
    semester = st.select_slider(
        "재학 학기",
        ["신입생",
         "1학년",
         "2학년",
         "3학년",
         "4학년",
         "5학년",
         "초과학기"])
    # 편입 여부 입력
    bool_transfer = st.checkbox("편입")

# stuInfo
stuInfo = f"""
#학생정보\n
입학년도: {id}\n
소속대학: {dept}\n
전공: {major}\n
재학 학기기: {semester}\n
편입여부: {bool_transfer}
"""


#---------------------------------#
#-------- Message Storing --------#
#---------------------------------#
# dialouge storage
if "messages" not in st.session_state :
    st.session_state["messages"] = []

# add new message to storage
def add_message(role, message) :
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# print all dialouge
def print_messages() :
    for chat_message in st.session_state["messages"] :
        st.chat_message(chat_message.role).write(chat_message.content)


#---------------------------------#
#------------- Chain -------------#
#---------------------------------#
def create_chain() :
    # prompt
    prompt = load_prompt("prompts/basic.yaml", encoding="cp949")

    # model - 인증정보 추가
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        #credentials=credentials
    )

    # output parser
    output_parser = StrOutputParser()

    # chain
    chain = (
        {"context": rag_process,
         "question": RunnablePassthrough(),
         "stuInfo": lambda x: stuInfo}
        | prompt
        | llm
        | output_parser
    )

    return chain   


#---------------------------------#
#---------- User Action ----------#
#---------------------------------#
if clear_btn:
    st.session_state["messages"] = []
print_messages()

# user input
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# if input
if user_input :
    # user input
    st.chat_message("user").write(user_input)

    # chain
    chain = create_chain()

    # chain call
    response = chain.stream(user_input)
    with st.chat_message("assistant"):
        # create empty container and print token by stream
        container = st.empty()
        ai_answer = ""
        for token in response :
            ai_answer += token
            container.markdown(ai_answer)

    # add dialougue to storage
    add_message("user", user_input)
    add_message("assistant", ai_answer)
