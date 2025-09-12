import streamlit as st
from dni_genai.streamlit.chat import StreamlitChatAgent
from dni_genai.llms.azure_openai import AzureChatOpenAILLM
from dni_genai.agents.basic_chat import BasicChatAgent
from dni_genai.agents.vector_rag import BasicVectorRAGAgent
from dni_genai.utils.text2speech import AzureText2Speech
from dni_genai.embedder.azure import AzureOpenAIEmbedder
from dni_genai.knowledge_base.chroma import ChromaKB

# # For linux deployment
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


################################################################################################################
# CONFIG
SHOW_CHAT_HISTORY = False
EMBEDDER = AzureOpenAIEmbedder()
KB = ChromaKB(embedder=EMBEDDER,
              persist_directory='data/ba-chunks')
CHAT_MODEL = AzureChatOpenAILLM()

################################################################################################################

def display_message_in_all_chats(st_chats: list[StreamlitChatAgent],
                                 st_containers: list, character: str,
                                 message: str, ):
    for i in range(len(st_chats)):
        st_chats[i].display_message(character=character,
                                    st_container=st_containers[i],
                                    msg=message)

def get_model_response_from_all_chats(
        st_chats: list[StreamlitChatAgent],
        st_containers: list,
        user_prompt: str,):
    for i in range(len(st_chats)):
        st_chats[i].get_model_response(user_prompt=user_prompt,
                                       st_container=st_containers[i])

def render_history_from_all_chats(
        st_chats: list[StreamlitChatAgent],
        st_containers: list,):

    for i in range(len(st_chats)):
        st_chats[i].render_chat_history(st_container=st_containers[i])

def config_for_all_chats(st_chats: list[StreamlitChatAgent], config: dict):
    for i in range(len(st_chats)):
        for k, v in config.items():
            setattr(st_chats[i], k, v)

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    # Chat input
    chat_input_container = st.container()

    chat_windows_row = st.columns(2, gap='medium')
    chat1_container = chat_windows_row[0].container()
    chat1_container.header("GPT-4o")
    chat2_container = chat_windows_row[1].container()
    chat2_container.header("RAG")

    containers = [chat1_container, chat2_container]

    if "st_chats" not in st.session_state:
        st_chat1 = StreamlitChatAgent(
            agent=BasicChatAgent(chat_model=AzureChatOpenAILLM()),
            show_chat_history=SHOW_CHAT_HISTORY,
            text2speech=AzureText2Speech())

        st_chat2 = StreamlitChatAgent(
            agent=BasicVectorRAGAgent(
                knowledge_base=KB,
                embedder=EMBEDDER,
                chat_model=CHAT_MODEL,
                k=5
            ),
            show_chat_history=SHOW_CHAT_HISTORY,
            text2speech=AzureText2Speech())


        st.session_state["st_chats"] = [st_chat1, st_chat2]

    # Sidebar
    with st.sidebar:
        # Voice on button
        st.session_state.voice_on = st.toggle("Activate text-to-speech", value=False)

        # First chat button
        st.session_state.first_chat_on = st.toggle("GPT-4o", value=True)

        # Second chat button
        st.session_state.second_chat_on = st.toggle("RAG", value=True)

        st.session_state.rag_agent_persona = st.text_area(
            "RAG Agent Persona:",
            value=f"""You are a customer service agent from British Airways.  
Always elaborate your answer with as much useful info as possible and present your response in a easy to read format, and provide links for user to take action where applicable.

Always answer you don't know if you don't have the information to answer the question, and ask user to contact British Airways instead.

If you're providing a url, always replace the domain with https://www.britishairways.com.
""",
            height=500)

    config_for_all_chats(st_chats=st.session_state["st_chats"],
                         config={"voice_on": st.session_state.voice_on})

    if prompt:=chat_input_container.chat_input():
        if st.session_state.first_chat_on:
            # Get response from foundation model
            st.session_state["st_chats"][0].get_model_response(
                user_prompt=prompt,
                st_container=containers[0])

        if st.session_state.second_chat_on:
            # Get response from RAG agent
            st.session_state["st_chats"][1].get_model_response(
                user_prompt=prompt,
                system_prompt=st.session_state.rag_agent_persona,
                st_container=containers[1], doc_source_key="url")



