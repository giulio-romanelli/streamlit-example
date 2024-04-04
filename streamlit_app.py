##-------------------------------------------------------------------------------------##
## Include and keys
##-------------------------------------------------------------------------------------## 
import speech_recognition as sr
from gtts import gTTS
import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from Utils import *

##-------------------------------------------------------------------------------------##
## Build chat
##-------------------------------------------------------------------------------------## 

# Load expertise
embedding_function = OpenAIEmbeddings()
vector_db = Chroma(
    persist_directory="./expertise", embedding_function=embedding_function
)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs=dict(k=5))

# Create prompt
my_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sei un assistente personale al servizio dei Clienti.\n"
            + "Nella risposta devi illustrare il tuo ragionamento logico step "
            + "by step e con tutti i dettagli.\n"
            "La risposta deve includere citazioni dai documenti.\n"
            + "Se non sai rispondere, usa la tua conoscenza interna.",
        ),
        (
            "human",
            "Usa le seguenti informazioni per rispondere alla domanda. Non "
            + "usare più di 5 frasi per rispondere.\n\n domanda:\n{question}"
            + "\n\ninformazioni: {context}\n\nRisposta:",
        ),
    ]
)

# Create chat
llm_4 = ChatOpenAI(model_name="gpt-4", temperature=0.0, model_kwargs={"top_p": 0})
my_chat = ( {"context": retriever, "question": RunnablePassthrough()} | my_prompt | llm_4 )

##-------------------------------------------------------------------------------------##
## Streamlit
##-------------------------------------------------------------------------------------## 

# Initiatlize environment
withAudio = True
withMic = True
inputFile = "input.wav" 
outputFile = "output.mp3"

# st.markdown("""
# <style>
#     [data-testid=stSidebar] {
#         margin-top: -75px;
#         width: 400px !important; 
#         background-color: white;
#     }
# </style>
# """, unsafe_allow_html=True)

# Sidebar
st.markdown("""
<style>
    [data-testid=stSidebar] {
        width: 500px !important; 
    }
</style>
""", unsafe_allow_html=True)
#st.sidebar.image('isybank_logo.svg', width = 450)
widget = st.sidebar.empty()
st.sidebar.markdown("")
st.sidebar.markdown("Ciao! Sono il tuo assistente virtuale personalizzato di isybank. Sono a disposizione per aiutarti ad aprire un conto, confrontare piani e prodotti")
st.sidebar.markdown("")
advisor = st.sidebar.selectbox( 'A chi preferisci rivolgerti?', ('Tommaso', 'Martina', 'Paolo', 'Sofia'))
if advisor == "Tommaso": 
    advisor_path = "./avatar/Tommaso"
    advisor_voice = "onyx"
elif advisor == "Martina": 
    advisor_path = "./avatar/Martina"
    advisor_voice = "fable"
elif advisor == "Paolo": 
    advisor_path = "./avatar/Paolo"
    advisor_voice = "echo"
elif advisor == "Sofia": 
    advisor_path = "./avatar/Sofia"
    advisor_voice = "nova"
widget.image(advisor_path + '_1.png', width=450)

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    if message["role"] == "assistant": 
        with st.chat_message(message["role"], avatar='logo.png'):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat prompt
prompt = ""
prompt = st.chat_input("Scrivi qui...")

# Audio prompt
volume = -1
question = ""
if withMic: 
    with st.sidebar:
        audio = mic_recorder(start_prompt="Parla qui...", stop_prompt="Registrazione completa", key='recorder', just_once=True, use_container_width=True)
    if audio: 
        audio_bytes = audio['bytes']
        volume = len(audio_bytes)
    if ( volume > 0 ):
        with open(inputFile, mode='wb') as f:
            f.write(audio_bytes)
        question=speechToTextOpenAI(inputFile, "it")
if ( len(question) > 0 ): prompt = question 

if prompt:

    # Chat message question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Commands
    question = prompt
    answer = ""
    if ( len(question) > 0 ): answer = my_chat.invoke(question).content
    
    # Play audio
    length = 1.0
    if ( withAudio and len(answer) > 0 ): 
        length = textToSpeechOpenAI(answer, outputFile, advisor_voice)
        playAudioEmbedded(outputFile)
        widget.image(advisor_path + '.gif', width=450)

    # Chat message answer
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant", avatar='logo.png'):
        #st.markdown(answer)
        st.write_stream(streamData(answer, length))
        widget.image(advisor_path + '_1.png', width=450)
