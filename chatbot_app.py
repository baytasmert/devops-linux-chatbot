# chatbot_app.py

import os
import streamlit as st
from dotenv import load_dotenv
import joblib
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from intent_model_utils import predict_intent

# Ortam değişkenlerini yükle
load_dotenv()

# Streamlit arayüz ayarları
st.set_page_config(page_title="DevOps & Linux Chatbot", layout="wide")
st.title("🛠️ DevOps & Linux Asistanı")

# LLM seçimi
model_choice = st.selectbox("Kullanmak istediğiniz LLM modeli:", ["gemini", "openrouter"])

# Intent model yüklemeleri
intent_model = SentenceTransformer("all-MiniLM-L6-v2")
intent_classifier = joblib.load("models/intent_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Vektör verisi yükle
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./vectorstore/chroma",
    embedding_function=embedding_model,
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Sistem mesajı (prompt context)
system_prompt = (
    "Sen bir DevOps ve Linux uzmanısın. Kullanıcının sorusunu analiz et, ardından doğru kaynaklardan veri çekerek açık, teknik ama anlaşılır bir yanıt üret. "
    "Yanıtların kısa ama etkili olsun, mümkünse örnek kullanım ver. "
    "Sadece aşağıdaki bağlamı kullan, eğer yanıt yoksa bilmiyorum de.\n\n{context}"
)

# Prompt tanımı
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# LLM seçim fonksiyonu
def get_llm(choice):
    if choice == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
            max_tokens=500
        )
    elif choice == "openrouter":
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="mistralai/mistral-7b-instruct:free",
            temperature=0.3
        )

llm = get_llm(model_choice)

# RAG zinciri: retrieval + stuff + LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Kullanıcı girişi
user_input = st.chat_input("Bir soru yazın:")

# Giriş işleme
if user_input:
    st.chat_message("user").markdown(user_input)
    intent = predict_intent(user_input)

    if intent == "greeting":
        response = "Merhaba! DevOps veya Linux ile ilgili nasıl yardımcı olabilirim?"
    elif intent == "farewell":
        response = "Görüşmek üzere! Başarılar dilerim."
    elif intent == "thank_you":
        response = "Rica ederim! Yardımcı olabildiysem ne mutlu."
    elif intent == "unknown":
        response = "Bu soruyu anlayamadım. Lütfen tekrar ifade eder misiniz?"
    elif intent == "off_topic":
        response = "Ben bir DevOps ve Linux asistanıyım. Bu konular dışındaki sorularda yardımcı olamıyorum 🙈."
    else:
        opening = f"Tamam, '{intent.replace('_', ' ')}' konusuna odaklanıyorum."
        try:
            result = rag_chain.invoke({"input": user_input})
            response = f"{opening}\n{result['answer']}"
        except Exception as e:
            response = f"⚠️ Bir hata oluştu: {str(e)}"

    st.chat_message("assistant").markdown(f"**🤖 Bot:** {response}")
