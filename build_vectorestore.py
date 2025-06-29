import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

# 1. PDF dizinini tanımla
PDF_DIR = "./books"  # PDF dosyalarının olduğu klasör
VECTORSTORE_DIR = "./vectorstore/chroma"  # ChromaDB dosyalarının kaydedileceği klasör

# 2. Embedding modeli
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Metin bölücü
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 4. PDF'leri yükle
all_docs = []
for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, filename)
        loader = PyPDFLoader(pdf_path)
        print(f"Yükleniyor: {filename}")
        docs = loader.load()
        split_docs = splitter.split_documents(docs)
        all_docs.extend(split_docs)

print(f"Toplam parça sayısı: {len(all_docs)}")

# 5. Chroma'ya aktar ve kaydet
vectorstore = Chroma.from_documents(documents=all_docs, embedding=embedding_model, persist_directory=VECTORSTORE_DIR)
vectorstore.persist()

print("✅ Tüm PDF içerikleri başarıyla ChromaDB'ye aktarıldı.")
