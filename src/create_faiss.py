from langchain_gigachat import GigaChatEmbeddings
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import re

dotenv_path = '../.env'
load_dotenv(dotenv_path)

# модели
embeddings = GigaChatEmbeddings(verify_ssl_certs=False, scope="GIGACHAT_API_PERS")

# загружаем документ
loader = PyPDFLoader(
    "../data/docs/ozon_fin_res.pdf",
)
docs = loader.load()
for doc in docs:
    doc.page_content = re.sub('\n+', '\n', doc.page_content)

# делим документы
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")  # загружаем токенизатор
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=512, chunk_overlap=128)
splitted_docs = text_splitter.split_documents(docs)

# загружаем в vectorestore
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
vector_store.add_documents(documents=splitted_docs)
vector_store.save_local("../data/faiss_index")
