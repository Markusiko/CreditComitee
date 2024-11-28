from langchain_gigachat import GigaChatEmbeddings
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

load_dotenv()

# модели
embeddings = GigaChatEmbeddings(verify_ssl_certs=False, scope="GIGACHAT_API_PERS")

# пути до документов
docs_naming = {
    './data/docs/credit_inspection.txt': 'Заключение Кредитного Подразделения',
    './data/docs/reputation.txt': 'Заключение Подразделения Безопасности',
    './data/docs/prko.txt': 'ПРКО',
    './data/docs/law.txt': 'Заключение Юридического Подразделения',
    './data/docs/rm_conclusion.txt': 'Заключение Риск-Менеджера'
}

# загружаем документы
docs = []

for txt_file_path in docs_naming:
    loader = TextLoader(txt_file_path)
    file_docs = loader.load()

    for doc in file_docs:
        doc.metadata.update({'source_type': docs_naming[txt_file_path]})

    docs.extend(file_docs)

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
vector_store.save_local("./data/faiss_comitee_bot")
