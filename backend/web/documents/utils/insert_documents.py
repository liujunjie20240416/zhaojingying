import lancedb
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import LanceDB
from langchain_text_splitters import RecursiveCharacterTextSplitter

from web.documents.utils.custom_embeddings import CustomEmbeddings


def insert_documents():
    loader = TextLoader(r'E:\wechat\texts\私聊_大白鹅_cleaned.txt',encoding='utf-8')#./web/documents/data.txt
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f'切分成{len(texts)} 个片段')

    embeddings = CustomEmbeddings()
    db = lancedb.connect('./web/documents/lancedb_storage')
    vector_db = LanceDB.from_documents(documents = texts,embedding=embeddings,connection=db,table_name='jingying',mode='overwrite')
    print(f'已插入{vector_db._table.count_rows()}行数据')