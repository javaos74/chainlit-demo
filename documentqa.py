import os
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
import base64 
from pinecone import Pinecone
import chainlit as cl


class PdfQA:
    '''PDF document question-answering module'''
    def __init__(self):
        self.pc = Pinecone()
        self.namespaces = set()
        self.index_name = "pdfqa"
        for ns in self.pc.Index( self.index_name).describe_index_stats()['namespaces'].keys():
            self.namespaces.add( ns)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        self.embeddings = OpenAIEmbeddings()
        self.docsearch: Pinecone 
    
    def process_file(self, file: cl.File):
        if file.mime == "text/plain":
            Loader = TextLoader
        elif file.mime == "application/pdf":
            Loader = PyPDFLoader

            loader = Loader(file.path)
            documents = loader.load()
            docs = self.text_splitter.split_documents(documents)
            for i, doc in enumerate(docs):
                doc.metadata["source"] = f"source_{i}"
            return docs


    def delete_namespace(self, namespace:str):
        self.pc.Index( self.index_name).delete( namespace=namespace, delete_all=True) 
        
    def get_docsearch(self, file: cl.File):
        docs = self.process_file(file)

        # Save data in the user session
        cl.user_session.set("docs", docs)

        # Create a unique namespace for the file
        # change from file.id -> file.name 
        namespace = base64.b64encode(file.name.encode()).decode() # bytes

        if namespace in self.namespaces:
            self.docsearch = PineconeVectorStore.from_existing_index(
                index_name=self.index_name, embedding=self.embeddings, namespace=namespace
            )
        else:
            self.docsearch = PineconeVectorStore.from_documents(
                docs, self.embeddings, index_name=self.index_name, namespace=namespace
            )
            self.namespaces.add(namespace)

        return self.docsearch

