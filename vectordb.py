#Lang chain is a framework that enhaces the use of LLM.
# It provides several features such as text splitters, VectorStores, embeddings (generation of text vectors) and many others. 
#There are other frameworks, but this project will focused only on lang chain.

#In this file we will build a Vector Store based on chroma
#VectorDb is a specialized database used to store and query numerical vectors that represent texts, images or audios.
#First, we need an embedding model to convert text into high-dimensional vectors. These vectors capture the semantic meaning of the phrase, 
#allowing us to compare the proximity of phrases based on their meaning.
import yaml

from langchain_chroma import Chroma

from langchain_community.document_loaders import DirectoryLoader, JSONLoader, PyPDFLoader #Read files from disk into lang chain documents (https://python.langchain.com/docs/how_to/document_loader_directory/)
                                                                                          #from formats JSON and PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter #Avoiding creating long vector the documents will be splitted using a splitter
                                                                  
from langchain_huggingface import HuggingFaceEmbeddings

with open("config.yaml","r") as file:
    config=yaml.safe_load(file)

def createChromaDB():
                                                                                    
    diretorio=config["ddocuments"]
    loader_pdf=DirectoryLoader(#Read all the pdf documents in the specified directory and creates an instance of PyPDFLoader
        path=diretorio,
        glob="*.pdf",
        loader_cls=PyPDFLoader
        
    )

    jq_schema='to_entries | map(.key + ":" + .value) | join("\\n")' # LangChain needs to know which part of the JSON to use as the document content.
                                                                    # This schema converts each key-value pair into a formatted string like "key: value".
                                                                    # All pairs are then joined together with line breaks to form a single text block.

    loader_json=DirectoryLoader(path=diretorio, # Read all the json documents in the specified directory and creates an instance of PyPDFLoader
                                glob="*.json",
                                loader_cls=JSONLoader,
                                loader_kwargs={"jq_schema":jq_schema})

    documents=loader_pdf.load()+loader_json.load()

    splitter=RecursiveCharacterTextSplitter( #Split the documents in chunks of size 500 and overlap of 75
        chunk_size=500,
        chunk_overlap=75
    )

    splited_document=splitter.split_documents(documents)


    name= config["modelname"] #Hugging face model for embedding (process of represent text as high dimensional vectors)

    embedding_model=HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs = {'device': 'cpu'},
        encode_kwargs = {'normalize_embeddings': True}#Creates the vectordb with normalized numbers
    )

    vectordb=Chroma.from_documents(splited_document,  #Building the vectorDB
                                   embedding=embedding_model,
                                   persist_directory=config["dvectordb"])



if __name__=="__main__":
    createChromaDB()

