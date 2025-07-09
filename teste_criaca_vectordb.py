import unittest.mock
import yaml
import os
import shutil
import unittest
from unittest.mock import patch
from langchain.schema import Document
from vectordb import createChromaDB

with open("config.yaml","r") as file:
    config=yaml.safe_load(file)
    
class Test_criacao_vectordb(unittest.TestCase):
    # These methods are called automatically by unittest:
# - setUpClass: runs once before any tests in the class
# - tearDownClass: runs once after all tests in the class
# - setUp: runs before each individual test method
    @classmethod
    def setUpClass(cls):
        os.makedirs(config["dvectordb"],exist_ok=True)
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(config["dvectordb"], ignore_errors=True)
    @classmethod
    def setUp(self):
        if os.path.exists(config["dvectordb"]):
            for file in os.listdir(config["dvectordb"]):
                path=os.path.join(config["dvectordb"],file)
                if os.path.isfile(path):
                    os.unlink(path)
                    
    @patch("vectordb.DirectoryLoader")  # Indicates where the object is used — in this case, inside the 'vectordb' file. We're mocking (simulating) the DirectoryLoader for testing.
    @patch("vectordb.Chroma")
    @patch("vectordb.HuggingFaceEmbeddings")
    
    #The order of the mock inputs in the function is the opposite of the order of the paths.
                                     #1              #2          #3
    def test_1_criacaoVectorDB(self, mock_embedding,mock_chroma,mock_loader):
        
        #Creating mock from mock_loader for each loader (pdf,json)
        mock_loader_pdf=unittest.mock.Mock()
        mock_loader_json=unittest.mock.Mock()
        
        
        # Initiating the PDF mock, simulating the process represented by the variable loader_pdf in the vectordb.py file.
        # When the load method is called, these two documents will be returned.
        mock_loader_pdf.load.return_value=[
            Document(page_content="Teste"),
            Document(page_content = "TESTE2")
        ]
        
        #Iniating the json mock
        mock_loader_json.load.return_value=[
            Document(page_content="Teste json"),
            Document(page_content = "TESTE json 2")
        ]
        
        #Using side_effect because DirectoryLoader is called twice
        #the first one for pdf files and the second for json
        #side_effect allow capturing different return values for each time the mock is called
        mock_loader.side_effect=[mock_loader_pdf,mock_loader_json]
            
        createChromaDB()
        
        mock_loader_pdf.load.assert_called_once()
        mock_loader_json.load.assert_called_once()
        
        name= config["modelname"] 
        mock_embedding.assert_called_with(
        model_name=name,
        model_kwargs = {'device': 'cpu'},
        encode_kwargs = {'normalize_embeddings': True})
        
        mock_chroma.from_documents.assert_called_once()
        
if __name__=="__main__":
    print("Iniciando testes \n")
    unittest.main(verbosity=2)