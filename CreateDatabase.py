# IMPORTS

## LOADING
from langchain_community.document_loaders import DirectoryLoader, JSONLoader

## SPLITTING??
import json
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.docstore.document import Document

## VECTORIZING
import transformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma




# MODIFIABLE VARIABLES
path = "../../cvelistV5/cves/2024/" # location relative to my SLURM file, DB.sh
embedding_model = "mixedbread-ai/mxbai-embed-large-v1"




# CODE
# INITIALIZE DATABASE
embeddings = HuggingFaceEmbeddings(
  model_name = embedding_model, 
  model_kwargs = {
    "device" : "cuda"
  }
)

db = Chroma(
  embedding_function = embeddings,
  collection_name = "CVEs",
  persist_directory = "../ChromaDB" # location relative to my SLURM file, DB.sh
)

## LOADING
def metadata_func(record : dict, metadata : dict) -> dict:
  metadata = record.get("cveMetadata")
  
  return metadata

loader = DirectoryLoader(
  path, 
  glob = "**/*.json",
  silent_errors = True,
  loader_cls = JSONLoader,
  loader_kwargs = {
    "jq_schema" : ".",
    "content_key" : ".containers",
    "is_content_key_jq_parsable" : True,
    "metadata_func" : metadata_func,
    "text_content" : False
  },
  recursive = True
)

docs = loader.load()

## SPLITTING
splitter = RecursiveJsonSplitter(max_chunk_size=300)

for document in docs:
  chunks = splitter.split_json(json.loads(document.page_content))

  for chunk in chunks:
    chunked_doc = [Document(
        page_content = str(chunk),
        metadata = document.metadata
      )]
    
    db.add_documents(
      documents = chunked_doc
    )