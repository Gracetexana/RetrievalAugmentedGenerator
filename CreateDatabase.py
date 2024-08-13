# IMPORTS

## LOADING
from langchain_community.document_loaders import DirectoryLoader, JSONLoader

## SPLITTING??
from langchain_text_splitters import CharacterTextSplitter

## VECTORIZING
import transformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma




# MODIFIABLE VARIABLES
path = "../Documents/" # location relative to my SLURM file, DB.sh
embedding_model = "mixedbread-ai/mxbai-embed-large-v1"




# CODE

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
    "content_key" : ".containers.cna",
    "is_content_key_jq_parsable" : True,
    "metadata_func" : metadata_func,
    "text_content" : False
  },
  recursive = True
)

docs = loader.load()

## SPLITTING
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
documents = text_splitter.split_documents(docs)

## VECTORIZING
embeddings = HuggingFaceEmbeddings(
  model_name = embedding_model, 
  model_kwargs = {
    "device" : "cuda"
  }
)

db = Chroma.from_documents(
  documents, 
  embeddings,
  persist_directory = "../ChromaDB" # location relative to my SLURM file, DB.sh
)