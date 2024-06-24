"""
CHANGE AS NEEDED
"""

# enter Hugging Face auth code
hfAuth = "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" #read only
hfAuth = "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" #fine-grained

# enter model that generator is based on (as seen on Hugging Face)
generatorModel = 'meta-llama/Meta-Llama-3-8B'

# enter model that retriever is based on (as seen on Hugging Face)
retrieverModel = "sentence-transformers/all-mpnet-base-v2"

# enter location of vector database
dbLoc = "/shared/rc/malont/acsac/faiss_index"

# enter the location of the questions you wish to ask
questionsLoc = "/home/gtl1500/CyberAdvisory/llama2Script/RAGQA.xlsx"




"""
IMPORTS
"""

# necessary for full script
import torch

# genertor
import transformers
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer

# retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# put them together to create a RAG
#from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# question answering
import pandas as pd




"""
SCRIPT
"""

# RETRIEVER
def createRetriever(retrieverModel):
  """
  creates the retriever portion of the RAG (retrieval augmented generator) based on selected model form HuggingFace
  retriever uses query to retrieve relevant documents from database (FAISS, in this case)
  """
  # access database
  dbEmbeddings = HuggingFaceEmbeddings(
      model_name = retrieverModel,
      model_kwargs = {
          "device": "cuda"
      }
  )
  
  database = FAISS.load_local(
      dbLoc,
      dbEmbeddings,
      allow_dangerous_deserialization = True
  )
  
  retriever = database.as_retriever(
    #search_type="similarity_score_threshold", 
    #search_kwargs={
    #"score_threshold": 0.7,
    #"k": 4
    #}
  )
  
  return retriever
  
  
# GENERATOR
def createGenerator(generatorModel):
  """
  creates the generator portion of the RAG (retrieval augmented generator) based on selected model from HuggingFace
  generator uses query + documents from retriever to generate appropriate response
  """
  bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
  '''
  generatorPipe = AutoModelForCausalLM.from_pretrained(
    generatorModel,
    token = hfAuth,
    #temperature = 0.1,
    #max_new_tokens = 1024,
    #repetition_penalty = 1.1,
    #torch_dtype = torch.bfloat16,
    quantization_config = bnb_config,
    #low_cpu_mem_usage = True
  )
  
  
  generatorPipe = transformers.pipeline(
      token = hfAuth,
      task = "text-generation",
      model = generatorModel,
      tokenizer = tokenizer,
      torch_dtype = torch.bfloat16,
      device_map = "auto",
      num_return_sequences = 1,
      eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
  )
  '''
  
  tokenizer = AutoTokenizer.from_pretrained(generatorModel)
  
  generatorPipe = transformers.pipeline(
    "text-generation", 
    model=generatorModel, 
    model_kwargs={
      "torch_dtype": torch.bfloat16
    }, 
    device_map="auto",
    eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    token = hfAuth
)
  
  
  generator = HuggingFacePipeline(
      pipeline = generatorPipe,
      model_kwargs = {"temperature": 0.1}
  )
  
  return generator


def createPrompt(promptText):
  
  template = """
  <|begin_of_text|>
  <|start_header_id|>system<|end_header_id|>
  {promptText}
  <|eot_id|>
  <|start_header_id|>user<|end_header_id|>
  Context: 
  {context}
  
  Question: {input}
  <|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
  """
  
  template = """You are a digital assistant that simply and thoroughly explains cybersecurity concepts. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use five sentences maximum. Keep the answer as concise as possible. 
  
  Context:
  
  {context}

  Question: {input}

  Digital Assistant:"""
  
  prompt = PromptTemplate(
    template = template,
    partial_variables = {"promptText": promptText},
    input_variables = ["context", "input"]
  )
  
  return prompt
  
def createRAG(retrieverModel, generatorModel, promptText = "Answer any questions in layman's terms based solely on the context: "):  
  retriever = createRetriever(retrieverModel)
  generator = createGenerator(generatorModel)
  prompt = createPrompt(promptText)
  
  response = create_stuff_documents_chain(generator, prompt)
  RAG = create_retrieval_chain(retriever, response)
  return RAG



"""
SCRIPT
"""
query = "Are there any devices on my network that are vulnerable to cross-site scripting (XSS) attacks, and how can I protect against these attacks?"

#retriever = createRetriever(retrieverModel)
#print(retriever.invoke(query))

#generator = createGenerator(generatorModel)
#print(generator(query))

RAG = createRAG(retrieverModel, generatorModel)
print(RAG.invoke({"input": query}).get("answer"))
