import sys
sys.path.append("..") # allows import of files in parent directory (I need this for HFAuth right below because I am running this file from the SLURM file, RAG.sh)

"""
IMPORTS
"""
# PersonalTokens is my own file
from PersonalTokens import HFAuth

import torch
from huggingface_hub import login
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from transformers import BitsAndBytesConfig, GenerationConfig
from langchain_huggingface import HuggingFacePipeline
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.query_constructor.base import StructuredQueryOutputParser
import re
from langchain_core.prompts import PromptTemplate





"""
CHANGE AS NEEDED
"""
# enter Hugging Face auth code
hfAuth = HFAuth # old version of this code had my actual auth token here - that token has been expired and the new token is hidden using .gitignore

# enter model that generator is based on (as seen on Hugging Face)
llmModel = 'meta-llama/Meta-Llama-3-8B'

# enter model that embeds and decodes documents in database (as seen on Hugging Face)
embeddingModel = "mixedbread-ai/mxbai-embed-large-v1"

# enter location of vector database (for me, it is relation to my SLURM file RAG.sh)
dbLoc = "../ChromaDB"





def main():
  llm = create_llm() # initialize llm
  
  question = "Does the vulnerability described in CVE-2024-0011 allow for the execution of arbitrary code on the affected system?"

  rag(
    question,
    llm,
    audience = "a child"
  )
  
  rag(
    question,
    llm,
    task = "sc",
    audience = "a cybersecurity expert"
  )

  rag(
    question,
    llm,
    task = "cons",
    audience = "a class of military recruits"
  )





"""
LLM
"""
def create_llm():
  """
  Initizalizes llm specified by llmModel from HuggingFace.
  
  Parameters
  ----------
  llmModel: String
      The string through which the model can be accessed from HuggingFace.
      
  Returns
  -------
  HuggingFacePipeline
      The llm as tailored below.
  """
  bnb_config = BitsAndBytesConfig( # quantization
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
  )
  
  llm = HuggingFacePipeline.from_model_id(
    model_id = llmModel,
    task = "text-generation",
    model_kwargs ={
      "temperature" : 0.1, # how random outputs are; 0.1 is not very random
      "do_sample" : True,
      "device_map" : "auto", # automatically allocates resources between GPUs (and CPUs if necessary)
      "quantization_config" : bnb_config, # quantization: more efficient computing, less memory
      "trust_remote_code" : True
    },
    pipeline_kwargs = {
      "max_new_tokens" : 1028,
      "repetition_penalty" : 1.1 # prevents some repetition in model responses
    }
  )

  tokenizer = llm.pipeline.tokenizer

  llm.pipeline.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
      ]
  
  return llm





"""
RAG
"""
def rag(
  question,
  llm,
  task = "qa",
  audience = "a general audience"
):
  """
  Print llm output.
  
  Parameters
  ----------
  question: string
      The string used to retrieve relevant documents (does not strictly have to be a question).
      If question-answering, the question that is answered.
  llm: HuggingFacePipeline
      The llm that will be generating outputs based on prompts.
  retriever: SelfQueryingRetriever
      A retriever that can use the provided question to filter documents using metadata and then search through the filtered documents for the information referenced in the question.
  chat_history: list
      A condensed version of the conversation; whole chat history will not be remembered.
  task: String
      Either "qa," "cons," or "sc" for question-answering, consequence analysis, and scenario creation, respectively.
      Determines which task the RAG will perform.
  audience: String
      Can change the tone of the output.
      
  Returns
  -------
  list
      A chat history.
  """
  partial_prompt, printTask = choose_task(task)
  
  print(printTask + "for " + audience + ": " + question + "\n\n")

  prompt = partial_prompt.partial(
    audience = audience
  )

  retriever = create_retriever()

  chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | RunnableLambda(get_llm_response)
  )

  print(chain.invoke(question))
  print()
  print()



# TASK
def choose_task(task):
  if (task == "cons"):
    partial_prompt = cons_prompt()
    printTask = "Consequences "
    
  elif (task == "sc"):
    partial_prompt = sc_prompt()
    printTask = "Scenario creator "
  
  else:
    if (task != "qa"):
      print("""
        Task not recognized, defaulting to question-answering. Task must be one of the following:
        
        qa (question-answering): answers the given question
        cons (consequences): outlines consequences of the specified cyberattack
        sc (scenario creator): describes a practice attack scenario centered around the specified cyberattack
      """)
      
    partial_prompt = qa_prompt()
    printTask = "Question answering "

  return partial_prompt, printTask



def qa_prompt():
  """
  Prompt used for question-answering.
  Content of prompt: "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Tailor your response as if speaking to {audience}.
    
  {context}
    
  Question: {question}
  Helpful Answer: "
  
  Returns
  -------
  PromptTemplate(...)
      A prompt template with content as shown above in which audience, context, and question are filled in at a later step.
  """
  prompt_template = (
"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Tailor your response as if speaking to {audience}.

{context}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>{query}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
  )
  return PromptTemplate(
    input_variables = ["audience", "context", "question"],
    template = prompt_template
  )
  
  
  
def cons_prompt():
  """
  Prompt used for outlining consequences of a specific cyberattack.
  Content of prompt: "Start with a one sentence synopsis of the following context. Describe the ramifications of ignoring an attack as outlined in the context. Be specific about how these ramifications come about. Tailor your response as if speaking to {audience}.
    
  {context}
    
  Helpful Answer: "
  
  Returns
  -------
  PromptTemplate(...)
      A prompt template with content as shown above in which audience and context are filled in at a later step.
  """
  prompt_template = (
"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>Start with a one sentence synopsis of the following context. Describe the ramifications of ignoring an attack as outlined in the context. Be specific about how these ramifications come about. Tailor your response as if speaking to {audience}.

{context}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
  )
  
  return PromptTemplate(
    input_variables = ["audience", "context"],
    template = prompt_template
  )



def sc_prompt():
  """
  Prompt used for creating practice attack scenarios centered around a specific cyberattack.
  Content of prompt: "Use the following pieces of context to describe a potential attack that could be faced within the power industry. Be specific in who the attacker is and what they are doing so that the security team can discuss an appropriate response. Tailor your response as if speaking to {audience}.
    
  {context}
  
  Helpful Answer: "
  
  Returns
  -------
  PromptTemplate(...)
      A prompt template with content as shown above in which audience and context are filled in at a later step.
  """
  prompt_template = (
"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>Use the following pieces of context to create a realistic and descriptive narrative of this attack occuring. Be specific in who the attacker is and what they are doing. Tailor your response as if speaking to {audience}.

{context}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
  )
  
  return PromptTemplate(
    input_variables = ["audience", "context"],
    template = prompt_template
  )

  

# FORMATTING
def get_llm_response(output):
  return snip(
    output = output, 
    starting_characters = "<|start_header_id|>assistant<|end_header_id|>"
  ) 





"""
RETRIEVER
"""
def create_retriever():
  """
  Creates the document retriever. Documents can be filtered by metadata. Documents are represented by vectors, and the vectors that are the closest match to the question asked to the RAG are retrieved.
      
  Returns
  -------
  SelfQueryRetriever
      Can retrieve relevant documents when invoked.
  """
  db, translator = access_chromaDB()

  retriever = (
    SelfQueryRetriever(
    query_constructor = structured_request_chain(),
    vectorstore = db,
    structured_query_translator = translator
    )
    | RunnableLambda(format_docs)
    | RunnableLambda(let_me_see)
  )
  
  return retriever



def format_docs(docs):
  """
  Returns a String of document page contents separated by two new lines.
  
  Parameters
  ----------
  docs: list
      A list of LangChain documents.
  
  Returns
  -------
  A String of only the page contents of those documents - no metadata.
  """
  return "\n\n".join(doc.page_content for doc in docs)

  
  
# DATABASE
def access_chromaDB():
  """
  Access a Chroma Vector Database.
    
  Parameters
  ----------
  dbLoc: String
      The path to the database. Relative or absolute.
  dbEmbeddings: HuggingFaceEmbeddings
      Translates documents from numbers that computers understand back into words that humans understand.
      
  Returns
  -------
  db
      The Chroma DB.
  translator 
      The ChromaTranslator() (used in json_query_generator > create_retriever()) that translates a filter statement from JSON format into an appropriate filter statement for Chroma.
  """
  db = Chroma(
    persist_directory = dbLoc, 
    embedding_function = db_embeddings(),
    collection_name = "CVEs"
  )
  
  translator = ChromaTranslator()
  
  return db, translator



def db_embeddings():
  """
  Returns the embeddings used to translate documents in the form of vectors back to documents that contain words. Must match the embeddings that were used to create the database (see createDatabase.py).
  
  Parameters
  ----------
  embeddingModel: String
      Used to ID the different embedding models on HuggingFace.
  
  Returns
  -------
  dbEmbeddings
      The function that can translate words to vectors and vectors to words.
  """
  dbEmbeddings = HuggingFaceEmbeddings(
      model_name = embeddingModel,
      model_kwargs = {
        "device" : "cuda"
      }
  )
  
  return dbEmbeddings
  
  

# STRUCTURED REQUEST
def structured_request_chain():
  """
  When invoked, turns a question into a JSON structured request.
  
  Parameters
  ----------
  llm: HuggingFacePipeline
      The LLM/AI used to generate the structured request
  
  Returns
  -------
  json_query_constructor
      A chain that will generate a structured request when invoked with a question.
  """
  output_parser = StructuredQueryOutputParser.from_components() # the llm will return a String that looks like JSON - this will identify the query and the filter from the String
  
  structured_request_chain = (
    RunnableLambda(cve_id) 
    | RunnableLambda(create_filter) 
    | RunnableLambda(structured_request) 
    | output_parser # and the JSON string is parsed to determine the query and filter
  )
  
  return structured_request_chain



def cve_id(input):
  return re.findall(r"CVE-[0-9]{4}-[0-9]{4}", input.get("query"))



def create_filter(list_of_matches):
  if len(list_of_matches) < 1:
    return "NO_FILTER"
  
  filters = []
  for match in list_of_matches:
    cveID = match
    filters.append(f"eq(\\\"cveId\\\", \\\"{cveID}\\\")")

  if len(filters) == 1:
    return (f"{filters[0]}")
  
  return (f"or({', '.join(filters)})")



def structured_request(filters):
  return (
f"""{{
  "query": "",
  "filter": "{filters}"
}}"""
  )
  
  
 


# OUTPUT PROCESSING FUNCTIONS
def snip(
  output, # llm output from which to snip
  starting_characters, # a string that describes the sequence of characters to start the snip
  include_start = False, # should starting characters be included?
  ending_characters = None, # a string that describes the sequence of characters to end the snip
  include_end = False # should the ending characters be included?
):
  """
  Extracts the desired pieces of an LLM/AI's output.
  
  Parameters
  ----------
  output: String
      The output of an LLM/AI.
  starting_characters: String
      The first few characters to look for to define the starting point of the extracted text. Should typically be the end of the prompt provided to the LLM/AI.
  include_start: boolean, default False
      Whether or not to include the start_characters in the extraction.
  ending_characters: String, default None
      The last few characters to look for to define the ending point of the extracted text. Extraction will go to the end of the output if no ending_characters are provided.
  include_end: boolean, default False
      Whether or not to include the end_characters in the extraction.
  
  Returns
  -------
  output[starting_index:ending_index]
      The desired snippet of the output.
  """
  starting_index = output.rfind( # find the beginning of your excerpt
    starting_characters
  )
  
  ending_index = -1 # unless specified, snip will go to end of output
  
  if (starting_index == -1): # beginning of excerpt can't be found
    starting_index = 0 
  
  after_starting_characters = starting_index + len(starting_characters) # index of end of starting characters
  
  if (not include_start): # if not including starting characters
    starting_index = after_starting_characters # set starting index to end of starting_characters sequence
  
  if (ending_characters != None): # if ending_characters are specified
    ending_index = output.find( # find them
      ending_characters, 
      after_starting_characters
    )

  if (ending_index == -1): # if ending_characters can't be found or weren't specified
    return output[starting_index:] # return starting index to end of output
    
  if (include_end): # if ending characters should be included
    ending_index += len(ending_characters) #change ending_index to after ending_characters
    
  return output[starting_index:ending_index] # return snip



def get_json_query(output):
  """
  Returns
  -------
  snip(...)
      Extraction of text starting at "{" (inclusive), ending at "}" (inclusive). See snip() above.
  """
  return snip(
    output = output,
    starting_characters = "{",
    include_start = True,
    ending_characters = "}",
    include_end = True
  )





# TROUBLESHOOTING
def let_me_see(text):
  print(f"{text=}\n\n")
  return text





# RUN
if __name__ == "__main__":
    main()