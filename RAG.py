"""
IMPORTS
"""
# VARIABLES
from HfAuth import hfAuth

# FULL SCRIPT
import torch

# RAG
from langchain_core.runnables import RunnableLambda # also in GENERATORS

# LLM
import transformers
from transformers import AutoTokenizer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# RETRIEVER
## create_retriever()
from langchain.retrievers.self_query.base import SelfQueryRetriever
## access_chromaDB()
from langchain_chroma import Chroma
from langchain.retrievers.self_query.chroma import ChromaTranslator
## db_embeddings()
from langchain_community.embeddings import HuggingFaceEmbeddings

# GENERATORS
## standalone_question_generator()
from langchain_core.runnables.passthrough import RunnablePassthrough # also in PROMPTS.task_prompt()
## json_query_generator()
from langchain.chains.query_constructor.base import StructuredQueryOutputParser

# PROMPTS
from langchain_core.prompts import PromptTemplate

# OUTPUT PROCESSING FUNCTIONS
## no imports





"""
CHANGE AS NEEDED
"""
# enter Hugging Face auth code
hfAuth = hfAuth # old version of this code had my actual auth token here - that token has been expired and the new token is hidden using .gitignore

# enter model that generator is based on (as seen on Hugging Face)
llmModel = 'meta-llama/Meta-Llama-3-8B'

# enter model that embeds and decodes documents in database (as seen on Hugging Face)
embeddingModel = "mixedbread-ai/mxbai-embed-large-v1"

# enter location of vector database
dbLoc = "./ChromaDB"





"""
SCRIPT
"""
if __name__ == "__main__":
    main()

def main():
  llm = create_llm(llmModel) # initialize llm
  
  dbEmbeddings = db_embeddings(embeddingModel) # embedding function corresponding to transformer model
  
  db, translator = access_chromaDB( # access persistent chromaDB and appropriate translator
    dbLoc, 
    dbEmbeddings
  )
  
  retriever = create_retriever( # initialize retriever
    llm, 
    db, 
    translator
  )
  
  question = "Does the vulnerability described in CVE-2024-0011 allow for the execution of arbitrary code on the affected system?"
  
  chat_history = rag( # rag prints output and stores input and output in chat history
    question,
    llm,
    retriever
  )
  
  question = "Anything else to add?"
  
  chat_history = rag(
    question,
    llm,
    retriever,
    chat_history
  )
  




"""
FUNCTIONS
"""

# RAG
def rag(
  question,
  llm,
  retriever,
  chat_history = [],
  task = "qa",
  audience = "a general audience"
):
  if (len(chat_history) > 0):
    standalone_question = standalone_question_generator(llm, chat_history).invoke(question)
    question = standalone_question
    
  task_prompt = task_prompt(
    llm, 
    retriever,
    audience, 
    task
  ).invoke(question)
  
  rag = (
    llm
    | RunnableLambda(get_helpful_answer)
  )
  
  llm_response = rag.invoke(task_prompt)
  print(llm_response)
  
  chat_input = "Input: " + task_prompt
  chat_output = "Output: " + llm_response
  chat_history = [chat_input, chat_output]
  
  return chat_history





# LLM
def create_llm(llmModel):
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
  
  tokenizer = AutoTokenizer.from_pretrained(llmModel)
  
  llmPipe = transformers.pipeline(
    "text-generation", 
    model=llmModel, 
    model_kwargs={
      "torch_dtype": torch.bfloat16
    }, 
    device_map="auto",
    eos_token_id = [
      tokenizer.eos_token_id, 
      tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ],
    token = hfAuth
  )
  
  llm = HuggingFacePipeline(
      pipeline = generatorPipe,
      model_kwargs = {"temperature": 0.1}
  )
  
  return llm



# RETRIEVER
def create_retriever(llm, db, translator):
  retriever = SelfQueryRetriever(
    query_constructor = json_query_generator(llm),
    vectorstore = db,
    structured_query_translator = translator,
    verbose = True
  )
  
  return retriever
  
  

def access_chromaDB(dbLoc, dbEmbeddings):
  db = Chroma(
    persist_directory = dbLoc,
    embedding_function = dbEmbeddings
  )
  
  translator = ChromaTranslator()
  
  return db, translator


def db_embeddings(embeddingModel):
  dbEmbeddings = HuggingFaceEmbeddings(
      model_name = embeddingModel,
      model_kwargs = {
          "device": "cuda"
      }
  )
  
  
  
  
  
# GENERATORS
def standalone_question_generator(llm, chat_history):
  standalone_question_generator = (
    {"chat_history" : chat_history, "question" : RunnablePassthrough()}
    | standalone_question_prompt()
    | RunnableLambda(get_standalone_question)
  )
  
  return standalone_question_generator



def json_query_generator(llm):
  output_parser = StructuredQueryOutputParser.from_components()
  
  json_query_generator = (
    query_constructor_prompt()
    | llm 
    | RunnableLambda(get_final) 
    | output_parser
  )
  
  return json_query_generator
  
  
  


# PROMPTS
def task_prompt(
  llm, 
  retriever,
  audience, 
  task
):
  variables = {
      "audience" : audience,
      "context" : retriever | format_docs
    }
  
  if (task == "cons"):
    prompt = cons_prompt()
  
  elif (task == "sc"):
    prompt = sc_prompt()
    
  else:
    variables["question"] = RunnablePassthrough()
    prompt = qa_prompt()
    
    if (task != "qa"):
      print("Task not recognized. Defaulting to question-answering. Task should be one of the following: \n\nqa (for question-answering), \ncons (to outline consequences), or \nsc (for scenario creation).")

  task_prompt = (
    variables
    | prompt
    
  return task_prompt
    


def qa_prompt():
  prompt_template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Tailor your response as if speaking to {audience}.\n\n{context}\n\nQuestion: {question}\nHelpful Answer: 
  """
  
  return PromptTemplate(
    input_variables = ["audience", "context", "question"],
    template = prompt_template
  )
  
  
  
def cons_prompt():
  prompt_template = """
    Start with a one sentence synopsis of the following context. For cybersecurity within the power industry, describe the ramifications of an attack as outlined in the context being ignored. Be specific about how these ramifications come about. Tailor your response as if speaking to {audience}.\n\n{context}\n\nHelpful Answer: 
  """
  
  return PromptTemplate(
    input_variables = ["audience", "context"],
    template = prompt_template
  )



def sc_prompt():
  prompt_template = """
    Use the following pieces of context to describe a potential attack that could be faced within the power industry. Be specific in who the attacker is and what they are doing so that the security team can discuss an appropriate response. Tailor your response as if speaking to {audience}.\n\n{context}\n\nHelpful Answer: 
  """
  
  return PromptTemplate(
    input_variables = ["audience", "context"],
    template = prompt_template
  )
  
  
  
def standalone_question_prompt():
  prompt_template = """
  Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
  
  Chat History:
  {chat_history}
  
  Follow Up Input: {question}
  Standalone question:
  """
  
  return PromptTemplate(
    input_variables = ["chat_history", "question"],
    template = prompt_template
  )



def query_constructor_example(i, question, query, filter_statement):
  example = f"""
    
    << Example {i}. >>
    User Query:
    {question}
    
    Structured Request:
    ```json
    {
        "query": "{query}",
        "filter": "{filter_statement}"
    }
    ```
  """
  return example
   
   
   
def query_constructor_prompt():

  query_format = """
    ```json
    {
        "query": string \ text string to compare to document contents
        "filter": string \ logical condition statement for filtering documents
    }
    ```
  """
  data_source : """
    ```json
    {
        "content": "Cybersecurity vulnerabilities",
        "attributes": {
          "cveId": {
              "description": "A unique alphanumeric identifier for the vulnerability. Format: CVE-YYYY-NNNN",
              "type": "string"
          }
        }
    }
    ```
  """
  
  examples = query_constructor_example(
    i = 1, 
    user_query = "Does CVE-2024-0008 affect multiple versions of the software?",
    query = "versions",
    filter_statement = "eq(\\\"cveId\\\", \\\"CVE-2024-0008\\\")"
  ) + query_constructor_example(
    i = 2, 
    user_query = "In CVE-2024-0015, is the vulnerability due to improper input validation in the DreamService.java file?",
    query = "description",
    filter_statement = "eq(\\\"cveId\\\", \\\"CVE-2024-0015\\\")"
  )
  
  prompt_template = f"""
    Your goal is to structure the user's query to match the request schema provided below.
    
    << Structured Request Schema >>
    When responding use a markdown code snippet with a JSON object formatted in the following schema:
    
    {query_format}
    
    The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.
    
    A logical condition statement is composed of one or more comparison and logical operation statements.
    
    A comparison statement takes the form: `comp(attr, val)`:
    - `comp` ($eq | $ne | $gt | $gte | $lt | $lte): comparator
    - `attr` (string):  name of attribute to apply the comparison to
    - `val` (string): is the comparison value
    
    A logical operation statement takes the form `op(statement1, statement2, ...)`:
    - `op` (and | or | not): logical operator
    - `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to
    
    Make sure that you only use the comparators and logical operators listed above and no others.
    Make sure that filters only refer to attributes that exist in the data source.
    Make sure that filters only use the attributed names with its function names if there are functions applied on them.
    Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
    Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
    Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.
    
    << Data Source >>
    {data_source}
      
    
    {examples}
    
    
    << Final >>
    User Query:
    {{question}}
    
    Structured Request:
  """
  
  return PromptTemplate(
    input_variables = ["question"],
    template = prompt_template
  )





# OUTPUT PROCESSING FUNCTIONS
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def snip(
  output, # llm output from which to snip
  starting_characters, # a string that describes the sequence of characters to start the snip
  include_start = False, # should starting characters be included?
  ending_characters = null, # a string that describes the sequence of characters to end the snip
  include_end = False # should the ending characters be included?
):
  starting_index = output.find( # find the beginning of your excerpt
    value = starting_characters
  )
  
  ending_index = -1 # unless specified, snip will go to end of output
  
  if (starting_index == -1): # beginning of excerpt can't be found
    return "Expected characters are not present in output."
  
  after_starting_characters = starting_index + len(starting_characters) # index of end of starting characters
  
  if (not include_start): # if not including starting characters
    starting_index = after_starting_characters # set starting index to end of starting_characters sequence
  
  if (ending_characters != null): # if ending_characters are specified
    ending_index = output.find( # find them
      value = ending_characters, 
      start = after_starting_characters
    )
    
  if (ending_index == -1): # if ending_characters can't be found or weren't specified
    return output[starting_index:] # return starting index to end of output
    
  if (include_end): # if ending characters should be included
    ending_index += len(ending_characters) #change ending_index to after ending_characters
    
  return output[starting_index:ending_index] # return snip



def get_standalone_question(output):
  return snip(
    output = output,
    starting_characters = "Standalone Question:",
    ending_characters = "?",
    include_end = True



def get_helpful_answer(output):
  return snip(
    output = output, 
    starting_characters = "Helpful Answer:",
    ending_characters = "Final Answer:"
  ) 



def get_final(output):
  return snip(
    output = output,
    starting_characters = "<< Final >>"
  )
  
  
  
def get_json_query(output):
  return snip(
    output = output,
    starting_characters = "{",
    include_start = True,
    ending_characters = "}",
    include_end = True
  )