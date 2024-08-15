import sys
sys.path.append("..") # allows import of files in parent directory (I need this for HFAuth right below because I am running this file from the SLURM file, RAG.sh)

"""
IMPORTS
"""
# VARIABLES
from PersonalTokens import HFAuth

# FULL SCRIPT
import torch

# RAG
from langchain_core.runnables import RunnableLambda # also in GENERATORS
from langchain_core.prompts.chat import MessagesPlaceholder

# LLM
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

# RETRIEVER
## create_retriever()
from langchain.retrievers.self_query.base import SelfQueryRetriever
## access_chromaDB()
from langchain_chroma import Chroma
from langchain_community.query_constructors.chroma import ChromaTranslator
## db_embeddings()
from langchain_huggingface import HuggingFaceEmbeddings

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
hfAuth = HFAuth # old version of this code had my actual auth token here - that token has been expired and the new token is hidden using .gitignore

# enter model that generator is based on (as seen on Hugging Face)
llmModel = 'meta-llama/Meta-Llama-3-8B'

# enter model that embeds and decodes documents in database (as seen on Hugging Face)
embeddingModel = "mixedbread-ai/mxbai-embed-large-v1"

# enter location of vector database (for me, it is relation to my SLURM file RAG.sh)
dbLoc = "../ChromaDB"





"""
SCRIPT
"""
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
  
  rag(
    question,
    llm,
    retriever
  )
  
  rag(
    question,
    llm,
    retriever,
    task = "sc"
  )

  rag(
    question,
    llm,
    retriever,
    task = "cons"
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
  standalone_question = question

  if (chat_history):
    standalone_question = standalone_question_generator(llm).invoke( # standalone_question_generator() is under GENERATORS
      {
        "chat_history": chat_history,
        "question": question
      }
    )
  
  context = format_docs(retriever.invoke(standalone_question)) # the documents that the LLM will use to answer the question
  
  if (task == "cons"):
    partial_prompt = cons_prompt() # cons_prompt() is under PROMPTS
    printTask = "Consequences: "
    
  elif (task == "sc"):
    partial_prompt = sc_prompt() # sc_prompt() is under PROMPTS
    printTask = "Scenario Creator: "
  
  else:
    if (task != "qa"):
      print("""
        Task not recognized, defaulting to question-answering. Task must be one of the following:
        
        qa (question-answering): answers the given question
        cons (consequences): outlines consequences of the specified cyberattack
        sc (scenario creator): describes a practice attack scenario centered around the specified cyberattack
      """)
      
    prompt = qa_prompt() # qa_prompts() is under PROMPTS
    printTask = "Question Answering: "
    partial_prompt = prompt.partial(
      question = standalone_question
    ) 
  
  generator_input = partial_prompt.format(
    audience = audience,
    context = context
  )

  print(printTask + question)

  chat_history.append(("system", generator_input))
  
  generator_output = (
    llm
    | RunnableLambda(get_helpful_answer)
  ).invoke(generator_input)

  chat_history.append(("ai", generator_output))
  
  print(generator_output)

  return chat_history





# LLM
def create_llm(llmModel):
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

  model = AutoModelForCausalLM.from_pretrained(
    llmModel,
    temperature = 0.1, # how random outputs are; 0.1 is not very random
    do_sample = True,
    device_map = "auto", # automatically allocates resources between GPUs (and CPUs if necessary)
    quantization_config = bnb_config, # quantization: more efficient computing, less memory
    token = hfAuth # login to HuggingFace for access if necessary (necessary for llama models)
  )

  tokenizer = AutoTokenizer.from_pretrained(
    llmModel,
    padding_side = "right",
    token = hfAuth # login to HuggingFace for access if necessary (necessary for llama models)
  )

  pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 1000,
    repetition_penalty = 1.2, # prevents some repetition in model responses
    token = hfAuth # login to HuggingFace for access if necessary (necessary for llama models)
  )
  
  llm = HuggingFacePipeline(
    pipeline = pipe
  )
  
  return llm



# RETRIEVER
def create_retriever(llm, db, translator):
  """
  Creates the document retriever. Documents can be filtered by metadata. Documents are represented by vectors, and the vectors that are the closest match to the question asked to the RAG are retrieved.
    
  Parameters
  ----------
  llm: HuggingFacePipeline
      The llm used to generate portions of the RAG. 
      Within the retriever, the llm detects a filter statement within the question and produces an appropriate json query to filter the documents.
  db: Chroma database 
      The vector database that holds the vector representations of documents that the RAG can use as context.
      Other vector databases could work, but I do not have the code for it yet.
  translator: ChromaTranslator
      Translates the json query created by the llm to use in filtering the documents.
      Must match the vector database.
      
  Returns
  -------
  SelfQueryRetriever
      Can retrieve relevant documents when invoked.
  """
  retriever = SelfQueryRetriever(
    query_constructor = json_query_generator(llm),
    vectorstore = db,
    structured_query_translator = translator,
    verbose = True
  )
  
  return retriever
  
  

def access_chromaDB(dbLoc, dbEmbeddings):
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
    embedding_function = dbEmbeddings
  )
  
  translator = ChromaTranslator()
  
  return db, translator


def db_embeddings(embeddingModel):
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
      model_name = embeddingModel
  )
  
  return dbEmbeddings
  
  
  
  
  
# GENERATORS
def standalone_question_generator(llm):
  """
  When invoked, uses that chat history and the question to create a standalone question - no ambiguous pronouns, etc.
  
  Parameters
  ----------
  llm: HuggingFacePipeline
      The LLM/AI used to generate the standalone question.
  chat_history: list
      The previous prompt (including the question/standalone question) used to generate an answer and the answer itself.
  
  Returns
  -------
  standalone_question_generator
      A chain that will generate a standalone question when invoked with a question and chat history.
  """
  standalone_question_generator = (
    standalone_question_prompt() # original question and chat history inserted into the prompt...
    | llm # which prompts the llm to create a standalone question...
    | RunnableLambda(get_standalone_question) # and trim off the prompt
  )
  
  return standalone_question_generator



def json_query_generator(llm):
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
  
  json_query_generator = (
    query_constructor_prompt() # see query_constructor_prompt() under PROMPTS
    | llm # the llm is prompted to create a JSON string structured as requested...
    | RunnableLambda(let_me_see)
    | RunnableLambda(get_final) # the prompt is trimmed off...
    | RunnableLambda(get_json_query)
    | output_parser # and the JSON string is parsed to determine the query and filter
  )
  
  return json_query_generator
    




# PROMPTS
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
  prompt_template = """
  Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Tailor your response as if speaking to {audience}.
    
  {context}
    
  Question: {question}
  Helpful Answer: 
  """
  
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
  prompt_template = """
  Start with a one sentence synopsis of the following context. Describe the ramifications of ignoring an attack as outlined in the context. Be specific about how these ramifications come about. Tailor your response as if speaking to {audience}.
    
  {context}
    
  Helpful Answer: 
  """
  
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
  prompt_template = """
  Use the following pieces of context to describe a potential attack that could be faced within the power industry. Be specific in who the attacker is and what they are doing so that the security team can discuss an appropriate response. Tailor your response as if speaking to {audience}.
    
  {context}
  
  Helpful Answer: 
  """
  
  return PromptTemplate(
    input_variables = ["audience", "context"],
    template = prompt_template
  )
  
  
  
def standalone_question_prompt():
  """
  Prompt used for turning an ambiguous question into a standalone question using chat history.
  Content of prompt: "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
  
  Chat History:
  {chat_history}
  
  Follow Up Input: {question}
  Standalone question: "
  
  Returns
  -------
  PromptTemplate(...)
      A prompt template with content as shown above in which chat_history and question are filled in at a later step.
  """
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
  """
  A helper function to create examples to include in the query_constructor_prompt().
  
  Parameters
  ----------
  i: int
      Should start at 1 and increase for every example provided.
  question: String
      An example question that could be asked of the RAG.
  query: String
      Words that WOULD be found in the filtered documents that could be used to answer the question.
  filter_statment: String
      A logical and comparative statement that describes a filter found within the question. If there is no filter, this should be NO_FILTER.
  
  Returns
  -------
  A String with the following format: "
  
    << Example {i}. >>
    User Query:
    {question}
    
    Structured Request:
    ```json
    {{{{
        "query": "{query}",
        "filter": "{filter_statement}"
    }}}}
    ```
  "
  """
  example = (f"""
    
    << Example {i}. >>
    User Query:
    {question}
    
    Structured Request:
    ```json
    {{{{
        "query": "{query}",
        "filter": "{filter_statement}"
    }}}}
    ```
  """)
  return example
   
   
   
def query_constructor_prompt():
  """
  Prompt for turning a question into a structured request to be used for the SelfQueryingRetriever(), which can filter documents and THEN find relevant info. In the prompt detailed below, query_content, data_source, and examples are described by variables in the function.
  Content of prompt: "Your goal is to structure the user's query to match the request schema provided below.
    
    << Structured Request Schema >>
    When responding use a markdown code snippet with a JSON object formatted in the following schema:
    
    {query_format}
    
    The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.
    
    A logical condition statement is composed of one or more comparison and logical operation statements.
    
    A comparison statement takes the form: `comp(attr, val)`:
    - `comp` {comparators}: comparator
    - `attr` (string):  name of attribute to apply the comparison to
    - `val` (string): is the comparison value
    
    A logical operation statement takes the form `op(statement1, statement2, ...)`:
    - `op` {logical_operators}: logical operator
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
    
    Structured Request:"
    
  Returns
  -------
  PromptTemplate(...)
      A prompt template with content as shown above in which question is filled in at a later step.
  """
  query_format = """
    ```json
    {{
        "query": string \ text string to compare to document contents
        "filter": string \ logical condition statement for filtering documents
    }}
    ```
  """

  comparators = "($eq | $ne | $gt | $gte | $lt | $lte)"

  logical_operators = "(and | or)"

  data_source = """
    ```json
    {{
        "content": "Cybersecurity vulnerabilities",
        "attributes": {{
          "cveId": {{
              "description": "A unique alphanumeric identifier for the vulnerability. Format: CVE-YYYY-NNNN",
              "type": "string"
          }}
        }}
    }}
    ```
  """
  
  examples = query_constructor_example(
    i = 1, 
    question = "Does CVE-2024-0008 affect multiple versions of the software?",
    query = "versions",
    filter_statement = "eq(\\\"cveId\\\", \\\"CVE-2024-0008\\\")"
  ) + query_constructor_example(
    i = 2, 
    question = "In CVE-2024-0015, is the vulnerability due to improper input validation in the DreamService.java file?",
    query = "description",
    filter_statement = "eq(\\\"cveId\\\", \\\"CVE-2024-0015\\\")"
  ) + query_constructor_example(
    i = 3,
    question = "Does the vulnerability described by CVE-2024-0015 affect Android versions 11, 12, 12L, and 13 with a default status of unaffected?",
    query = "versions",
    filter_statement = "eq(\\\"cveId\\\", \\\"CVE-2024-0015\\\")"
  ) + query_constructor_example(
    i = 4,
    question = "Does the vulnerability described by CVE-2024-0009 have a solution that I can implement?",
    query = "solutions",
    filter_statement = "eq(\\\"cveId\\\", \\\"CVE-2024-0009\\\")"
  )
  
  prompt_template = (f"""
    Your goal is to structure the user's query to match the request schema provided below.
    
    << Structured Request Schema >>
    When responding use a markdown code snippet with a JSON object formatted in the following schema:
    
    {query_format}
    
    The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.
    
    A logical condition statement is composed of one or more comparison and logical operation statements.
    
    A comparison statement takes the form: `comp(attr, val)`:
    - `comp` {comparators}: comparator
    - `attr` (string):  name of attribute to apply the comparison to
    - `val` (string): is the comparison value
    
    A logical operation statement takes the form `op(statement1, statement2, ...)`:
    - `op` {logical_operators}: logical operator
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
    {{query}}
    
    Structured Request:
  """)
  
  return PromptTemplate(
    input_variables = ["query"],
    template = prompt_template
  )





# OUTPUT PROCESSING FUNCTIONS
def list_to_string(list):
  """
  Returns a String composed of a list of Strings joined together by two new lines.

  Parameters
  ----------
  list: list
      The list of strings to be joined together.
    
  Returns
  -------
  "\\n\\n".join(list)
      A String composed of the list of Strings put together.
  """
  return "\n\n".join(list)



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
  return list_to_string(doc.page_content for doc in docs)



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
  starting_index = output.find( # find the beginning of your excerpt
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



def get_standalone_question(output):
  """
  Returns
  -------
  snip(...)
      Extraction of text starting at "Standalone Question:" (not inclusive), ending at "?" (inclusive). See snip() above.
  """
  return snip(
    output = output,
    starting_characters = "Standalone Question:",
    ending_characters = "?",
    include_end = True
  )



def get_helpful_answer(output):
  """
  Returns
  -------
  snip(...)
      Extraction of text starting at "Helpful Answer:" (not inclusive), ending at "Final Answer:" (not inclusive). See snip() above.
  """
  return snip(
    output = output, 
    starting_characters = "Helpful Answer:",
    ending_characters = "Final Answer:"
  ) 



def get_final(output):
  """
  Returns
  -------
  snip(...)
      Extraction of text starting at "<< Final >>" (not inclusive), ending at end of the original text. See snip() above.
  """
  return snip(
    output = output,
    starting_characters = "<< Final >>"
  )
  
  
  
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
  print(f"{text=}")
  return text





# RUN
if __name__ == "__main__":
    main()