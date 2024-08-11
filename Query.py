# IMPORTS
## LLM
from langchain_community.llms import HuggingFacePipeline

## RETRIEVER
from torch import cuda, bfloat16
import transformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains.query_constructor.base import (
  AttributeInfo,
  StructuredQueryOutputParser,
  get_query_constructor_prompt
)
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_core.runnables import RunnableLambda




# MODIFIABLE VARIABLES
database_directory = "./ChromaDB"
embedding_model = "mixedbread-ai/mxbai-embed-large-v1"
hf_auth = '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'
model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'





# CODE

## LLM
def create_llm(model_id, hf_auth, device):
  # Quantization configuration using bitsandbytes library
  # setting it to load a large model w/ less GPU memory
  bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
  )
  
  # Initialize Hugging face items w/ access token
  model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    token=hf_auth
  )
  
  model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_auth
  )
  
  # Enable evaluation mode - allows model inference
  model.eval()

  # Prints if it makes it this far
  print(f"Model loaded on {device}")
  
  # Initialize the tokenizer
  tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth,
    padding_side="right"
  )

  # Add pad token as eos token
  tokenizer.pad_token_id = tokenizer.eos_token_id
  
  # Add in the pad_token_ids... fixes eos_token_id error
  model.generation_config.pad_token_id = tokenizer.pad_token_id
  
  # Initialize the terminators
  terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  # Initialize the generator parameter
  generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    eos_token_id=terminators,
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=1024,  # max number of tokens to generate in the output
    repetition_penalty=1.1,  # without this output begins repeating
  )
  
  # Implement Hugging Face Pipeline in LangChain
  llm = HuggingFacePipeline(pipeline=generate_text)
  
  return llm
  
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

llm = create_llm(model_id, hf_auth, device)

## RETRIEVER

embedding_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(
  model_name = embedding_model, 
  model_kwargs = embedding_kwargs
)

db = Chroma(
  persist_directory = database_directory,
  embedding_function = embeddings
)


prompt_template = """
Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{{
    "query": string \ text string to compare to document contents
    "filter": string \ logical condition statement for filtering documents
}}
```

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


<< Example 1. >>
User Query:
Does CVE-2024-0008 affect multiple versions of the software?

Structured Request:
```json
{{
    "query": "versions",
    "filter": "eq(\\"cveId\\", \\"CVE-2024-0008\\")"
}}
```


<< Example 2. >>
User Query:
In CVE-2024-0015, is the vulnerability is due to improper input validation in the DreamService.java file.

Structured Request:
```json
{{
    "query": "description",
    "filter": "eq(\\"cveId\\", \\"CVE-2024-0015\\")"
}}
```


<< Final >>
User Query:
{query}

Structured Request:
"""

prompt = PromptTemplate(
  input_variables = ["query"],
  template = prompt_template
)

def trim_output(output):
  open_index = output.rfind("{")
  close_index = output.rfind("}")
  return output[open_index: close_index+1]

#print(llm.invoke(prompt.format(query = "Does the vulnerability described in CVE-2024-0011 allow for the execution of arbitrary code on the affected system?"))))

output_parser = StructuredQueryOutputParser.from_components()
query_constructor = prompt | llm | RunnableLambda(trim_output) | output_parser

retriever = SelfQueryRetriever(
  query_constructor=query_constructor,
  vectorstore=db,
  structured_query_translator=ChromaTranslator(),
  verbose = True
)

print(retriever.invoke("Does the vulnerability described in CVE-2024-0011 allow for the execution of arbitrary code on the affected system?"))