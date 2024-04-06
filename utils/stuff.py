
def fix_query(query):
    r"""
    Remove postgresql syntax symbols from
    a generated query
    """
    regex = "::\w*"
    fixed = re.sub(regex, "", query)
    return fixed

def generate_sql_query(inp):
    r"""
    Translate a natural language query 
    to sql.
    """
    model_name_or_path = "TheBloke/sqlcoder2-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="gptq-8bit-128g-actorder_True",
                                             )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    prompt_template = generate_prompt(inp)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1
    )
    sql_pipeline = HuggingFacePipeline(pipeline=pipe)
    out = sql_pipeline(prompt_template)
    out = fix_query(out)
    return out 

def generate_prompt(question, prompt_file="prompt.md", metadata_file="metadata.json"):
    r"""
    Generate a prompt containing the minimal 
    database schema.
    """
    with open(prompt_file, "r") as f:
        prompt = f.read()
    table_metadata_string = generate_slim(metadata_file, question)
    prompt = prompt.format(
        user_question=question, table_metadata_string=table_metadata_string
    )
    return prompt