import re
import sqlite3

# question to sql
def question_to_sql( model_mode: str, settings: dict,db_path: str,table: str,
                     question: str, history_text: str, local_bundle=None) -> str:
    cols = table_columns(db_path, table)
    prompt = build_system_prompt(table, cols)

    userQ = question.strip()

    if model_mode == "API" and history_text:
        userQ = f"Conversation so far:\n{history_text}\n\nNew question:\n{userQ}"
        
    #API mode 
    if model_mode == "API":
        api_key = settings.get("GROQ_API_KEY")
        model_name = settings.get("GROQ_MODEL")
        if not api_key:
            return "there is no api key"

        raw = API_sql_generator(api_key, model_name, prompt, userQ) or ""
        sql = extract_sql_query(raw)

        if sql == "NOT_SQL":
            return "NOT_SQL"
        return sql if sql.endswith(";") else sql + ";"

    # Local mode 
    if local_bundle is None:
        return "there is no local model"

    tok, model, device = local_bundle
    raw = local_generate(tok, model, device, prompt, userQ, max_new_tokens=80)
    print("RAW LOCAL:\n", raw)

    sql = extract_sql_query(raw)
    if sql == "NOT_SQL":
        return "NOT_SQL"
    return sql if sql.endswith(";") else sql + ";"

#dataset columns
def table_columns(db_path: str, table: str) -> list[str]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute(f"PRAGMA table_info({table});").fetchall()
    return [r[1] for r in rows]
#  model prompt 
def build_system_prompt(table: str, cols: list[str]) -> str:
    col_lines = ", ".join(cols)

    return f"""
        You are a text-to-SQL generator for SQLite.

        Return ONLY ONE LINE.
        Either:
        1) A single valid SQLite SELECT query that starts with SELECT and ends with ;
        OR
        2) Exactly: NOT_SQL

        Table: {table}
        Columns: {col_lines}

        Examples:
        Q: how many employees are there
        A: SELECT COUNT(*) AS total_employees FROM employees;

        Q: list 5 employees
        A: SELECT EmployeeNumber, Age, Department FROM employees LIMIT 5;

        Q: hello
        A: NOT_SQL
        """.strip()
# API question to sql
def API_sql_generator(api_key: str, model: str, prompt: str, user: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    r = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":prompt},{"role":"user","content":user}],
    )
    out = r.choices[0].message.content
    return (out or "").strip()

# local model question to sql
def local_generate(tok, model, device, system_prompt: str, user_prompt: str, max_new_tokens: int = 120) -> str:
    import torch

    merged = system_prompt + "\n\nQ: " + user_prompt + "\nSQL:"
    enc = tok(merged, return_tensors="pt", truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
        gen_ids = out[0][enc["input_ids"].shape[1]:] 
        return tok.decode(gen_ids, skip_special_tokens=True).strip()
    
# extract query form model ans
def extract_sql_query(text: str) -> str:
    t = (text or "").strip()

    t = re.sub(r"^```sql\s*", "", t, flags=re.I)
    t = re.sub(r"^```\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    m_sql = re.search(r"\bSQL\s*:\s*(.*)", t, flags=re.I | re.S)
    candidate = m_sql.group(1).strip() if m_sql else t

    m = re.search(r"\bSELECT\b.*?(;|\Z)", candidate, flags=re.I | re.S)
    if m:
        sql = m.group(0).strip()
        
        if ";" in sql:
            sql = sql.split(";", 1)[0].strip() + ";"
        else:
            sql = sql + ";"
        return sql

    if re.search(r"\bNOT_SQL\b", candidate, flags=re.I):
        return "NOT_SQL"

    return "NOT_SQL"
# takes sql query and return result from db
def run_sql(db_path: str, sql: str, params=()):
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError(f"SQL is invalid: {sql!r}")

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description] 
        rows = cur.fetchall()
    return rows, cols





