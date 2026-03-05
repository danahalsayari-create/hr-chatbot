import pandas as pd
import streamlit as st
from core import question_to_sql, run_sql
from analytics import quick_analysis, sentiment_sample

#load and return tokenizer , model , device location
@st.cache_resource
def load_local_model(model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    device = torch.device("cpu")

    tok = AutoTokenizer.from_pretrained(model_name) 
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tok, model, device

# page configration 
st.set_page_config(page_title="HR Chatbot", layout="wide",initial_sidebar_state="expanded")
st.title("HR Chatbot")



# paths
DB_PATH = "database/hr.db"
#*
TABLE = "employees"
DATASET_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

#setting dic
settings = {
    "GROQ_API_KEY": st.secrets.get("GROQ_API_KEY"),
    "GROQ_MODEL": "llama-3.3-70b-versatile",
    "LOCAL_MODEL": "prem-research/prem-1B-SQL",
}
#sidebar
page = st.sidebar.radio("Page", ["Chat", "Quick Analysis", "Sentiment"])
model_mode = st.sidebar.selectbox("Model", ["API", "Local"])


show_sql = st.sidebar.checkbox("Show SQL query", value=False)

#set memory
st.session_state.setdefault("history", []) 

#memory to model 
def memory_in(max_messages: int = 8) -> str:
    previus_messages = st.session_state.history[-max_messages*2:]
    lines = []
    for t in previus_messages:
        who = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{who}: {t['content']}")
    return "\n".join(lines)

#quick analysis using pandas
if page == "Quick Analysis":
    st.subheader("Quick Analysis (Pandas)")
    st.json(quick_analysis(DATASET_PATH))

#sentiment page
elif page == "Sentiment":
    st.subheader("Sentiment (HF)")
    df = sentiment_sample(DATASET_PATH, n=200)
    st.write(df["sentiment"].value_counts())
    st.dataframe(df.head(20), width="stretch")

#chatbot page
else:
    st.subheader("Chat with HR dataset")

    #display history
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("table"):
                table = m["table"]
                df_view = pd.DataFrame(table["data"], columns=table.get("columns"))
                st.dataframe(df_view, width="stretch")

    q = st.chat_input("Ask me about HR dataset....")

    if q:
        #store message
        st.session_state.history.append({"role": "user", "content": q})
        
        with st.chat_message("user"):
            st.markdown(q)

    
        local_bundle = None
        if model_mode == "Local":
            local_bundle = load_local_model(settings["LOCAL_MODEL"])

        sql = question_to_sql(
            model_mode, settings, DB_PATH, TABLE,
            q, memory_in(),
            local_bundle=local_bundle
        )
        with st.chat_message("assistant"):
                if sql == "NOT_SQL":
                    ans = "I can answer questions about the HR dataset only."
                    st.markdown(ans)
                    df = None
                    
            #about dataset
                else:
                    if show_sql:
                        st.code(sql, language="sql")

                    rows, cols = run_sql(DB_PATH, sql)
                    df = pd.DataFrame(rows, columns=cols)

                    if len(df.columns) == 1 and len(df) == 1:
                        ans = f"The answer is: **{df.iloc[0,0]}**"
                        st.markdown(ans)
                    else:
                        ans = "The answer is:"
                        st.markdown(ans)
                        st.dataframe(df, width="stretch")


        
        msg = {"role": "assistant", "content": ans}
        if df is not None:
                msg["table"] = {
                    "data": df.to_dict(orient="records"),
                    "columns": list(df.columns),
                }
        st.session_state.history.append(msg)   
        
        

