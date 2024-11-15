# LIBRARIES ----​
import streamlit as st
import pandas as pd
import sqlalchemy as sql
import os
import plotly.express as px

from langchain import OpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# CONNECT TO DATABASE ----
sql_engine = sql.create_engine("sqlite:///database/leads_scored.db")
conn = sql_engine.connect()
metadata = sql.MetaData()
metadata.reflect(bind=sql_engine)
os.environ["OPENAI_API_KEY"] = ""

# LLMS ----
llm = OpenAI(
    temperature=0,
    max_tokens=256,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
db = SQLDatabase(engine=sql_engine, metadata=metadata)
sql_chain_with_steps = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True,
                                                 return_intermediate_steps=True)


# FUNCTIONS ----
def get_text():
    return st.text_input("You: ", "Hello, how can I help?", key="input")


def generate_chat_response(prompt):
    # Detect if any chart is requested
    is_chart_requested = any(keyword in prompt.lower() for keyword in ['chart', 'plot', 'graph'])

    # Detect the specific type of chart requested
    prompt_lower = prompt.lower()
    is_bar_chart = "bar chart" in prompt_lower
    is_line_chart = "line chart" in prompt_lower
    is_scatter_chart = "scatter plot" in prompt_lower
    is_donut_chart = "pie chart" in prompt_lower

    # Generate SQL and response
    res = sql_chain_with_steps(prompt)
    chatbot_response = res['result']
    chatbot_sql_code = res['intermediate_steps'][1]
    chatbot_sql_query_df = pd.read_sql_query(sql.text(chatbot_sql_code), conn)

    # Generate appropriate chart based on the request if a chart is requested
    chart = None
    if is_chart_requested:
        if is_bar_chart:
            chart = px.bar(chatbot_sql_query_df, x=chatbot_sql_query_df.columns[1], y=chatbot_sql_query_df.columns[0],
                           orientation='h')
        elif is_line_chart:
            chart = px.line(chatbot_sql_query_df, x=chatbot_sql_query_df.columns[1], y=chatbot_sql_query_df.columns[0])
        elif is_scatter_chart:
            chart = px.scatter(chatbot_sql_query_df, x=chatbot_sql_query_df.columns[1],
                               y=chatbot_sql_query_df.columns[0])
        elif is_donut_chart:
            chart = px.pie(chatbot_sql_query_df, names=chatbot_sql_query_df.columns[0],
                           values=chatbot_sql_query_df.columns[1])
        chart.update_layout(autosize=False,width=1000,height=700)

    return chatbot_response, chatbot_sql_query_df, chatbot_sql_code, chart


# APP ----
st.set_page_config(layout="wide")
st.title("Lead Scoring Analyzer")
col1, col2 = st.columns(2)

with col1:
    st.header("Leads Scoring Data")
    tab1, tab2, tab3 = st.tabs(["Leads Scored", "Products", "Transactions"])
    with tab1:
        df = pd.read_sql_table('leads_scored', conn)
        st.dataframe(df)
    with tab2:
        df = pd.read_sql_table('products', conn)
        st.dataframe(df)
    with tab3:
        df = pd.read_sql_table('transactions', conn)
        st.dataframe(df)

with col2:
    st.header("Ask me anything about the lead scoring data")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    user_input = get_text()
    send_button_clicked = st.button('Apply')

    if send_button_clicked:
        chat_response, sql_query_df, sql_code, chart = generate_chat_response(user_input)
        new_chat_entry = (user_input, chat_response, sql_query_df, sql_code, chart)
        st.session_state['chat_history'].insert(0, new_chat_entry)

    # Display chat history with charts if any
    for user_text, bot_text, query_df, query_code, chart in st.session_state['chat_history']:
        st.text(f"You: {user_text}")
        st.text(f"Bot: {bot_text}")
        st.dataframe(query_df)
        st.code(query_code, language='sql', line_numbers=True)

        if chart:
            st.plotly_chart(chart)


    if st.button('Clear Chat History'):
        st.session_state['chat_history'] = []