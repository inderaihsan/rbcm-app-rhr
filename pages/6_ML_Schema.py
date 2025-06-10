import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd

st.session_state.setdefault('auth', None)

st.title("📊 Machine Learning Schema")

# Input form
host = st.text_input("Hostname")
username = st.text_input("Username")
password = st.text_input("Password", type="password")
database_name = st.text_input("Database")

# Connect button
if st.button("Connect to Database"):
    try:
        engine = create_engine(f"postgresql://{username}:{password}@{host}/{database_name}")
        st.session_state['engine'] = engine
        st.session_state['auth'] = True
        st.success("Connected successfully!")
    except Exception as e:
        st.error(f"Connection failed: {e}")
        st.session_state['auth'] = False

# If connected, allow custom query execution
if st.session_state.get("auth"):
    default_query = "SELECT * FROM your_table_name LIMIT 100;"
    user_query = st.text_area("SQL Query", value=default_query, height=100)

    if st.button("Execute Query"):
        try:
            engine = st.session_state['engine']
            with engine.connect() as connection:
                result = connection.execute(text(user_query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                if df.empty:
                    st.warning("Query executed successfully, but returned no results.")
                else:
                    st.success(f"Query executed successfully. Returned {len(df)} rows.")
                    st.dataframe(df)
        except Exception as e:
            st.error(f"Query execution failed: {e}")
