import streamlit as st
import pandas as pd
import os
import tempfile
import matplotlib.pyplot as plt
import networkx as nx
import traceback
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import re

# Load .env for API keys
load_dotenv()

# Streamlit setup
st.set_page_config(page_title="CSV ERD & Chart Generator", layout="wide")
st.title("📊 CSV ER Diagram & Chart Generator with LLM")

# Session state init
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = {}

# 1. Upload CSVs
st.header("Step 1: Upload Multiple CSV Files")
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        st.session_state.csv_data[file.name] = pd.read_csv(file)

    st.success(f"Uploaded {len(st.session_state.csv_data)} file(s).")

    # 2. Generate ER Diagram with FK inference
    st.header("Step 2: Auto-generated ER Diagram (with foreign key inference)")
    G = nx.DiGraph()

    # Add nodes
    for table_name in st.session_state.csv_data:
        G.add_node(table_name)

    # Foreign key inference logic: *_id -> id in target table
    for table_name, df in st.session_state.csv_data.items():
        for col in df.columns:
            if col.endswith('_id'):
                ref_table_candidate = col[:-3].lower()
                for target_table, target_df in st.session_state.csv_data.items():
                    if table_name == target_table:
                        continue
                    if 'id' in target_df.columns and ref_table_candidate in target_table.lower():
                        G.add_edge(table_name, target_table, label=col)

    # Add edges for exact common column names
    for fname, df in st.session_state.csv_data.items():
        for other_fname, other_df in st.session_state.csv_data.items():
            if fname == other_fname:
                continue
            common_cols = set(df.columns).intersection(set(other_df.columns))
            for col in common_cols:
                if not G.has_edge(fname, other_fname):
                    G.add_edge(fname, other_fname, label=col)

    # Draw graph
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, ax=ax)
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
    st.pyplot(fig)

    # 3. Business Query to Chart
    st.header("Step 3: Ask Business Query to Generate Chart + Code")
    user_query = st.text_area("Enter your query (e.g., 'Show monthly sales trend for top 5 products')")

    if user_query:
        st.subheader("📄 CSV Files & Columns")
        for fname, df in st.session_state.csv_data.items():
            st.text(f"{fname}: {list(df.columns)}")

        # CSV structure for prompt
        full_header_info = "\n".join([
            f"{fname}: {', '.join(df.columns)}" for fname, df in st.session_state.csv_data.items()
        ])

        # Prompt for LLM
        prompt = f"""
You are a Python data analysis assistant.
Only use the available columns listed below. Do not guess or hallucinate column names.
Use the correct CSV file where a column exists.

If you want to access the column `department_id`, look at the file which contains that column (like `Teacher.csv`).

If you need to join two tables:
- Columns ending in `_id` are usually foreign keys.
- Join these to the `id` column of the referenced table.
- For example: if `Enrollment.csv` has `course_id` and `Course.csv` has `id`, then join like this:

  pd.merge(enrollment, course, left_on='course_id', right_on='id')

Do not assume both files have the same column name for joining. Always use the actual column names from the headers.

Use pandas and matplotlib to generate a chart based on the user's query.

CSV Headers:
{full_header_info}

Query: {user_query}

Return only valid Python code without any explanations.
"""


        try:
            llm = ChatGroq(temperature=0, model_name="meta-llama/llama-4-scout-17b-16e-instruct")
            chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("{query}"))
            code = chain.run(query=prompt)

            # Extract Python code from response
            code_blocks = re.findall(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
            if code_blocks:
                code = code_blocks[0]
            else:
                code = code.strip().split("\n\n")[0]

            code = code.replace("plt.show()", "")  # Remove plt.show()
            st.subheader("🧠 Generated Python Code")
            st.code(code, language='python')

            # Validate column names
            all_columns = set()
            for df in st.session_state.csv_data.values():
                all_columns.update(df.columns)

            invalid_columns = []
            for col in re.findall(r"['\"]([a-zA-Z0-9_]+)['\"]", code):
                if col.endswith("_id") or col.endswith("_name") or col.endswith("_date"):
                    if col not in all_columns:
                        invalid_columns.append(col)

            if invalid_columns:
                st.warning(f"⚠️ The following column(s) may not exist: {invalid_columns}")

            # 4. Run the generated code
            st.subheader("📊 Generated Chart")

            temp_dir = tempfile.TemporaryDirectory()
            csv_dir = temp_dir.name
            for fname, df in st.session_state.csv_data.items():
                df.to_csv(os.path.join(csv_dir, fname), index=False)

            original_cwd = os.getcwd()
            os.chdir(csv_dir)

            try:
                plt.close('all')
                globals_dict = {'pd': pd, 'plt': plt}
                locals_dict = {}

                try:
                    exec(code, globals_dict, locals_dict)
                    fig = plt.gcf()
                    st.pyplot(fig)
                except KeyError as ke:
                    st.error(f"🛑 Column not found in your data: {ke}")
                except Exception:
                    st.error("Something went wrong during code execution.")
                    st.exception(traceback.format_exc())
            finally:
                os.chdir(original_cwd)

        except Exception:
            st.error("Failed to generate chart or code.")
            st.exception(traceback.format_exc())
