import streamlit as st
import pandas as pd
import os
import tempfile
import matplotlib.pyplot as plt
import networkx as nx
import traceback
import re
import graphviz
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env for API keys
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Streamlit setup
st.set_page_config(page_title="CSV ERD & Chart Generator", layout="wide")
st.title("📊 CSV ER Diagram & Chart Generator with LLM (Gemini)")

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

    # 2. High-Level ER Diagram with Graphviz
    st.header("Step 2: High-Level ER Diagram (Entity Relationships)")

    relationships = []
    for table_name, df in st.session_state.csv_data.items():
        for col in df.columns:
            if col.endswith('_id'):
                ref_entity = col[:-3].capitalize()
                src_entity = table_name.replace(".csv", "").capitalize()
                relationships.append((src_entity, ref_entity, col))

    dot = graphviz.Digraph(format='png')
    dot.attr(rankdir='LR', size='8,5')

    # Nodes
    all_entities = set()
    for src, tgt, _ in relationships:
        all_entities.add(src)
        all_entities.add(tgt)

    for entity in all_entities:
        dot.node(entity, shape='box', style='filled', color='lightblue')

    # Edges
    for src, tgt, label in relationships:
        dot.edge(src, tgt, label=label, fontsize='10', fontcolor='gray30')

    st.graphviz_chart(dot)

    # 2.5 UML Diagram (Class Diagram Style)
    st.header("Step 2.5: UML Diagram (CSV Structure Overview)")

    uml = graphviz.Digraph(format="png")
    uml.attr(rankdir="TB", size="10,10")

    for table_name, df in st.session_state.csv_data.items():
        class_name = table_name.replace(".csv", "")
        fields = "\l".join(df.columns) + "\l"  # left-justified fields with line breaks
        uml.node(class_name, label=f"{class_name}|{fields}", shape="record", style="filled", fillcolor="lightyellow")

    # Add lines between *_id fields and other tables
    for table_name, df in st.session_state.csv_data.items():
        src = table_name.replace(".csv", "")
        for col in df.columns:
            if col.endswith("_id"):
                target = col[:-3].capitalize()
                uml.edge(src, target, label=col, fontcolor="gray40", fontsize="10")

    st.graphviz_chart(uml)

    ## YHA TAK ADD KRA HAI BAS

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

        # Prompt for Gemini
        prompt = f"""
You are a Python data analysis assistant.

Your task is to write correct, executable Python code using pandas and matplotlib to analyze CSV files and generate visualizations.

🧠 Key Rules to Follow:
1. **Use only the columns listed below. Do not guess or hallucinate column names.**
2. **Use only the CSV file where a column is actually present.**
   - Example: If `department_id` is needed, use the file that contains `department_id` (e.g., Teacher.csv).
3. **Always load the correct CSVs into pandas using `pd.read_csv()`** with exact file names.
4. **Rename overlapping or generic column names** like `id`, `name`, `email`, `title`, etc., *immediately* after loading each DataFrame to avoid conflicts during merges.
   - Use consistent, descriptive names: e.g., `id` in Student.csv becomes `student_id`, `name` becomes `student_name`, etc.
5. **Always rename merge key columns before merging** to prevent `_x`, `_y` suffixes in the output.
6. **Perform merges using clearly renamed key columns only.**
7. **Visualizations must be based on the final processed DataFrame** — make sure column names used in plots exist in the final DataFrame.
8. **Return only executable Python code — no explanations or comments unless required by Python syntax.**
9. **Always set labels, titles, and axis information in your matplotlib visualizations for clarity.**
10. Ensure all matplotlib plots are displayed using `plt.show()`.

📂 CSV Headers:
{full_header_info}

📌 User Query:
{user_query}
"""


        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            code = response.text

            # Extract code block if present
            code_blocks = re.findall(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
            if code_blocks:
                code = code_blocks[0]
            else:
                code = code.strip().split("\n\n")[0]

            code = code.replace("plt.show()", "")
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
