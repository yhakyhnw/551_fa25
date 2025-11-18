import tempfile
import io
import contextlib
import streamlit as st
from SQL_package import PSO  


def ds_sum(title, df):
    st.markdown(f"#### {title}:")
    headers = list(df.data.keys())

    if df.data:
        n_rows = len(list(df.data.values())[0])
    else:
        n_rows = 0

    st.write(f"Columns: {headers}")
    st.write(f"Rows: {n_rows}")


def step_desc(step):
    op_label = step.get("op", "")

    desc_parts = []
    if op_label == "groupby_agg":
        group_cols_str = step.get("group_cols", "")
        agg_spec_str = step.get("agg_spec", "")

        if group_cols_str:
            desc_parts.append(f"group_by = {group_cols_str}")

        if agg_spec_str:
            desc_parts.append(f"agg = {agg_spec_str}")

    elif op_label == "filter":
        expr_str = step.get("expr", "")
        if expr_str:
            desc_parts.append(f"expr = {expr_str}")

    elif op_label == "join":
        left_on_str = step.get("left_on", "")
        right_on_str = step.get("right_on", "")
        join_type_str = step.get("join_type", "")

        if left_on_str:
            desc_parts.append(f"left_on = {left_on_str}")

        if right_on_str:
            desc_parts.append(f"right_on = {right_on_str}")

        if join_type_str:
            desc_parts.append(f"join_type = {join_type_str}")

    elif op_label == "project":
        cols_str = step.get("columns", "")
        if cols_str:
            desc_parts.append(f"columns = {cols_str}")

    desc_text = " | ".join(desc_parts)
    return op_label, desc_text

def run_with_capture(function, *args, **kwargs):
    output_buffer = io.StringIO()

    with contextlib.redirect_stdout(output_buffer):
        result = function(*args, **kwargs)
    
    captured_logs = output_buffer.getvalue()
    
    return result, captured_logs

def _autoAdvance():
    demo = st.session_state.get("demo_mode")

    if demo == "1 dataset demo":
        if st.session_state.get("df1") is not None:
            st.session_state.current_stage = 1

    elif demo == "2 dataset demo (join)":
        if st.session_state.get("df1") is not None and st.session_state.get("df2") is not None:
            st.session_state.current_stage = 1

def loadPSO(uploaded_file, encoding = "UTF-8", delimiter = ",", header = None):
    if uploaded_file is None:
        return None

    filename = uploaded_file.name

    # check csv
    if not filename.lower().endswith(".csv"):
        st.error("Error: Only .csv files are supported.")
        return None

    # write to temp file for parser
    suffix = "." + filename.rsplit(".", 1)[-1]

    with tempfile.NamedTemporaryFile(delete = False, suffix = suffix) as f_in:
        f_in.write(uploaded_file.getvalue())
        tmp_path = f_in.name

    return PSO.parser(file = tmp_path, header = header, encoding = encoding, delimiter = delimiter)

def parser_params(label):
    st.markdown(f"##### Parser options for {label}")

    if "parser_encoding" not in st.session_state:
        st.session_state.parser_encoding = "UTF-8"
    if "parser_delimiter" not in st.session_state:
        st.session_state.parser_delimiter = ","

    encoding = st.text_input("File encoding (UTF-8 default)", 
                             value = st.session_state.parser_encoding, 
                             key = f"encoding_{label}")

    delimiter = st.text_input("Delimiter (comma default)", 
                              value = st.session_state.parser_delimiter, max_chars = 1, 
                              key = f"delimiter_{label}")

    header_str = st.text_input("Header override (comma-separated, optional)", 
                               value = "", 
                               key = f"header_{label}")

    st.session_state.parser_encoding = encoding
    st.session_state.parser_delimiter = delimiter

    if header_str.strip():
        header = []
        for header_item in header_str.split(","):
            header.append(header_item.strip())
    else:
        header = None

    return encoding, delimiter, header

def data_load(label, df_key, upload_flag_key):
    st.subheader(f"{label} source")
    col_dummy, col_upload = st.columns(2)

    if col_dummy.button("Use sample data", key = f"use_sample_{df_key}"):
        st.session_state[df_key] = PSO.parser()
        st.session_state[f"{df_key}_source"] = "sample"
        st.session_state[upload_flag_key] = False
        
        _autoAdvance()
        st.rerun()

    if col_upload.button("Upload CSV", key = f"upload_btn_{df_key}"):
        st.session_state[upload_flag_key] = True

    if st.session_state.get(upload_flag_key):
        encoding, delimiter, header = parser_params(label)
        uploaded = st.file_uploader("CSV file", type = ["csv"], key = f"upload_{label}")

        if uploaded:
            st.session_state[df_key] = loadPSO(uploaded, 
                                               encoding = encoding,
                                               delimiter = delimiter, 
                                               header = header)
                                                            
            if st.session_state[df_key] is not None:
                st.session_state[f"{df_key}_source"] = uploaded.name
                st.success(f"Loaded {uploaded.name}")

            st.session_state[upload_flag_key] = False

            _autoAdvance()
            st.rerun()

def step1():
    st.markdown("### Step 1: Data Upload")

    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = None


    if "df1" not in st.session_state:
        st.session_state.df1 = None
    if "df2" not in st.session_state:
        st.session_state.df2 = None

    st.markdown("#### Select demo type:")

    col1, col2 = st.columns(2)

    if col1.button("1 dataset demo"):
        st.session_state.demo_mode = "1 dataset demo"

    if col2.button("2 dataset demo (join)"):
        st.session_state.demo_mode = "2 dataset demo (join)"

    if st.session_state.demo_mode == "1 dataset demo":
        data_load("Dataset 1", "df1", "show_upload_1")

    if st.session_state.demo_mode == "2 dataset demo (join)":
        data_load("Dataset 1", "df1", "show_upload_1")
        st.markdown("---")
        data_load("Dataset 2", "df2", "show_upload_2")

    if st.session_state.demo_mode == "1 dataset demo" and st.session_state.df1 is not None:
        st.session_state.current_stage = 1

    if st.session_state.demo_mode == "2 dataset demo (join)" and st.session_state.df1 is not None and st.session_state.df2 is not None:
        st.session_state.current_stage = 1

    # show what chosen
    if st.session_state.demo_mode == "1 dataset demo":
        src1 = st.session_state.get("df1_source")

        if src1:
            st.markdown("#### Current selection")
            st.info(f"Dataset 1: {src1}")

    elif st.session_state.demo_mode == "2 dataset demo (join)":

        src1 = st.session_state.get("df1_source")
        src2 = st.session_state.get("df2_source")

        if src1 or src2:
            st.markdown("#### Current selections")
        if src1:
            st.info(f"Dataset 1: {src1}")
        if src2:
            st.info(f"Dataset 2: {src2}")

def pipeline_diagram():
    pipeline = st.session_state.get("pipeline", [])
    demo_mode = st.session_state.get("demo_mode")

    if demo_mode == "2 dataset demo (join)":
        base = "Dataset 1 (+ Dataset 2)"
    else:
        base = "Dataset 1"

    if not pipeline:
        st.write(base)
        return

    labels = []
    for step in pipeline:
        op = step.get("op")
        params = []

        if op == "groupby_agg":
            if step.get("group_cols"):
                params.append(f"group_by={step['group_cols']}")
            if step.get("agg_spec"):
                params.append(f"agg={step['agg_spec']}")

        elif op == "filter":
            if step.get("expr"):
                params.append(f"expr={step['expr']}")

        elif op == "join":
            if step.get("left_on"):
                params.append(f"left_on={step['left_on']}")
            if step.get("right_on"):
                params.append(f"right_on={step['right_on']}")
            if step.get("join_type"):
                params.append(f"type={step['join_type']}")

        elif op == "project":
            if step.get("columns"):
                params.append(f"cols={step['columns']}")

        if params:
            label = f"{op}({', '.join(params)})"
        else:
            label = op

        labels.append(label)

    diagram = base + "  ➜  " + "  ➜  ".join(labels)
    st.write(diagram)

def groupby_agg_params():
    st.markdown("##### Configure group by + aggregate")
    
    group_cols = st.text_input("Group by columns (comma-separated)", key = "groupby_cols_input")
    agg_spec = st.text_input("Aggregation spec (e.g. col1:sum,col2:mean)", key = "groupby_agg_input")

    if st.button("Confirm groupby+agg"):

        if "pipeline" not in st.session_state:
            st.session_state.pipeline = []
        
        st.session_state.pipeline.append({"op": "groupby_agg", 
                                          "group_cols": group_cols, 
                                          "agg_spec": agg_spec})
        
        st.success("Added groupby+agg to pipeline.")
        st.rerun()

def filter_params():
    st.markdown("##### Configure filter")
    
    expr = st.text_input("Filter expression (e.g. col1 > 10 & col2 == 'A')", key = "filter_expr_input")

    if st.button("Confirm filter"):
    
        if "pipeline" not in st.session_state:
            st.session_state.pipeline = []
        
        st.session_state.pipeline.append({"op": "filter", "expr": expr})
        
        st.success("Added filter to pipeline.")
        st.rerun()

def join_params():
    st.markdown("##### Configure join")
    
    left_on = st.text_input("Left key column (Dataset 1)", key = "join_left_on_input")
    right_on = st.text_input("Right key column (Dataset 2)", key = "join_right_on_input")
    join_type = st.text_input("Join type (inner, left, right, outer)", value = "inner", key = "join_type_input")

    if st.button("Confirm join"):
        
        if "pipeline" not in st.session_state:
            st.session_state.pipeline = []
        
        st.session_state.pipeline.append({"op": "join",
                                          "left_on": left_on,
                                          "right_on": right_on,
                                          "join_type": join_type})
        
        st.success("Added join to pipeline.")
        st.rerun()

def project_params():
    st.markdown("##### Configure projection")
    
    cols = st.text_input("Projection columns (comma-separated)", key = "project_cols_input")
    
    if st.button("Confirm projection"):
        if "pipeline" not in st.session_state:
            st.session_state.pipeline = []
        st.session_state.pipeline.append({"op": "project", 
                                          "columns": cols})
        
        st.success("Added projection to pipeline.")
        st.rerun()

def step2():
    st.markdown("### Step 2: Data Analysis")

    # show dataset info
    if "df1" in st.session_state and st.session_state.df1 is not None:
        ds_sum("Dataset 1", st.session_state.df1)

    if "df2" in st.session_state and st.session_state.df2 is not None:
        ds_sum("Dataset 2", st.session_state.df2)

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = []

    st.markdown("---")
    st.markdown("#### Configure operations")

    demo_mode = st.session_state.get("demo_mode")

    tab_gb, tab_filt, tab_join, tab_proj = st.tabs(["Group By & Aggregate", "Filter", "Join", "Project"])

    with tab_gb:
        groupby_agg_params()

    with tab_filt:
        filter_params()

    with tab_join:
        if demo_mode != "2 dataset demo (join)":
            st.warning("Join is only available for the 2 dataset demo (join).")
        else:
            join_params()

    with tab_proj:
        project_params()

    st.markdown("---")
    st.markdown("#### Current pipeline")
    pipeline_diagram()

    # make sure join is used for 2 dataset demo
    demo_mode = st.session_state.get("demo_mode")
    pipeline = st.session_state.get("pipeline")

    if demo_mode == "2 dataset demo (join)":
        if pipeline and pipeline[0].get("op") != "join":
            st.warning("For the 2‑dataset demo, the first pipeline step should be a JOIN.")

    if pipeline:
        st.markdown("##### Edit pipeline steps")

        for step_index, step in enumerate(pipeline):
            op_label, desc_text = step_desc(step)
            if not op_label:
                op_label = f"step {step_index + 1}"

            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.write(f"{step_index + 1}. {op_label}")
                if desc_text:
                    st.caption(desc_text)
            with col_b:
                if st.button("Delete", key = f"delete_step_{step_index}"):
                    del pipeline[step_index]
                    st.session_state.pipeline = pipeline
                    st.rerun()

        if st.button("Clear entire pipeline", key = "clear_pipeline"):
            st.session_state.pipeline = []
            st.rerun()

    if st.button("Step 3: Run pipeline"):
        st.session_state.current_stage = 2
        st.rerun()

def step3():
    st.markdown("### Step 3: Results Display")

    pipeline = st.session_state.get("pipeline", [])
    df1 = st.session_state.get("df1")
    df2 = st.session_state.get("df2")
    demo_mode = st.session_state.get("demo_mode")
    step_results = []

    if df1 is None:
        st.error("No Dataset 1 available to run the pipeline.")
        st.markdown("---")
        return

    if not pipeline:
        st.warning("Pipeline is empty. Please add operations in Step 2")
        st.markdown("---")
        return

    current = df1

    for step_num, step in enumerate(pipeline, start = 1):
        step_logs = ""
        op = step.get("op")
        st.markdown(f"**Step {step_num}: {op}**")

        _, desc_text = step_desc(step)
        if desc_text:
            st.caption(desc_text)

        # groupby + aggregate
        if op == "groupby_agg":
            group_cols_str = step.get("group_cols", "")
            agg_spec_str = step.get("agg_spec", "")

            by = []

            for group_col in group_cols_str.split(","):
                c_stripped = group_col.strip()
            
                if c_stripped:
                    by.append(c_stripped)

            agg_param = None

            if agg_spec_str:
                spec_parts = []
                for agg_part in agg_spec_str.split(","):
                    p_stripped = agg_part.strip()

                    if p_stripped:
                        spec_parts.append(p_stripped)

                agg_dict = {}

                for part in spec_parts:
                    if ":" in part:
                        col, func = part.split(":", 1)
                        col = col.strip()
                        func = func.strip()
            
                        if col:
                            agg_dict.setdefault(col, []).append(func)
            
                if agg_dict:
                    agg_param = agg_dict

            try:
                if by:
                    gb, gb_logs = run_with_capture(current.group_by, by)
                    step_logs += gb_logs
            
                else:
                    st.warning("No group by columns provided; using all rows as a single group.")
                    gb = None

                if gb is None:
                    st.error("Failed to create group_by object. Check group by columns.")
                    st.markdown("---")
                    return

                if agg_param is None:
                    current, agg_logs = run_with_capture(gb.agg, "count")
            
                else:
                    current, agg_logs = run_with_capture(gb.agg, agg_param)
            
                step_logs += agg_logs

            except Exception as e:
                if step_logs.strip():
                    st.code(step_logs, language = "text")
            
                st.error(f"{e}")
                st.markdown("---")
                return

        # filter
        elif op == "filter":
            expr = step.get("expr", "")
            if not expr.strip():
                st.warning("Empty filter expression; skipping this step.")
                continue
            
            try:
                current, step_logs = run_with_capture(current.filter, expr)
            except Exception as e:
                st.error(f"{e}")
            
                if step_logs.strip():
                    st.code(step_logs, language = "text")
                st.markdown("---")
                return

        # join
        elif op == "join":
            if demo_mode != "2 dataset demo (join)":
                st.warning("Join step ignored because demo mode is not '2 dataset demo (join)'.")
                continue

            if df2 is None:
                st.error("Join requested but Dataset 2 is not available.")
                st.markdown("---")
                return

            left_on = step.get("left_on", "")
            right_on = step.get("right_on", "")
            join_type = step.get("join_type", "inner")

            if not left_on.strip():
                st.error("Join step missing 'left_on' column name.")
                st.markdown("---")
                return

            try:
                current, step_logs = run_with_capture(current.join, df2, left_on = left_on, right_on = right_on or None, join_type = join_type)
            except Exception as e:
                st.error(f"{e}")
                if step_logs.strip():
                    st.code(step_logs, language = "text")
                st.markdown("---")
                return

        # projection
        elif op == "project":
            cols_str = step.get("columns", "")
            col_list = []

            for column_name in cols_str.split(","):
                c_stripped = column_name.strip()

                if c_stripped:
                    col_list.append(c_stripped)

            if not col_list:
                st.warning("Projection step has no columns specified; skipping this step.")
                continue
            try:
                current, step_logs = run_with_capture(current.projection, ",".join(col_list))
                
            except Exception as e:
                st.error(f"{e}")
                if step_logs.strip():
                    st.code(step_logs, language = "text")
                st.markdown("---")
                return

        else:
            st.warning(f"Unknown operation '{op}' encountered; skipping.")
            continue

        if current is None or not isinstance(current, PSO):
            st.error("Pipeline step did not produce a valid dataset. Stopping execution.")
            if step_logs.strip():
                st.code(step_logs, language = "text")
            st.markdown("---")
            return

        try:
            snapshot = str(current)
        except Exception:
            snapshot = current

        step_results.append((step_num, op, desc_text, snapshot))

    st.markdown("#### Final result")
    try:
        st.code(str(current), language = "text")
    except Exception:
        st.code(current, language = "text")

    st.markdown("---")
    st.markdown("#### Results after each step")

    for step_num, op, desc_text, snapshot in step_results:
        with st.expander(f"View data after step {step_num}", expanded = False):
            if isinstance(snapshot, str):
                st.code(snapshot, language = "text")
            else:
                st.write(snapshot)


def main():
    # hard reset
    if st.button("Reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.current_stage = 0
        st.rerun()

    # title
    st.markdown("# Analysis of second-hand car prices")
    st.markdown("### DSCI 551 Project (SQL)")
    st.markdown("---")

    # progress bar
    if "current_stage" not in st.session_state:
        st.session_state.current_stage = 0

    stages = ["Data Upload", "Data Analysis", "Data Projection"]
    current_stage = st.session_state.current_stage
    progress_value = (current_stage + 1) / len(stages)

    st.progress(progress_value)
    st.markdown(f"**Stage {current_stage + 1} of {len(stages)}: {stages[current_stage]}**")

    # run it!!!!!
    if current_stage == 0:
        step1()
    elif current_stage == 1:
        step2()
    elif current_stage == 2:
        step3()

if __name__ == "__main__":
    main()