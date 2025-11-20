import tempfile
import io, os, sys
import contextlib
import itertools
import streamlit as st

base_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.dirname(base_directory)
packages_directory = os.path.join(root_directory, "packages")

sys.path.append(packages_directory)

from NoSQL_package import NoSql, pretty_print_nosql  

def convert_array_to_ndjson(src_path: str) -> str:

    with open(src_path, "r", encoding="utf-8") as infile:
        # Detect first non-whitespace character
        while True:
            ch = infile.read(1)
            if not ch:
                # empty file
                return src_path
            if not ch.isspace():
                break

        # If file does not start with JSON array → treat as NDJSON already
        if ch != "[":
            return src_path

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ndjson", mode="w", encoding="utf-8")
        out = tmp_file

        depth = 0
        in_string = False
        escape = False
        buffer = []

        while True:
            ch = infile.read(1)
            if not ch:  # EOF
                break

            if in_string:
                buffer.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False

            else:
                if ch == '"':
                    in_string = True
                    buffer.append(ch)

                elif ch == '{':
                    depth += 1
                    buffer.append(ch)

                elif ch == '}':
                    buffer.append(ch)
                    depth -= 1

                    if depth == 0:
                        obj = "".join(buffer).strip()
                        if obj:
                            out.write(obj + "\n")
                        buffer = []

                elif ch == ']':
                    # End of JSON array
                    break

                else:
                    if depth > 0:
                        buffer.append(ch)

        out.close()
        return tmp_file.name

def capture_pretty(ns_obj, dp_lim: int | None = 5) -> str:
    # Handle generator/iterable preview
    if not isinstance(ns_obj, (list, tuple)) and hasattr(ns_obj, "__next__"):
        # Direct generator: take first chunk
        try:
            ns_target = next(ns_obj)
        except StopIteration:
            return "<empty iterable>"
        except Exception:
            ns_target = ns_obj
    elif not isinstance(ns_obj, (list, tuple)) and hasattr(ns_obj, "__iter__"):
        # Iterable but not a direct generator, fall back to iter()
        try:
            ns_target = next(iter(ns_obj))
        except StopIteration:
            return "<empty iterable>"
        except Exception:
            ns_target = ns_obj
    elif isinstance(ns_obj, (list, tuple)) and ns_obj and isinstance(ns_obj[0], NoSql):
        ns_target = ns_obj[0]
    else:
        ns_target = ns_obj

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            pretty_print_nosql(ns_target, dp_lim = dp_lim)
        except Exception:
            # Fallback: just print the string representation so the user sees something
            print(str(ns_target))
    return buffer.getvalue()

def nosql_ds_sum(title, ns_obj):
    st.markdown(f"#### {title}:")
    if ns_obj is None:
        st.info("No data loaded.")
        return

    # Removed generator conversion and unknown data format reporting for streaming data.

    # Check if chunked data
    if isinstance(ns_obj, (list, tuple)) and ns_obj and isinstance(ns_obj[0], NoSql):
        chunks = ns_obj
        try:
            chunk_sizes = [len(chunk.data) for chunk in chunks if hasattr(chunk, 'data') and isinstance(chunk.data, list)]
        except Exception:
            st.write("Unknown data format")
            return

        if not chunk_sizes:
            st.write("Unknown data format")
            return

        total_docs = sum(chunk_sizes)
        from collections import Counter
        size_counts = Counter(chunk_sizes)

        parts = []
        for size, count in sorted(size_counts.items(), reverse=True):
            chunk_word = "chunk" if count == 1 else "chunks"
            doc_word = "document" if size == 1 else "documents"
            parts.append(f"{count} {chunk_word} × {size} {doc_word} per chunk")
        summary = " + ".join(parts)

        st.write(f"Total documents: {total_docs}")
        st.write(summary)

        # Find first non-empty chunk
        first_chunk = None
        for chunk in chunks:
            if hasattr(chunk, 'data') and isinstance(chunk.data, list) and len(chunk.data) > 0:
                first_chunk = chunk
                break

        if first_chunk is not None:
            first_doc = first_chunk.data[0]
            if isinstance(first_doc, dict):
                keys = list(first_doc.keys())
                st.write(f"Key fields: {keys}")
            else:
                st.write("Unknown data format")
        else:
            st.write("No documents found in chunks")

    else:
        # Only handle real NoSql instances; ignore anything else (e.g., generators)
        if not isinstance(ns_obj, NoSql):
            return
        if not hasattr(ns_obj, 'data') or not isinstance(ns_obj.data, list):
            return
        docs = ns_obj.data
        st.write(f"Total documents: {len(docs)}")
        keys_set = set()
        for doc in docs:
            if isinstance(doc, dict):
                keys_set.update(doc.keys())
            else:
                return
        keys = sorted(keys_set)
        st.write(f"Key fields: {keys}")

def nosql_step_desc(step):
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
        left_on = step.get("left_on", "")
        right_on = step.get("right_on", "")
        if left_on:
            desc_parts.append(f"local_field = {left_on}")
        if right_on:
            desc_parts.append(f"foreign_field = {right_on}")


    elif op_label == "project":
        cols_str = step.get("columns", "")
        if cols_str:
            desc_parts.append(f"fields = {cols_str}")

    desc_text = " | ".join(desc_parts)
    return op_label, desc_text




def group_by_params():
    st.markdown("##### Configure group by + aggregate")

    group_cols = st.text_input(
        "Group by fields (comma-separated)", key="nosql_groupby_cols_input"
    )
    agg_spec = st.text_input(
        "Aggregation spec (e.g. price:sum,qty:avg, *:count)",
        key="nosql_groupby_agg_input",
    )

    if st.button("Confirm groupby+agg", key="nosql_confirm_groupby"):
        if "nosql_pipeline" not in st.session_state:
            st.session_state.nosql_pipeline = []

        st.session_state.nosql_pipeline.append(
            {"op": "groupby_agg", "group_cols": group_cols, "agg_spec": agg_spec}
        )
        st.success("Added groupby+agg to NoSQL pipeline.")
        st.rerun()


def filter_params():
    st.markdown("##### Configure filter")

    expr = st.text_input(
        "Filter expression (Mongo-style JSON predicate as string, or shorthand)",
        key="nosql_filter_expr_input",
    )

    if st.button("Confirm filter", key="nosql_confirm_filter"):
        if "nosql_pipeline" not in st.session_state:
            st.session_state.nosql_pipeline = []

        st.session_state.nosql_pipeline.append({"op": "filter", "expr": expr})
        st.success("Added filter to NoSQL pipeline.")
        st.rerun()


def join_params():
    st.markdown("##### Configure join")

    left_on = st.text_input(
        "Local field (Dataset 1)",
        key="nosql_join_left_on_input",
    )
    right_on = st.text_input(
        "Foreign field (Dataset 2)",
        key="nosql_join_right_on_input",
    )

    if st.button("Confirm join", key="nosql_confirm_join"):
        if "nosql_pipeline" not in st.session_state:
            st.session_state.nosql_pipeline = []

        st.session_state.nosql_pipeline.append(
            {
                "op": "join",
                "left_on": left_on,
                "right_on": right_on,
            }
        )
        st.success("Added join to NoSQL pipeline.")
        st.rerun()


def project_params():
    st.markdown("##### Configure projection")

    cols = st.text_input(
        "Fields to keep (comma-separated)", key="nosql_project_cols_input"
    )

    if st.button("Confirm projection", key="nosql_confirm_project"):
        if "nosql_pipeline" not in st.session_state:
            st.session_state.nosql_pipeline = []

        st.session_state.nosql_pipeline.append(
            {"op": "project", "columns": cols}
        )
        st.success("Added projection to NoSQL pipeline.")
        st.rerun()

def _autoAdvance():
    demo_mode = st.session_state.get("nosql_demo_mode")

    if st.session_state.get("nosql_current_stage", 0) != 0:
        return

    if demo_mode == "1 dataset demo":
        if st.session_state.get("nosql_df1") is not None:
            st.session_state.nosql_current_stage = 1
            st.rerun()

    elif demo_mode == "2 dataset demo (join)":
        if (st.session_state.get("nosql_df1") is not None and
                st.session_state.get("nosql_df2") is not None):
            st.session_state.nosql_current_stage = 1
            st.rerun()


def data_load(label, df_key, upload_flag_key):
    st.subheader(f"{label} source")

    use_chunk = st.session_state.get("nosql_use_chunk", False)
    chunk_size_value = st.session_state.get("nosql_chunk_size") if use_chunk else None

    col_dummy, col_upload = st.columns(2)

    # dummy
    if col_dummy.button("Use sample data", key=f"use_sample_{df_key}"):
        try:
            if use_chunk and chunk_size_value is not None:
                loaded = NoSql.read_dummy(chunk_size=int(chunk_size_value))
            else:
                loaded = NoSql.read_dummy()

            if use_chunk:
                if isinstance(loaded, NoSql):
                    loaded = (loaded,)

                elif not isinstance(loaded, (list, tuple)):
                    pass
            else:
                if not isinstance(loaded, NoSql):
                    loaded = NoSql(list(loaded))

            st.session_state[df_key] = loaded
            st.session_state[f"{df_key}_source"] = "sample"
            st.success(f"{label}: loaded sample dataset")

            _autoAdvance()

        except Exception as error:
            st.error(f"Error loading sample dataset: {error}")

    # upload json
    if col_upload.button("Upload JSON", key=f"upload_btn_{df_key}"):
        st.session_state[upload_flag_key] = True

    if st.session_state.get(upload_flag_key):
        uploaded = st.file_uploader(
            "JSON file",
            type=["json"],
            key=f"upload_{label}",
        )

        if uploaded is not None:
            filename = uploaded.name

            if not filename.lower().endswith(".json"):
                st.error("Error: Only .json files are supported.")
            else:
                suffix = "." + filename.rsplit(".", 1)[-1]

                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(uploaded.getvalue())
                        tmp_path = temp_file.name

                    ndjson_path = convert_array_to_ndjson(tmp_path)

                    if use_chunk and chunk_size_value is not None:
                        loaded = NoSql.read_json(ndjson_path, chunk_size=int(chunk_size_value))
                    else:
                        loaded = NoSql.read_json(ndjson_path)

                    if use_chunk:
                        if isinstance(loaded, NoSql):
                            loaded = (loaded,)

                        elif not isinstance(loaded, (list, tuple)):
                            pass
                    else:
                        if not isinstance(loaded, NoSql):
                            loaded = NoSql(list(loaded))

                    st.session_state[df_key] = loaded
                    st.session_state[f"{df_key}_source"] = filename
                    st.success(f"{label}: loaded {filename}")

                    _autoAdvance()
                    st.session_state[upload_flag_key] = False

                except Exception as error:
                    st.error(f"Error loading JSON file: {error}")


def step1():

    st.markdown("### Step 1: Data Upload (NoSQL)")
    if "nosql_demo_mode" not in st.session_state:
        st.session_state.nosql_demo_mode = None

    st.markdown("#### Select demo type:")

    col1, col2 = st.columns(2)

    if col1.button("1 dataset demo", key="nosql_demo1"):
        st.session_state.nosql_demo_mode = "1 dataset demo"

    if col2.button("2 dataset demo (join)", key="nosql_demo2"):
        st.session_state.nosql_demo_mode = "2 dataset demo (join)"

    if st.session_state.nosql_demo_mode is not None:
        st.markdown("#### Chunking options (applies to all loaded datasets)")
        use_chunk = st.checkbox("Use chunking?", key="nosql_use_chunk")
        if use_chunk:
            st.number_input(
                "Chunk size",
                value=50000,
                min_value=1,
                step=1,
                key="nosql_chunk_size",
            )
        else:
            if "nosql_chunk_size" in st.session_state:
                del st.session_state["nosql_chunk_size"]

    if st.session_state.nosql_demo_mode == "1 dataset demo":
        data_load("Dataset 1", "nosql_df1", "nosql_show_upload_1")
    elif st.session_state.nosql_demo_mode == "2 dataset demo (join)":
        data_load("Dataset 1", "nosql_df1", "nosql_show_upload_1")
        data_load("Dataset 2", "nosql_df2", "nosql_show_upload_2")


def step2():
    """Step 2: Data Analysis (NoSQL)."""
    st.markdown("### Step 2: Data Analysis (NoSQL)")

    demo_mode = st.session_state.get("nosql_demo_mode")

    def _materialize_dataset(key: str):
        obj = st.session_state.get(key)
        if obj is None:
            return None

        # Already a single NoSql object
        if isinstance(obj, NoSql):
            return obj

        # List/tuple of NoSql chunks
        if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], NoSql):
            merged_docs = []
            for chunk in obj:
                if hasattr(chunk, "data") and isinstance(chunk.data, list):
                    merged_docs.extend(chunk.data)
            ns = NoSql(merged_docs)
            st.session_state[key] = ns
            return ns

        # Any other iterable (e.g., generator, itertools.tee) of NoSql chunks or dict docs
        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
            merged_docs = []
            try:
                for chunk in obj:
                    if isinstance(chunk, NoSql):
                        if hasattr(chunk, "data") and isinstance(chunk.data, list):
                            merged_docs.extend(chunk.data)
                    elif isinstance(chunk, dict):
                        merged_docs.append(chunk)
            except TypeError:
                # Not actually iterable in the way we expect; fall through
                pass

            if merged_docs:
                ns = NoSql(merged_docs)
                st.session_state[key] = ns
                return ns

        # Fallback: leave as-is and return None to indicate unusable for analysis
        return None

    def _ensure_chunk_container(key: str):
        """
        Ensure that the object stored under `key` is in a form suitable for nosql_ds_sum
        to report chunk sizes:

        - If it's a single NoSql (non-chunked), return it as-is.
        - If it's already a list/tuple of NoSql chunks, return it.
        - If it's a generator/iterable of NoSql chunks, materialize it into a tuple of
          chunks, store back into session_state, and return that tuple.
        """
        obj = st.session_state.get(key)
        if obj is None:
            return None

        # Single NoSql (non-chunked)
        if isinstance(obj, NoSql):
            return obj

        # Already a list/tuple of chunks
        if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], NoSql):
            return obj

        # Generator or other iterable of NoSql chunks
        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
            chunks = []
            try:
                for chunk in obj:
                    chunks.append(chunk)
            except TypeError:
                # Not iterable in the way we expect; fall through
                return obj

            if chunks:
                # Store back as an immutable tuple of chunks
                st.session_state[key] = tuple(chunks)
                return st.session_state[key]

        return obj

    # Prepare objects for dataset overview (preserve chunk structure if present)
    overview1 = _ensure_chunk_container("nosql_df1")
    overview2 = _ensure_chunk_container("nosql_df2") if demo_mode == "2 dataset demo (join)" else None

    st.markdown("#### Dataset overview")
    if overview1 is not None:
        nosql_ds_sum("Dataset 1", overview1)
    if demo_mode == "2 dataset demo (join)" and overview2 is not None:
        nosql_ds_sum("Dataset 2", overview2)

    # Materialize datasets for analysis (merge all chunks into single NoSql)
    ns1 = _materialize_dataset("nosql_df1")
    ns2 = _materialize_dataset("nosql_df2") if demo_mode == "2 dataset demo (join)" else None

    # Initialize pipeline state if needed
    if "nosql_pipeline" not in st.session_state:
        st.session_state.nosql_pipeline = []

    pipeline = st.session_state.get("nosql_pipeline", [])

    st.markdown("---")
    st.markdown("#### Add functions to pipeline (minimum 1 required)")

    tab_gb, tab_filt, tab_join, tab_proj = st.tabs(
        [
            "Group By & Aggregate",
            "Filter",
            "Join",
            "Project",
        ]
    )

    with tab_gb:
        group_by_params()

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

    if pipeline:
        for step_index, step in enumerate(pipeline):
            op_label, desc_text = nosql_step_desc(step)
            if not op_label:
                op_label = f"step {step_index + 1}"

            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.write(f"{step_index + 1}. {op_label}")
                if desc_text:
                    st.caption(desc_text)
            with col_b:
                if st.button("Delete", key=f"nosql_delete_step_{step_index}"):
                    del pipeline[step_index]
                    st.session_state.nosql_pipeline = pipeline
                    st.rerun()

        if st.button("Clear entire pipeline", key="nosql_clear_pipeline"):
            st.session_state.nosql_pipeline = []
            st.rerun()

    if st.button("Step 3: Run pipeline", key="nosql_step3_button"):
        st.session_state.nosql_current_stage = 2
        st.rerun()

    st.markdown("---")
    st.markdown("#### Dataset sample (preview)")
    if ns1 is not None:
        st.markdown("**Dataset 1**")
        st.code(capture_pretty(ns1, dp_lim=3), language="text")

    if demo_mode == "2 dataset demo (join)" and ns2 is not None:
        st.markdown("**Dataset 2**")
        st.code(capture_pretty(ns2, dp_lim=3), language="text")

def step3():
    """Step 3: Display results (NoSQL)."""
    st.markdown("### Step 3: Results Display (NoSQL)")

    demo_mode = st.session_state.get("nosql_demo_mode")
    pipeline = st.session_state.get("nosql_pipeline", [])

    def _normalize_result(obj):
        """Normalize pipeline step results into a NoSql object when appropriate.

        - If it's already a NoSql, return it.
        - If it's a list of dicts, wrap in NoSql.
        - Otherwise, return as-is.
        """
        if isinstance(obj, NoSql):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return NoSql(obj)
        return obj

    def _materialize_dataset(key: str):
        """
        Ensure that the object stored under `key` is a single NoSql instance.

        - If it's already a NoSql, return it.
        - If it's a list/tuple of NoSql chunks, merge their `.data` into one NoSql.
        - If it's an iterable (generator, tee, etc.) yielding NoSql chunks or dict docs,
          iterate once, merge, and store back as a NoSql.
        """
        obj = st.session_state.get(key)
        if obj is None:
            return None

        if isinstance(obj, NoSql):
            return obj

        if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], NoSql):
            merged_docs = []
            for chunk in obj:
                if hasattr(chunk, "data") and isinstance(chunk.data, list):
                    merged_docs.extend(chunk.data)
            ns = NoSql(merged_docs)
            st.session_state[key] = ns
            return ns

        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
            merged_docs = []
            try:
                for chunk in obj:
                    if isinstance(chunk, NoSql):
                        if hasattr(chunk, "data") and isinstance(chunk.data, list):
                            merged_docs.extend(chunk.data)
                    elif isinstance(chunk, dict):
                        merged_docs.append(chunk)
            except TypeError:
                pass

            if merged_docs:
                ns = NoSql(merged_docs)
                st.session_state[key] = ns
                return ns

        return None

    ns1 = _materialize_dataset("nosql_df1")
    ns2 = _materialize_dataset("nosql_df2") if demo_mode == "2 dataset demo (join)" else None

    if ns1 is None or not isinstance(ns1, NoSql):
        st.error("Dataset 1 is not available or not a valid NoSql object. Please complete Step 1.")
        return

    if not pipeline:
        st.info("Pipeline is empty. Showing Dataset 1 as the final result.")
        st.markdown("---")
        st.markdown("#### Final result (Dataset 1, pretty-printed)")
        st.code(capture_pretty(ns1, dp_lim=5), language="text")

        if demo_mode == "2 dataset demo (join)" and ns2 is not None:
            st.markdown("---")
            st.markdown("#### Dataset 2 sample (pretty-printed)")
            st.code(capture_pretty(ns2, dp_lim=5), language="text")
        return

    st.markdown("#### Current pipeline")
    st.markdown("---")

    current = ns1

    for step_index, step in enumerate(pipeline, start=1):
        op_label, desc_text = nosql_step_desc(step)
        if not op_label:
            op_label = f"step {step_index}"

        st.markdown(f"**Step {step_index}: {op_label}**")
        if desc_text:
            st.caption(desc_text)

        op = step.get("op")

        try:
            # groupby + aggregate
            if op == "groupby_agg":
                group_cols_str = step.get("group_cols", "")
                agg_spec_str = step.get("agg_spec", "")

                group_fields = [c.strip() for c in group_cols_str.split(",") if c.strip()]
                agg_param = None
                if agg_spec_str:
                    agg_dict: dict[str, list[str]] = {}
                    for part in agg_spec_str.split(","):
                        part = part.strip()
                        if not part:
                            continue
                        if ":" in part:
                            field, func = part.split(":", 1)
                            field = field.strip()
                            func = func.strip()
                            if field:
                                agg_dict.setdefault(field, []).append(func)
                    if agg_dict:
                        agg_param = agg_dict

                if agg_param is None:
                    agg_param = {"*": ["count"]}

                result = current.aggregate(group_fields or None, agg_param)
                current = _normalize_result(result)

            elif op == "filter":
                expr_raw = step.get("expr", "")
                if not expr_raw.strip():
                    st.warning("Empty filter expression; skipping this step.")
                    continue
                try:
                    import ast
                    expr = ast.literal_eval(expr_raw)
                except Exception:
                    raise ValueError("Invalid filter expression: must be a Python-style dict")
                result = current.filter(expr)
                current = _normalize_result(result)

            elif op == "join":
                if demo_mode != "2 dataset demo (join)":
                    st.warning("Join step ignored because demo mode is not '2 dataset demo (join)'.")
                    continue

                if ns2 is None or not isinstance(ns2, NoSql):
                    st.error("Join requested but Dataset 2 is not available or not a valid NoSql object.")
                    return

                left_on = step.get("left_on", "").strip()
                right_on = step.get("right_on", "").strip()

                if not left_on:
                    st.error("Join step missing local_field (Dataset 1).")
                    return

                if not right_on:
                    # If foreign field is omitted, default to same as local
                    right_on = left_on

                result = current.join(
                    from_field=ns2,
                    local_field=left_on,
                    foreign_field=right_on,
                    as_field="joined",
                )
                current = _normalize_result(result)

            elif op == "project":
                cols_str = step.get("columns", "")
                fields = [c.strip() for c in cols_str.split(",") if c.strip()]
                if not fields:
                    st.warning("Projection step has no fields specified; skipping this step.")
                    continue
                proj_dict = {name: 1 for name in fields}
                result = current.project(proj_dict)
                current = _normalize_result(result)

            else:
                st.warning(f"Unknown operation '{op}' encountered; skipping.")
                continue

        except Exception as error:
            st.error(f"Error at step {step_index} ({op}): {error}")
            st.markdown("---")
            st.markdown("#### Partial result up to error (pretty-printed)")
            st.code(capture_pretty(current, dp_lim=5), language="text")
            return

    st.markdown("#### Final result")

    # If result is a list of dicts, wrap in NoSql so pretty_print_nosql can format it nicely
    pretty_target = current
    if isinstance(current, list) and current and isinstance(current[0], dict):
        pretty_target = NoSql(current)

    # Decide safe pretty-print limit
    limit = None
    if isinstance(pretty_target, NoSql) and hasattr(pretty_target, "data"):
        n_docs = len(pretty_target.data)
        if n_docs > 200:
            limit = 200  # cap large results
    else:
        limit = 200  # non‑NoSql fallback safeguard

    text = capture_pretty(pretty_target, dp_lim=limit)

    if limit is not None:
        st.caption(f"Showing first {limit} documents (full result truncated).")

    st.code(text, language="text")


def main():

    # soft reset
    if st.button("Reset NoSQL demo"):
        nosql_keys = ["nosql_demo_mode", "nosql_df1", "nosql_df2", "nosql_df1_source", "nosql_df2_source",
                      "nosql_show_upload_1", "nosql_show_upload_2", "nosql_current_stage",
                      "nosql_use_chunk", "nosql_chunk_size", "nosql_pipeline", "nosql_groupby_cols_input", "nosql_groupby_agg_input",
                      "nosql_filter_expr_input",
                      "nosql_join_left_on_input", "nosql_join_right_on_input",
                      "nosql_project_cols_input"]

        for state_key in nosql_keys:
            if state_key in st.session_state:
                del st.session_state[state_key]

        # back to step 1
        st.session_state.nosql_current_stage = 0
        st.rerun()

    #title
    st.markdown("# Analysis of Data Science Related Salary Around the World")
    st.markdown("### DSCI 551 Project (NoSQL)")
    st.markdown("---")

    if "nosql_current_stage" not in st.session_state:
        st.session_state.nosql_current_stage = 0

    stages = ["Data Upload", "Function Pipeline Design", "Results Display"]
    current_stage = st.session_state.nosql_current_stage
    progress_value = (current_stage + 1) / len(stages)

    st.progress(progress_value)
    st.markdown(f"**Stage {current_stage + 1} of {len(stages)}: {stages[current_stage]}**")

    # run it again!!!!!
    if current_stage == 0:
        step1()
    elif current_stage == 1:
        step2()
    elif current_stage == 2:
        step3()


if __name__ == "__main__":
    main()
