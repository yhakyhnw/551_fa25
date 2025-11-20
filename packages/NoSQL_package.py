

# Safe optional import for tqdm
try:
    from tqdm.auto import tqdm
except ImportError:  # fallback if tqdm is not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Helper functions for coercing string values to numbers
def _coerce_scalar(value):
    """
    Convert numeric-looking strings to int/float.
    Leave everything else unchanged.
    """
    if not isinstance(value, str):
        return value

    s = value.strip()
    if not s:
        return value

    # Try int (handles +123, -123, 0, etc.)
    sign_idx = 0
    if s[0] in "+-":
        sign_idx = 1

    if sign_idx < len(s) and s[sign_idx:].isdigit():
        try:
            return int(s)
        except ValueError:
            return value

    # Try float (handles 12.34, -0.5, 1e6, etc.)
    try:
        float_val = float(s)
        return float_val
    except ValueError:
        return value


def _coerce_doc(doc):
    """
    Apply _coerce_scalar to every value in a dict.
    """
    if not isinstance(doc, dict):
        return doc
    return {key: _coerce_scalar(val) for key, val in doc.items()}

class NoSql:
    def __init__(self, data = None):
        
        # dummy data 
        if data is None:
            self.data = [doc.copy() for doc in DUMMY_COUNTRY_LANGUAGE_DATA]
            return
        
        # import data
        if isinstance(data, dict):
            self.data = [data]
        
        # chunk data
        elif isinstance(data, list):
            if all(isinstance(d, dict) for d in data):
                self.data = data
            elif all(isinstance(d, list) for d in data):
                self.data = data
            elif all(isinstance(d, NoSql) for d in data):
                self.data = data
            else:
                raise TypeError("Data list elements must be dicts, lists of dicts, or NoSql chunks")
        else:
            raise TypeError("Data must be a dict or a compatible list (dicts, lists of dicts, or NoSql chunks)")

    def __repr__(self):
        return f"NoSql(n_docs={len(self.data)})"

    def __getitem__(self, keys=None):
        if keys is None:
            return NoSql(self.data)

        if isinstance(keys, str):
            fields = [keys]
        elif isinstance(keys, (list, tuple)):
            fields = list(keys)
        else:
            raise TypeError("Invalid key type for projection")
        return self.project({k: 1 for k in fields})

    def project(self, keys) -> "NoSql":
        docs = self._ensure_flat_docs(self.data)
        if not isinstance(keys, dict):
            raise TypeError("Projection keys must be a dict")
        if not keys:
            return NoSql(docs)
        if len(docs) == 0:
            return NoSql([])

        vals = set(keys.values())

        if not vals.issubset({0,1}):
            raise ValueError("Projectio values must be 0 or 1")

        is_inclusive = 1 in vals and 0 not in vals
        is_exclusive = 0 in vals and 1 not in vals

        if not (is_inclusive or is_exclusive):
            raise ValueError("Do not mix inclusive and exclusive in one proejction")

        projected = []
        if is_inclusive:
            for doc in docs:
                subset = {k: v for k, v in doc.items() if k in list(keys.keys())}
                projected.append(subset)
        else:
            for doc in docs:
                subset = {k: v for k, v in doc.items() if k not in list(keys.keys())}
                projected.append(subset)
        return NoSql(projected)
    
    @classmethod
    def read_json(cls, filepath: str, chunk_size: int | None = None):
        if chunk_size is None:
            return cls._read_json_all(filepath)
        
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        
        return cls._read_json_chunks(filepath, chunk_size)

    @classmethod
    def read_dummy(cls, chunk_size: int | None = None):

        # fresh copy of dummy data
        data = [doc.copy() for doc in DUMMY_COUNTRY_LANGUAGE_DATA]

        if chunk_size is None:
            return cls(data)

        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        def _gen():
            for i in range(0, len(data), chunk_size):
                yield cls(data[i : i + chunk_size])

        return _gen()

    @classmethod
    def _read_json_all(cls, filepath: str) -> "NoSql":
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Case 1: JSON array of dicts
        if text.startswith('['):
            arr, rest = cls._parse_list(text)
            arr = [_coerce_doc(d) for d in arr]
            if rest.strip():
                raise ValueError("Extra data after JSON array")
            if not all(isinstance(d, dict) for d in arr):
                raise ValueError("JSON array must contain only objects (dicts)")
            return cls(arr)
        
        # Case 2: newline-delimited JSON
        docs = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            if not line.startswith('{'):
                raise ValueError(f"Line {i}: Expected JSON object in NDJSON format")
            obj, rest = cls._parse_object(line)
            if rest.strip():
                raise ValueError(f"Line {i}: Extra content after JSON object")
            docs.append(_coerce_doc(obj))

        if not docs:
            raise ValueError("File is empty or not valid NDJSON/array format")

        return cls(docs)

    @classmethod
    def _read_json_chunks(cls, filepath: str, chunk_size: int) -> "NoSql":
        with open(filepath, "r", encoding="utf-8") as f:
            start = f.read(1024)
            if not start:
                raise ValueError("File is empty")
            stripped = start.lstrip()
            first = stripped[0]

        # Case 1: JSON array of dicts
        if first == '[':
            all_ns = cls._read_json_all(filepath)   # NoSql([...])
            data = all_ns.data
            total_docs = len(data)
            for start_index in tqdm(
                range(0, total_docs, chunk_size),
                desc="Chunking JSON array",
                unit="chunk",
            ):
                yield NoSql(data[start_index:start_index + chunk_size])
        
        # Case 2: newline-delimited JSON
        else:
            batch = []
            with open(filepath, "r", encoding="utf-8") as f:
                for i, line in enumerate(tqdm(f, desc="Reading JSON (NDJSON)", unit="line"), 1):
                    line = line.strip()
                    if not line:
                        continue
                    if not line.startswith('{'):
                        raise ValueError(f"Line {i}: Expected JSON object in NDJSON format")
                    obj, rest = cls._parse_object(line)
                    if rest.strip():
                        raise ValueError(f"Line {i}: Extra content after JSON object")
                    batch.append(_coerce_doc(obj))
                    if len(batch) == chunk_size:
                        yield NoSql(batch)
                        batch = []
            if batch:
                yield NoSql(batch)

    @staticmethod
    def _parse_string(s):
        s = s.lstrip()
        if not s or s[0] != '"':
            raise ValueError((f'Expected string, got: {repr(s[:20])}'))
        s = s[1:]
        end_pos = s.find('"')
        if end_pos == -1:
            raise ValueError("Undetermined string")
        value = s[:end_pos]
        rest = s[end_pos+1:]
        return value, rest
    
    @staticmethod
    def _parse_number(s):
        s = s.lstrip()
        if not s:
            raise ValueError("Expected number but got empty string")
        chs = ''
        is_float = False
        i = 0
        for ch in s:
            if ch.isdigit() or ch in ('.', '-'):
                if ch == '.':
                    is_float = True
                chs += ch
                i += 1
            else:
                break
        if not chs or chs in ('-', '.'):
            raise ValueError(f"Invalid number at: {repr(s[:20])}")
        value = float(chs) if is_float else int(chs)
        rest = s[i:]
        return value, rest

    @classmethod
    def _parse_list(cls, s):
        s = s.lstrip()
        if not s or s[0] != '[':
            raise ValueError(f"Expected '[' at start of list, got: {repr(s[:20])}")
        s = s[1:]
        s = s.lstrip()

        if s and s[0] == ']':
            return [], s[1:]

        out = []
        while True:
            val, s = cls._parse_value(s)
            out.append(val)

            s = s.lstrip()
            if not s:
                raise ValueError("Unterminated list")
            if s[0] == ']':
                return out, s[1:]
            if s[0] != ',':
                raise ValueError(f"Expected ',' or ']' in list, got: {repr(s[:10])}")
            s = s[1:]

    @classmethod
    def _parse_object(cls, s):
        s = s.lstrip()
        if not s or s[0] != '{':
            raise ValueError(f"Expected '{{' at start of object, got: {repr(s[:20])}")
        s = s[1:]
        s = s.lstrip()
        obj = {}

        if s and s[0] == '}':
            return obj, s[1:]

        while True:
            s = s.lstrip()
            if not s:
                raise ValueError('Expecting "}" but reached end')

            key, s = cls._parse_string(s)

            s = s.lstrip()
            if not s or s[0] != ':':
                raise ValueError(f"Expected ':' after key {key!r}, got: {repr(s[:10])}")
            s = s[1:]

            val, s = cls._parse_value(s)
            obj[key] = val

            s = s.lstrip()
            if not s:
                raise ValueError("Unterminated object")
            if s[0] == '}':
                return obj, s[1:]
            if s[0] != ',':
                raise ValueError(f"Expected ',' or '}}' in object, got: {repr(s[:10])}")
            s = s[1:]

    @classmethod
    def _parse_value(cls, s):
        s = s.lstrip()
        if not s:
            raise ValueError("Unexpected end while parsing value")

        ch = s[0]
        if ch == '"':
            return cls._parse_string(s)
        if ch == '{':
            return cls._parse_object(s)
        if ch == '[':
            return cls._parse_list(s)
        if s.startswith('true'):
            return True, s[4:]
        if s.startswith('false'):
            return False, s[5:]
        return cls._parse_number(s)

    def join(self, from_field: "NoSql", local_field: str, foreign_field: str, as_field: str) -> "NoSql":
        
        if not isinstance(from_field, NoSql):
            raise TypeError("from_field must be a NoSql object")
        
        for name, val in (("local_field", local_field), ("foreign_field", foreign_field), ("as_field", as_field)):
            if not isinstance(val, str):
                raise TypeError(f"{name} must be a string")
            if name == "as_field" and not val:
                raise ValueError("as_field must be a non-empty string")

        if not self.data:
            return NoSql([])

        left_docs = self._ensure_flat_docs(self.data)
        right_docs = self._ensure_flat_docs(from_field.data)

        def _get_value(doc, path):
            fields = path.split('.')
            cur = doc
            for field in fields:
                if isinstance(cur, dict) and field in cur: # Only consider key, value pair case
                    cur = cur[field]
                else:
                    return None
            
            if isinstance(cur, (list, tuple, dict)):
                return None
            
            return cur
        
        right_index = {}
        for rdoc in right_docs:
            foreign_val = _get_value(rdoc, foreign_field)
            if foreign_val is None:
                continue
            right_index[foreign_val] = rdoc

        joined = []
        for ldoc in left_docs:
            local_val = _get_value(ldoc, local_field)
            new_doc = dict(ldoc)

            match = None
            if local_val is not None:
                match = right_index.get(local_val)

            if match is None:
                # No match: keep an explicit None for the joined field
                new_doc[as_field] = None
            else:
                new_doc[as_field] = dict(match)
                for k, v in match.items():
                    new_doc[f"{as_field}.{k}"] = v

            joined.append(new_doc)

        return NoSql(joined)


    @staticmethod
    def _ensure_flat_docs(docs):
        if not docs:
            return []

        # list of NoSql chunks
        if all(isinstance(chunk, NoSql) for chunk in docs):
            flattened = []
            for chunk in docs:
                for d in chunk.data:
                    if not isinstance(d, dict):
                        raise TypeError("Expected dict inside NoSql chunks")
                    flattened.append(d)
            return flattened

        # list of lists of dicts
        if all(isinstance(chunk, list) for chunk in docs):
            flattened = []
            for chunk in docs:
                for d in chunk:
                    if not isinstance(d, dict):
                        raise TypeError("Expected dict inside nested lists")
                    flattened.append(d)
            return flattened

        # list of dicts
        if not all(isinstance(d, dict) for d in docs):
            raise TypeError("Docs must be dicts or nested lists of dicts or NoSql chunks")

        return docs


    def group_by(self, keys):
        docs = self._ensure_flat_docs(self.data)

        # Normalize keys
        if isinstance(keys, str):
            group_keys = [keys]
        elif isinstance(keys, (list, tuple)):
            group_keys = list(keys)
        else:
            raise TypeError("group_by keys must be str or list of str")

        groups = {}
        for doc in docs:
            key_tuple = tuple(doc.get(k) for k in group_keys)
            groups.setdefault(key_tuple, []).append(doc)

        out = []
        for key_vals, bucket in groups.items():
            group_doc = {k: v for k, v in zip(group_keys, key_vals)}
            group_doc["_group"] = bucket
            out.append(group_doc)

        return out


    def aggregate(self, group_keys, agg_spec):
        docs = self._ensure_flat_docs(self.data)

        # Normalize group keys
        if isinstance(group_keys, str):
            group_keys = [group_keys]
        elif isinstance(group_keys, (list, tuple)):
            group_keys = list(group_keys)
        else:
            raise TypeError("group_keys must be a string or list of strings")

        if not isinstance(agg_spec, dict):
            raise TypeError("agg_spec must be a dict")

        groups = {}
        for doc in docs:
            key_tuple = tuple(doc.get(k) for k in group_keys)
            groups.setdefault(key_tuple, []).append(doc)

        def _apply_agg(values, fn):
            nums = [v for v in values if isinstance(v, (int, float))]
            if not nums:
                return None

            if fn == "sum": return sum(nums)
            if fn in ("avg", "mean"): return sum(nums) / len(nums)
            if fn == "min": return min(nums)
            if fn == "max": return max(nums)
            if fn == "count": return len(nums)
            raise ValueError(f"Unknown aggregation: {fn}")

        result = []

        for key_vals, bucket in groups.items():
            row = {k: v for k, v in zip(group_keys, key_vals)}

            for field, funcs in agg_spec.items():

                # Normalize funcs
                if isinstance(funcs, str):
                    funcs = [funcs]

                # Special case "*" = count docs
                if field == "*":
                    for f in funcs:
                        if f != "count":
                            raise ValueError("'*' only supports count")
                        row["count"] = len(bucket)
                    continue

                # Regular numeric aggregator
                values = [d.get(field) for d in bucket]
                for f in funcs:
                    val = _apply_agg(values, f)
                    row[f"{field}_{f}"] = val

            result.append(row)

        return result

    @classmethod
    def filter_auto(cls, source, query):

        if isinstance(source, NoSql):
            return source.filter(query)

        def _generator():
            for chunk in source:
                if not isinstance(chunk, NoSql):
                    raise TypeError("filter_auto iterable must yield NoSql objects")
                yield chunk.filter(query)

        return _generator()
    
    def single_filter(self, key, condition, chain = False):
        if not isinstance(key, str):
            raise TypeError("single_filter key must be a string")
        if not key:
            raise ValueError("single_filter key must be a non-empty string")

        docs = self._ensure_flat_docs(self.data)
        if not docs:
            if chain:
                return set()
            return NoSql([])

        if isinstance(condition, dict):
            cond_dict = condition
        else:
            cond_dict = {"$eq": condition}

        def _match_op(doc_val, op, val):
            if op == "$eq":
                return doc_val == val
            if op == "$ne":
                return doc_val != val

            if op == "$gt":
                try:
                    return doc_val > val
                except TypeError:
                    return False
            if op == "$gte":
                try:
                    return doc_val >= val
                except TypeError:
                    return False
            if op == "$lt":
                try:
                    return doc_val < val
                except TypeError:
                    return False
            if op == "$lte":
                try:
                    return doc_val <= val
                except TypeError:
                    return False

            if op == "$in":
                if not isinstance(val, (list, tuple, set)):
                    raise TypeError("$in expects a list/tuple/set of values")
                return doc_val in val
            if op == "$nin":
                if not isinstance(val, (list, tuple, set)):
                    raise TypeError("$nin expects a list/tuple/set of values")
                return doc_val not in val

            raise ValueError(f"Unsupported operator in single_filter: {op}")

        matched_indexes = set()

        for i, doc in enumerate(docs):
            doc_val = doc.get(key, None)

            ok = True
            for op, val in cond_dict.items():
                if not _match_op(doc_val, op, val):
                    ok = False
                    break

            if ok:
                matched_indexes.add(i)

        if chain:
            return matched_indexes

        out_docs = []
        for i in sorted(matched_indexes):
            out_docs.append(docs[i])

        return NoSql(out_docs)

    def and_or(self, operator, index_sets):
        if not isinstance(operator, str):
            raise TypeError("and_or operator must be a string")

        op = operator.strip().upper()
        if op not in {"AND", "OR"}:
            raise ValueError("and_or operator must be 'AND' or 'OR'")

        normalized_sets = []
        for idx_set in index_sets:
            if isinstance(idx_set, set):
                normalized_sets.append(idx_set)
            elif isinstance(idx_set, (list, tuple)):
                normalized_sets.append(set(idx_set))
            else:
                raise TypeError("and_or expects each element in index_sets to be a set, list, or tuple")

        if not normalized_sets:
            return set()

        combined = normalized_sets[0].copy()

        for s in normalized_sets[1:]:
            if op == "AND":
                combined = combined.intersection(s)
            else:
                combined = combined.union(s)

        return combined

    def find_arg_parser(self, query):

        if not isinstance(query, dict):
            raise TypeError("parser expects a dict query")

        if not query:
            raise ValueError("parser received an empty query dict")

        def _parse_node(node):
            if not isinstance(node, dict):
                raise TypeError("Each query node must be a dict")

            has_and = "$and" in node
            has_or = "$or" in node

            # Logical node: exactly one of $and / $or at this level
            if has_and or has_or:
                if has_and and has_or:
                    raise ValueError("A query level cannot contain both $and and $or")

                # Disallow mixing logical operator with other keys at same level
                if len(node) != 1:
                    raise ValueError("When using $and or $or at a level, no other keys are allowed")

                if has_and:
                    op_name = "AND"
                    subqueries = node["$and"]
                else:
                    op_name = "OR"
                    subqueries = node["$or"]

                if not isinstance(subqueries, list) or not subqueries:
                    raise ValueError(f"{op_name} must be given a non-empty list of subqueries")

                children = []
                for subquery in subqueries:
                    child_tree = _parse_node(subquery)
                    children.append(child_tree)

                return {"type": "op", "op": op_name, "children": children}

            # Field-level node: implicit AND of all field conditions in this dict
            items = list(node.items())
            if not items:
                raise ValueError("Empty field-level query")

            # Single field: just a leaf
            if len(items) == 1:
                field, condition = items[0]
                return {"type": "leaf", "key": field, "condition": condition}

            # Multiple fields: AND of leaves
            children = []
            for field, condition in items:
                child_leaf = {"type": "leaf", "key": field, "condition": condition}
                children.append(child_leaf)

            return {"type": "op", "op": "AND", "children": children}

        return _parse_node(query)


    def filter(self, query) -> "NoSql":

        docs = self._ensure_flat_docs(self.data)
        if not docs:
            return NoSql([])

        tree = self.find_arg_parser(query)

        def _eval_node(node):
            if not isinstance(node, dict):
                raise TypeError("Internal parser node must be a dict")

            node_type = node.get("type")

            if node_type == "leaf":
                key = node.get("key")
                condition = node.get("condition")
                return self.single_filter(key, condition, chain = True)

            if node_type == "op":
                op_name = node.get("op")
                children = node.get("children", [])
                if not isinstance(children, list) or not children:
                    raise ValueError("Operator node must have a non-empty children list")

                child_sets = []
                for child in children:
                    child_index_set = _eval_node(child)
                    if not isinstance(child_index_set, set):
                        raise TypeError("Child evaluation must return a set of indices")
                    child_sets.append(child_index_set)

                return self.and_or(op_name, child_sets)

            raise ValueError(f"Unknown node type in query tree: {node_type!r}")

        final_index_set = _eval_node(tree)

        if not final_index_set:
            return NoSql([])

        out_docs = []
        for index in sorted(final_index_set):
            out_docs.append(docs[index])

        return NoSql(out_docs)


def pretty_print_nosql(ns_obj, indent_spaces: int = 4, dp_lim: int | None = None):

    if not isinstance(ns_obj, NoSql):
        raise TypeError("pretty_print_nosql expects a NoSql object")

    def _pretty_value(value, indent_level):
        base_indent = " " * indent_level
        inner_indent = " " * (indent_level + indent_spaces)

        if isinstance(value, dict):
            items = list(value.items())
            if not items:
                return base_indent + "{}"

            lines = [base_indent + "{"]
            for idx, (key, val) in enumerate(items):
                is_last = (idx == len(items) - 1)

                if isinstance(val, (dict, list)):
                    lines.append(f"{inner_indent}{key}:")
                    nested_block = _pretty_value(val, indent_level + indent_spaces)
                    nested_lines = nested_block.split("\n")
                    # Add comma to last nested line if not last field
                    if not is_last and nested_lines:
                        nested_lines[-1] = nested_lines[-1] + ","
                    lines.extend(nested_lines)
                else:
                    line = f"{inner_indent}{key}: {repr(val)}"
                    if not is_last:
                        line += ","
                    lines.append(line)

            lines.append(base_indent + "}")
            return "\n".join(lines)

        if isinstance(value, list):
            if not value:
                return base_indent + "[]"

            lines = [base_indent + "["]
            for idx, item in enumerate(value):
                is_last = (idx == len(value) - 1)

                if isinstance(item, (dict, list)):
                    nested_block = _pretty_value(item, indent_level + indent_spaces)
                    nested_lines = nested_block.split("\n")
                    if not is_last and nested_lines:
                        nested_lines[-1] = nested_lines[-1] + ","
                    lines.extend(nested_lines)
                else:
                    line = inner_indent + repr(item)
                    if not is_last:
                        line += ","
                    lines.append(line)

            lines.append(base_indent + "]")
            return "\n".join(lines)

        return base_indent + repr(value)

    # Determine which docs to print
    if ns_obj.data and all(isinstance(chunk, NoSql) for chunk in ns_obj.data):
        first_chunk = ns_obj.data[0]
        if not isinstance(first_chunk, NoSql):
            raise TypeError("First chunk is not a NoSql object")
        if not isinstance(first_chunk.data, list) or not all(isinstance(d, dict) for d in first_chunk.data):
            raise TypeError("First chunk's data is not a list of dicts")
        docs_to_iter = first_chunk.data
    else:
        docs_to_iter = ns_obj.data

    for doc_index, doc in enumerate(docs_to_iter, 1):
        if dp_lim is not None and doc_index > dp_lim:
            break
        pretty_doc = _pretty_value(doc, 0)
        print(pretty_doc)
        print()


# dummy data for demo
DUMMY_COUNTRY_LANGUAGE_DATA = [
    { "code": "IQ", "country": "Iraq", "language": "Arabic" },
    { "code": "CY", "country": "Cyprus", "language": "Greek" },
    { "code": "RS", "country": "Serbia", "language": "Serbian" },
    { "code": "CL", "country": "Chile", "language": "Spanish" },
    { "code": "LV", "country": "Latvia", "language": "Latvian" },
    { "code": "IE", "country": "Ireland", "language": "English" },
    { "code": "NG", "country": "Nigeria", "language": "English" },
    { "code": "US", "country": "United States", "language": "English" },
    { "code": "GH", "country": "Ghana", "language": "English" },
    { "code": "EC", "country": "Ecuador", "language": "Spanish" },
    { "code": "OM", "country": "Oman", "language": "Arabic" },
    { "code": "PE", "country": "Peru", "language": "Spanish" },
    { "code": "MK", "country": "North Macedonia", "language": "Macedonian" },
    { "code": "PH", "country": "Philippines", "language": "Filipino" },
    { "code": "TR", "country": "Turkey", "language": "Turkish" },
    { "code": "AU", "country": "Australia", "language": "English" },
    { "code": "QA", "country": "Qatar", "language": "Arabic" },
    { "code": "FI", "country": "Finland", "language": "Finnish" },
    { "code": "CZ", "country": "Czech Republic", "language": "Czech" },
    { "code": "LS", "country": "Lesotho", "language": "Sesotho" },
    { "code": "CD", "country": "Democratic Republic of the Congo", "language": "French" },
    { "code": "HN", "country": "Honduras", "language": "Spanish" },
    { "code": "CN", "country": "China", "language": "Mandarin Chinese" },
    { "code": "RU", "country": "Russia", "language": "Russian" },
    { "code": "VE", "country": "Venezuela", "language": "Spanish" },
    { "code": "NL", "country": "Netherlands", "language": "Dutch" },
    { "code": "PR", "country": "Puerto Rico", "language": "Spanish" },
    { "code": "PA", "country": "Panama", "language": "Spanish" },
    { "code": "MX", "country": "Mexico", "language": "Spanish" },
    { "code": "SI", "country": "Slovenia", "language": "Slovene" },
    { "code": "PK", "country": "Pakistan", "language": "Urdu" },
    { "code": "ML", "country": "Mali", "language": "French" },
    { "code": "KE", "country": "Kenya", "language": "Swahili" },
    { "code": "VN", "country": "Vietnam", "language": "Vietnamese" },
    { "code": "IT", "country": "Italy", "language": "Italian" },
    { "code": "DE", "country": "Germany", "language": "German" },
    { "code": "LT", "country": "Lithuania", "language": "Lithuanian" },
    { "code": "GT", "country": "Guatemala", "language": "Spanish" },
    { "code": "NO", "country": "Norway", "language": "Norwegian" },
    { "code": "DO", "country": "Dominican Republic", "language": "Spanish" },
    { "code": "BR", "country": "Brazil", "language": "Portuguese" },
    { "code": "PT", "country": "Portugal", "language": "Portuguese" },
    { "code": "JP", "country": "Japan", "language": "Japanese" },
    { "code": "RO", "country": "Romania", "language": "Romanian" },
    { "code": "AE", "country": "United Arab Emirates", "language": "Arabic" },
    { "code": "HU", "country": "Hungary", "language": "Hungarian" },
    { "code": "SA", "country": "Saudi Arabia", "language": "Arabic" },
    { "code": "XK", "country": "Kosovo", "language": "Albanian" },
    { "code": "BA", "country": "Bosnia and Herzegovina", "language": "Bosnian" },
    { "code": "SK", "country": "Slovakia", "language": "Slovak" },
    { "code": "TW", "country": "Taiwan", "language": "Mandarin Chinese" },
    { "code": "AS", "country": "American Samoa", "language": "Samoan" },
    { "code": "MD", "country": "Moldova", "language": "Romanian" },
    { "code": "ZM", "country": "Zambia", "language": "English" },
    { "code": "AR", "country": "Argentina", "language": "Spanish" },
    { "code": "TH", "country": "Thailand", "language": "Thai" },
    { "code": "DK", "country": "Denmark", "language": "Danish" },
    { "code": "UA", "country": "Ukraine", "language": "Ukrainian" },
    { "code": "ID", "country": "Indonesia", "language": "Indonesian" },
    { "code": "GI", "country": "Gibraltar", "language": "English" },
    { "code": "IN", "country": "India", "language": "Hindi" },
    { "code": "IR", "country": "Iran", "language": "Persian" },
    { "code": "CO", "country": "Colombia", "language": "Spanish" },
    { "code": "AT", "country": "Austria", "language": "German" },
    { "code": "CH", "country": "Switzerland", "language": "German" },
    { "code": "AM", "country": "Armenia", "language": "Armenian" },
    { "code": "GR", "country": "Greece", "language": "Greek" },
    { "code": "ZA", "country": "South Africa", "language": "Zulu" },
    { "code": "MU", "country": "Mauritius", "language": "English" },
    { "code": "MT", "country": "Malta", "language": "Maltese" },
    { "code": "MY", "country": "Malaysia", "language": "Malay" },
    { "code": "EE", "country": "Estonia", "language": "Estonian" },
    { "code": "SV", "country": "El Salvador", "language": "Spanish" },
    { "code": "CR", "country": "Costa Rica", "language": "Spanish" },
    { "code": "HR", "country": "Croatia", "language": "Croatian" },
    { "code": "JM", "country": "Jamaica", "language": "English" },
    { "code": "SE", "country": "Sweden", "language": "Swedish" },
    { "code": "FR", "country": "France", "language": "French" },
    { "code": "BE", "country": "Belgium", "language": "Dutch" },
    { "code": "HK", "country": "Hong Kong", "language": "Cantonese" },
    { "code": "NZ", "country": "New Zealand", "language": "English" },
    { "code": "LU", "country": "Luxembourg", "language": "Luxembourgish" },
    { "code": "CF", "country": "Central African Republic", "language": "French" },
    { "code": "DZ", "country": "Algeria", "language": "Arabic" },
    { "code": "CA", "country": "Canada", "language": "English" },
    { "code": "AD", "country": "Andorra", "language": "Catalan" },
    { "code": "BS", "country": "Bahamas", "language": "English" },
    { "code": "ES", "country": "Spain", "language": "Spanish" },
    { "code": "LB", "country": "Lebanon", "language": "Arabic" },
    { "code": "IL", "country": "Israel", "language": "Hebrew" },
    { "code": "GB", "country": "United Kingdom", "language": "English" },
    { "code": "BG", "country": "Bulgaria", "language": "Bulgarian" },
    { "code": "EG", "country": "Egypt", "language": "Arabic" },
    { "code": "SG", "country": "Singapore", "language": "English" },
    { "code": "JO", "country": "Jordan", "language": "Arabic" },
    { "code": "KR", "country": "South Korea", "language": "Korean" },
    { "code": "PL", "country": "Poland", "language": "Polish" }]