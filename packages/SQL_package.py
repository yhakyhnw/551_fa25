from __future__ import annotations
from typing import Optional, Dict, Tuple, List, Iterable, Any, Union
import os
import re

"""
Helper functions 
"""
def _get_header(file, encoding, delimiter) -> list:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Error: '{file}' does not exist")

    try:
        with open(file, "r", encoding = encoding) as f_in:
            first_line = f_in.readline().strip()
    except LookupError:
        raise ValueError(f"Error in PSO.parser: unknown encoding '{encoding}'")

    return first_line.split(delimiter)

def _arg_checker(*args):
    if not args:
        return False, "empty argument"
    if len(args) > 1:
        return False, "multiple arguments"
    s = args[0]
    if not isinstance(s, str):
        return False, "non-string argument"
    if not s.strip():
        return False, "empty string argument"
    return True, "ok"

def _is_numeric(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _col_is_numeric(col_vals) -> bool:
    is_numeric = False
    for val in col_vals:
        if val is None:
            continue
        if _is_numeric(val):
            is_numeric = True
        else:
            return False
    return is_numeric

def _strip_full_parentheses(args):
    args = args.strip()
    while args.startswith("(") and args.endswith(")"):
        depth = 0
        balanced = True
        for i, char in enumerate(args):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and i != len(args) - 1:
                    balanced = False
                    break
        if balanced:
            args = args[1:-1].strip()
        else:
            break
    return args

"""
Group By class
"""
class GroupBy:
    VALID_OPS = {"sum", "mean", "min", "max", "count"}

    def __init__(self, groups: dict, data: dict, by: list):
        self.groups = groups
        self.data = data
        self.by = by

    def __repr__(self):
        return f"GroupBy(n_groups={len(self.groups)}, by={self.by})"

    def __str__(self):
        return f"GroupBy(n_groups={len(self.groups)}, by={self.by})"

    def agg(self, agg: Union[str, List[str], Dict[str, Union[str, List[str]]]] = None) -> PSO:
        if agg is None:
            raise ValueError("Error in .agg(agg): 'agg' is required")

        spec = {}
        if isinstance(agg, str):
            spec['*'] = [agg]
        elif isinstance(agg, list):
            spec['*'] = list(agg)
        elif isinstance(agg, dict):
            for key, val in agg.items():
                if isinstance(val, str):
                    ops = [val]
                elif isinstance(val, list):
                    ops = val[:]
                else:
                    raise ValueError(f"Error in .agg(agg): invalid ops for key '{key}'")
                spec[key] = ops
        else:
            raise ValueError("Error in .agg(agg): 'agg' must be a string, list, or dict")

        # Validate ops
        for ops in spec.values():
            for op in ops:
                if op not in self.VALID_OPS:
                    raise ValueError(f"Error in .agg(agg): '{op}' is not a valid operator")

        all_cols = list(self.data.keys())
        by_set = set(self.by)
        
        non_by_cols = [col for col in all_cols if col not in by_set]

        numeric_cols = [col for col in non_by_cols if _col_is_numeric(self.data[col])]

        final_plan = {}

        # Explicit column request
        for key, ops in spec.items():
            if key == '*':
                continue
            if key not in self.data:
                raise ValueError(f"Error in .agg(agg): column '{key}' not found")

            # Enforce numeric-only aggregation for non-count operations
            is_numeric_col = _col_is_numeric(self.data[key])

            final_plan.setdefault(key, [])
            for op in ops:
                if op != "count" and not is_numeric_col:
                    raise ValueError(
                        f"Error in .agg(agg): column '{key}' is not numeric and cannot be aggregated with '{op}'"
                    )
                if op not in final_plan[key]:
                    final_plan[key].append(op)
                    
        # All numeric columns
        if '*' in spec:
            for col in numeric_cols:
                final_plan.setdefault(col, [])
                for op in spec['*']:
                    if op not in final_plan[col]:
                        final_plan[col].append(op)

        out_header = []
        out_header.extend(self.by)

        agg_cols = []
        requested_count = False

        for col, ops in final_plan.items():
            if "count" in ops:
                requested_count = True

            for op in ops:
                if op == 'count':
                    continue

                out_name = f"{col}_{op}"
                agg_cols.append((out_name, col, op))
                out_header.append(out_name)

        only_count_requested = ("*" in spec and spec["*"] == ["count"]) and (len(final_plan) == 0)
        if only_count_requested:
            requested_count = True

        if requested_count and "count" not in out_header:
            out_header.append("count")

        out_cols = {header: [] for header in out_header}

        def do_aggregate(op: str, series: List[Any]) -> Any:
            vals = [v for v in series if v is not None]

            if op == 'count':
                return None

            if not vals:
                return None

            if op in ('sum', 'mean'):
                if not all(_is_numeric(v) for v in vals):
                    return None
                return sum(vals) if op == 'sum' else sum(vals)/len(vals)

            if op == 'min':
                try: 
                    return min(vals)
                except TypeError:
                    return None
            if op == 'max':
                try:
                    return max(vals)
                except TypeError:
                    return None
            else:
                return None

        for key_tuple, idxs in sorted(self.groups.items(), key = lambda kv: kv[0]):
            for bname, bval in zip(self.by, key_tuple):
                out_cols[bname].append(bval)

            for out_name, col, op in agg_cols:
                series = [self.data[col][i] for i in idxs]
                out_cols[out_name].append(do_aggregate(op, series))

            if requested_count:
                out_cols['count'].append(len(idxs))

        output = {header: tuple(out_cols[header]) for header in out_header}
        return PSO.parser(data = output, header = out_header)

"""
SQL class (Called PSO)
"""
class PSO:
    def __init__(self, data: dict, header: list,
                 encoding: str = "UTF-8", delimiter: str = ","):
        self.data = data
        self.header = header
        self.encoding = encoding
        self.delimiter = delimiter

    """
    Parser
    """
    @classmethod
    def parser(cls, file: str = None, data: dict = None,
                   header: Optional[list] = None, encoding: str = "UTF-8", delimiter: str = ",") -> "PSO":
        
        # 1. dummy data if no args
        if file is None and data is None:
            dummy = {"Mfg": ("Volkswagen", "Lexus", "Subaru", "Cadillac", "Toyota",
                             "Land Rover", "Mazda", "Ram", "Chrysler", "GMC",
                             "Volvo", "Audi", "Chevrolet", "Tesla", "Hyundai",
                             "Ford", "Porsche", "Acura", "Nissan", "Kia",
                             "Jeep", "BMW", "Dodge", "Mercedes-Benz", "Honda"),
                    "Country": ("Germany", "Japan", "Japan", "USA", "Japan",
                                "UK", "Japan", "USA", "USA", "USA",
                                "Sweden", "Germany", "USA", "USA", "South Korea",
                                "USA", "Germany", "Japan", "Japan", "South Korea",
                                "USA", "Germany", "USA", "Germany", "Japan"),
                    "Founded": (1937, 1989, 1953, 1902, 1937,
                                1948, 1920, 2010, 1925, 1911,
                                1927, 1909, 1911, 2003, 1967,
                                1903, 1931, 1986, 1933, 1944,
                                1941, 1916, 1900, 1926, 1948)}
            return cls(dummy, list(dummy.keys()), encoding = encoding, delimiter = delimiter)

        # 2. reject double file and data
        if file is not None and data is not None:
            raise ValueError("Error in PSO.sql_parser: must have exactly 1 of either data or file")

        # 3. data mode
        if data is not None:
            return cls(data, list(data.keys()), encoding = encoding, delimiter = delimiter)

        # 4. file mode
        if header is None:
            header = _get_header(file, encoding, delimiter)
        expected = len(header)
        cols = {h: [] for h in header}
        num_pat = re.compile(r'^-?\d+(?:\.\d+)?$')

        if file is None or expected == 0:
            return cls({header_ind: tuple() for header_ind in header}, header, encoding = encoding, delimiter = delimiter)

        try:
            f_in = open(file, "r", encoding = encoding)
        except LookupError:
            raise ValueError(f"Error in PSO.parser: unknown encoding '{encoding}'")
        
        with f_in:
            _ = f_in.readline()
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(delimiter)

                # skip row if too many columns
                if len(parts) > expected:
                    continue

                # pad with None if too few columns
                if len(parts) < expected:
                    parts.extend([None] * (expected - len(parts)))

                # cast values
                for h, raw in zip(header, parts):
                    if raw is None:
                        casted = None
                    else:
                        s = raw.strip()
                        if s == "":
                            casted = None
                        elif num_pat.match(s):
                            casted = float(s) if "." in s else int(s)
                        else:
                            casted = s
                    cols[h].append(casted)

        data_dict = {header: tuple(vals) for header, vals in cols.items()}

        return cls(data_dict, header, encoding = encoding, delimiter = delimiter)
            
    def __repr__(self):
        if not self.data:
            return "Empty PSO()"
    
        cols = list(self.data.keys())
        num_rows = len(next(iter(self.data.values())))
        display_limit = 20
    
        rows = []
        for row_index in range(num_rows):
            row = []
            for col_name in cols:
                val = self.data[col_name][row_index]
                if val is not None:
                    row.append(str(val))
                else:
                    row.append("")
            rows.append(row)
    
        if num_rows > display_limit:
            limited_rows = []
            for row in rows[:display_limit]:
                limited_rows.append(row)

            dots_row = []
            for _ in cols:
                dots_row.append("...")
            
            limited_rows.append(dots_row)
            rows = limited_rows
    
        col_widths = {}
        for col in cols:
            header_len = len(col)

            # compute max_val_len with expanded loop
            max_val_len = 0
            for value in self.data[col]:
                if value is None:
                    val_str = ""
                else:
                    val_str = str(value)

                if len(val_str) > max_val_len:
                    max_val_len = len(val_str)

            if header_len >= max_val_len and header_len >= 4:
                col_widths[col] = header_len
            elif max_val_len >= header_len and max_val_len >= 4:
                col_widths[col] = max_val_len
            else:
                col_widths[col] = 4
    
        header_parts = []
        for col in cols:
            header_parts.append(col.ljust(col_widths[col]))

        header_str = " | ".join(header_parts)
        
        sep_parts = []
        for col in cols:
            sep_parts.append("-" * col_widths[col])

        sep_str = "-+-".join(sep_parts)
        
        row_strs = []
        for row in rows:
            padded_values = []
            for col_index in range(len(cols)):
                padded_values.append(row[col_index].ljust(col_widths[cols[col_index]]))
            row_str = " | ".join(padded_values)
            row_strs.append(row_str)
            
        table = header_str + "\n" + sep_str

        for rs in row_strs:
            table = table + "\n" + rs
            
        return table
    
    """
    Function 1: Project
    """
    def projection(self, *cols):
        try:
            if not cols:
                raise ValueError("No columns specified")

            if len(cols) == 1 and isinstance(cols[0], str):
                input_cols = [col_name.strip() for col_name in cols[0].split(",") if col_name.strip()]
            else:
                input_cols = [str(col_name).strip() for col_name in cols if str(col_name).strip()]

            if not input_cols:
                raise ValueError("Empty column list")

            missing_cols = [col_name for col_name in input_cols if col_name not in self.data]
            if missing_cols:
                raise KeyError(f"Columns not found: {missing_cols}")

            output = {col_name: self.data[col_name] for col_name in input_cols}
            return PSO.parser(data = output)

        except (KeyError, ValueError) as err:
            raise ValueError(f"Error in .projection: {err}")
        
    """
    Function 2: Join
    """
    def join(self, other: "PSO", *, left_on: str, right_on: str | None = None, join_type: str = "inner") -> "PSO":
        try:
            if not isinstance(other, PSO):
                raise TypeError(f"'{other}' must be a PSO instance")

            if right_on is None:
                right_on = left_on

            if left_on not in self.data:
                raise KeyError(f"left key '{left_on}' not found")
            if right_on not in other.data:
                raise KeyError(f"right key '{right_on}' not found")

            normalized_join_type = str(join_type).lower()
            valid_types = {"inner", "left", "right", "outer"}
            if normalized_join_type not in valid_types:
                raise ValueError(f"Type must be one of {sorted(valid_types)}")

            right_key_index: Dict[Any, List[int]] = {}
            for right_row_index, right_key_value in enumerate(other.data[right_on]):
                right_key_index.setdefault(right_key_value, []).append(right_row_index)

            left_columns = list(self.data.keys())
            right_columns = list(other.data.keys())

            right_payload_columns = [c for c in right_columns if c != right_on]

            overlapping_columns = set(left_columns).intersection(right_payload_columns)
            renamed_right_columns = {c: (f"right_{c}" if c in overlapping_columns else c) for c in right_payload_columns}

            output_columns = left_columns + [renamed_right_columns[c] for c in right_payload_columns]
            output_data_lists = {c: [] for c in output_columns}

            def append_joined_row(left_row_index: int | None, right_row_index: int | None):
                # left side
                for column in left_columns:
                    value = self.data[column][left_row_index] if left_row_index is not None else None
                    output_data_lists[column].append(value)
                # right side
                for right_col in right_payload_columns:
                    out_col = renamed_right_columns[right_col]
                    value = other.data[right_col][right_row_index] if right_row_index is not None else None
                    output_data_lists[out_col].append(value)

            matched_right_rows: set[int] = set()

            for left_row_index, left_key_value in enumerate(self.data[left_on]):
                matching_right_rows = right_key_index.get(left_key_value, [])

                if matching_right_rows:
                    for right_row_index in matching_right_rows:
                        append_joined_row(left_row_index, right_row_index)
                        matched_right_rows.add(right_row_index)
                else:
                    if normalized_join_type in {"left", "outer"}:
                        append_joined_row(left_row_index, None)

            # Unmatched right rows (right/outer)
            if normalized_join_type in {"right", "outer"}:
                left_key_values = set(self.data[left_on])

                for right_row_index, right_key_value in enumerate(other.data[right_on]):
                    if right_row_index in matched_right_rows:
                        continue

                    if normalized_join_type == "right" or right_key_value not in left_key_values:
                        append_joined_row(None, right_row_index)

            output = {col: tuple(values) for col, values in output_data_lists.items()}
            return PSO.parser(data = output)

        except (TypeError, KeyError, ValueError) as err:
            raise ValueError(f"Error in .join: {err}")
    
    """
    Function 3: Group By & Aggregate
    """
    def group_by(self, by: str or list = None) -> GroupBy:
        if by is None:
            raise ValueError("Error in .group_by(by): 'by' is required\n")
        if not isinstance(by, (str, list)):
            raise ValueError("Error in .group_by(by): 'by' must be a string or a list\n")
        
        by = [by] if isinstance(by, str) else by

        for b in by:
            if b not in self.header:
                raise ValueError(f"Error in .group_by(by): '{b}' column does not exist in dataset\n")

        num_rows = len(next(iter(self.data.values()))) if self.data else 0

        # Get the number of rows
        if num_rows == 0:
            return GroupBy({}, self.data, [])
        
        # Create groups based on row combinations
        groups = {}
        for row_index in range(num_rows):
            # Create a key tuple with values from each grouping column for row row_index
            key = tuple(self.data[col][row_index] for col in by)
            
            if key not in groups:
                groups[key] = []
            
            # Add the row index to this group
            groups.setdefault(key,[]).append(row_index)
        
        return GroupBy(groups, self.data, by)
    
    """
    Function 4: Filter
    """
    def single_filter(self, *args, chain = False):
        
        ok, msg = _arg_checker(*args)
        if not ok:
            raise ValueError(f"Error in .single_filter(args): {msg}")
    
        operators = {"=", "!=", "<", ">", "<=", ">=", "=="}
        columns = set(self.data.keys())
        operation = {"==": lambda x, y: x == y,
                     "!=": lambda x, y: x != y,
                     "<": lambda x, y: x < y,
                     ">": lambda x, y: x > y,
                     "<=": lambda x, y: x <= y,
                     ">=": lambda x, y: x >= y}
    
        # parse args (and has precedence over or)
        args = args[0]
        args = args.replace("or", "|").replace("OR", "|").replace("and", "&").replace("AND", "&")
    
        or_args = [oa for oa in args.split("|")]
        group_results = []
    
        for or_op in or_args:
            and_args = [aa for aa in or_op.split("&")]
            and_filterlist = []
    
            for and_op in and_args:
                matched = re.findall(r'^\s*(.*?)\s*(<=|>=|==|!=|<|>|=)\s*(.*?)\s*$', and_op)
                if not matched:
                    raise ValueError(f"Error in .single_filter(args): invalid clause syntax: '{and_op.strip()}'")
    
                col, op, val = matched[0]
                col, op, val = col.strip(), op.strip(), val.strip()
    
                if col not in columns:
                    raise KeyError(f"Error in .single_filter(args): '{col}' column does not exist in dataset")
                if op not in operators:
                    raise ValueError(f"Error in .single_filter(args): '{op}' is not a valid operator")
                if not val:
                    raise ValueError(f"Error in .single_filter(args): '{and_op.strip()}' empty comparator value")
    
                # get rid of quotes
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
    
                if op == "=":
                    op = "=="
    
                and_filterlist.append((col, op, val))
    
            # build index sets
            indexes = []
            for col, op, val in and_filterlist:
                tempset = set()
                col_vals = self.data[col]
    
                # try numeric conversion
                if re.match(r'^-?\d+(?:\.\d*)?$', str(val)):
                    val = float(val) if '.' in str(val) else int(val)
    
                for row_index, col_val in enumerate(col_vals):
                    try:
                        if operation[op](col_val, val):
                            tempset.add(row_index)
                    except TypeError:
                        raise TypeError(f"Error in .single_filter(args): Incompatible comparator in clause '{col} {op} {val}'")
                indexes.append(tempset)
    
            # intersection for AND
            if not indexes:
                continue
    
            final_indexes = indexes[0].copy()
            for s in indexes[1:]:
                final_indexes = final_indexes.intersection(s)
    
            group_results.append(final_indexes)
    
        # union for OR
        if not group_results:
            if chain:
                return set()            
            return {}
    
        final_indexes = set().union(*group_results)

        if chain:
            return final_indexes
        
        output = {col: tuple(self.data[col][row_index] for row_index in sorted(final_indexes))
                  for col in self.data}
    
        return PSO.parser(data = output)

    def multi_parser(args):
        args = args.strip()
        
        # remove outermost parentheses
        args = _strip_full_parentheses(args)
    
        # find OR
        depth = 0
        split_pos = None
        for index, char in enumerate(args):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "|" and depth == 0:
                split_pos = index
                break

        if split_pos is not None:
            # split on OR and loop
            left = args[:split_pos]
            right = args[split_pos + 1:]

            left_pairs = PSO.multi_parser(left)
            right_pairs = PSO.multi_parser(right)

            # get max depth
            max_depth = 0
            for depth, clause in left_pairs + right_pairs:
                if depth > max_depth:
                    max_depth = depth

            pairs = []
            pairs.extend(left_pairs)
            pairs.extend(right_pairs)

            total_depth = max_depth + 1 if max_depth > 0 else 1
            
            full_clause = f"{left.strip()} | {right.strip()}"
            pairs.append((total_depth, full_clause))
            return pairs

        # if no OR, look for AND
        depth = 0
        split_pos = None
        for index, char in enumerate(args):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "&" and depth == 0:
                split_pos = index
                break

        if split_pos is not None:
            left = args[:split_pos]
            right = args[split_pos + 1:]

            left_pairs = PSO.multi_parser(left)
            right_pairs = PSO.multi_parser(right)

            # get max depth
            max_depth = 0
            for depth, clause in left_pairs + right_pairs:
                if depth > max_depth:
                    max_depth = depth

            pairs = []
            pairs.extend(left_pairs)
            pairs.extend(right_pairs)

            total_depth = max_depth + 1 if max_depth > 0 else 1
            
            full_clause = f"{left.strip()} & {right.strip()}"
            pairs.append((total_depth, full_clause))
            return pairs

        # extract and organize
        atom = _strip_full_parentheses(args.strip())
    
        if not atom:
            return []
        return [(1, atom)]

    def _not_helper(self, args):
        args = args.strip()
        args = _strip_full_parentheses(args)

        inner_index = self.single_filter(args, chain = True)

        if not isinstance(inner_index, set):
            inner_index = set()

        if self.data:
            num_rows = len(list(self.data.values())[0])
        else: 
            num_rows = 0

        full_index = set(range(num_rows))        

        return full_index - inner_index
    
    def filter(self, *args):
        ok, msg = _arg_checker(*args)
        if not ok:
            raise ValueError(f"Error in .filter(args): {msg}")

        args = args[0]
        args = args.replace("or", "|").replace("OR", "|").replace("and", "&").replace("AND", "&")

        # check () count match
        if not args.count("(") == args.count(")"):
            raise ValueError("Error in .filter(args): () not closed")

        empty = {col: tuple() for col in self.data}
        
        # early single filter if no parentheses
        if args.count("(") == 0 and args.count(")") == 0:
            if args.strip().lower().startswith("not "):
                inner_clause = args.strip()[4:].lstrip()
                inner_index_list = sorted(self._not_helper(inner_clause))

                if not inner_index_list:
                    return PSO.parser(data = empty)

                output = {col: tuple(self.data[col][row_index] for row_index in inner_index_list) for col in self.data}
                return PSO.parser(data = output)

            return self.single_filter(args)

        # early single filter if one pair of parentheses at front and end
        if args.startswith("(") and args.endswith(")") and args.count("(") == 1 and args.count(")") == 1:
            clause = args[1:-1].strip()

            if clause.strip().lower().startswith("not "):
                inner_clause = clause.strip()[4:].lstrip()
                inner_index_list = sorted(self._not_helper(inner_clause))

                if not inner_index_list:
                    return PSO.parser(data = empty)

                output = {col: tuple(self.data[col][row_index] for row_index in inner_index_list) for col in self.data}
                return PSO.parser(data = output)
            
            return self.single_filter(clause)

        # parse precedence and clause
        prec_clause = PSO.multi_parser(args)
        prec_clause = sorted(prec_clause, key = lambda x: x[0])
        
        # iterate through clauses and produce results
        results = {}

        for level, clause in prec_clause:
            clause = clause.strip()
            filter_clause = _strip_full_parentheses(clause)

            if level == 1:
                if filter_clause.strip().lower().startswith("not "):
                    inner = filter_clause.strip()[4:].lstrip()
                    indexes = self._not_helper(inner)
                    results[filter_clause] = indexes
                    continue
                
                indexes = self.single_filter(filter_clause, chain = True)
                results[filter_clause] = indexes
                continue

            depth = 0
            split_pos, operator = None, None

            for index, char in enumerate(clause):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif (char == "&" or char == "|") and depth == 0:
                    split_pos = index
                    operator = char
                    break

            if split_pos is None or operator is None:
                print(f"Warning in .filter: no operator for '{clause}'")
                continue

            left_str = clause[:split_pos].strip()
            right_str = clause[split_pos + 1:].strip()

            left_key = _strip_full_parentheses(left_str)
            right_key = _strip_full_parentheses(right_str)

            left_set = results.get(left_key, set())
            right_set = results.get(right_key, set())

            if operator == "&":
                combined = left_set & right_set
            else:
                combined = left_set | right_set

            results[filter_clause] = combined

        # final results
        results_key = _strip_full_parentheses(args)
        final_index = results.get(results_key, set())
        final_index = sorted(final_index)

        if not final_index:
            return PSO.parser(data = empty)

        output = {col: tuple(self.data[col][row_index] for row_index in final_index) for col in self.data}
        return PSO.parser(data = output) 
