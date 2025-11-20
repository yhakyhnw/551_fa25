import tempfile
import io, os, sys
import contextlib
import streamlit as st

base_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.dirname(base_directory)
packages_directory = os.path.join(root_directory, "packages")

sys.path.append(packages_directory)

from NoSQL_package import NoSql  

