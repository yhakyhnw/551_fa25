# DSCI 551 Fall 2025 Project

Simple overview of use/purpose.

## 1. Description

An in-depth paragraph about your project and overview of use.

## 2. Dependencies

streamlit, io, os, sys, re, typing

### 3. Folder Infrastructure

- The **app/** directory contains .py files necessary to run the streamlit program:
    - `full_py.py` — combined full application  
    - `nosql_app.py` — NoSQL interface  
    - `sql_app.py` — SQL interface  

- The **packages/** directory contains the two core processing modules:
  - `NoSQL_package.py` — NoSQL package
  - `SQL_package.py` — SQL package


### 4. Executing program

There are two ways to run the full_app.py program:
1. Internet: https://fullapp-551-hks.streamlit.app/
2. Local:
    i. Download entire repo onto locale
    ii. Ensure all dependencies are installed
    iii. On terminal, navigate to app folder
    iv. Execute the following on terminal:
        streamlit run full_app.py

## Authors

Yonghoon Kim <br>
DaeYong Han <br>
WooSeob Sim <br>
