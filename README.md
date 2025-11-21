# DSCI 551 Fall 2025 Project

USC Fall 2025 DSCI 551 <br>
Instructor: WenSheng Wu

## 1. Description

**Project Objective:** <br>
Understand what happens under the hood of SQL and NoSQL functions. 

## 2. Dependencies

streamlit, io, os, sys, re, typing

## 3. Folder Infrastructure

- The **app/** directory contains .py files necessary to run the streamlit program:
    - `full_py.py` — combined full application  
    - `nosql_app.py` — NoSQL interface  
    - `sql_app.py` — SQL interface  

- The **packages/** directory contains the two core processing modules:
  - `NoSQL_package.py` — NoSQL package
  - `SQL_package.py` — SQL package

## 4. Executing the Program

You can run the application in two different ways:

### **Option 1 — Run Online**
Use Streamlit link:  
**https://fullapp-551-hks.streamlit.app/**

### **Option 2 — Run Locally**
1. Download the entire repository to your local machine.  
2. Ensure all required dependencies are installed.  
3. Open a terminal and navigate to the **app/** directory:  
   ```
   cd app
   ```  
4. Run Streamlit with:  
   ```
   streamlit run full_app.py
   ```
## Authors

Yonghoon Kim <br>
DaeYong Han <br>
WooSeob Sim <br>
