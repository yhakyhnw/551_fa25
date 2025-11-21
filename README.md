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

- The **demo_data/** directory contains the two datasets used in the demo:
note: the dummy datasets embedded in each of the package was designed to work with these two files!
    - `salaries.json` - NoSQL demo file of salaries for differing data science jobs
    - `vehicle_price_prediction.csv` - SQL demo file of second hand car prices

## 4. Executing the Application

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

## 5. Featured Functions

### **NoSQL Package**  
Supports both standard and chunked (streaming-style) operations: <br>
Note: only accepts .json filetypes

- **Project** — select specific fields  
- **Filter** — MongoDB‑style query expressions  
- **Join** — field‑based document inner join  
- **Group by** — group documents by one or more keys  
- **Aggregate** — numerical aggregations (sum, mean, min, max, count)

### **SQL Package**  
Implements core relational operations similar to traditional SQL: <br>
Note: only accepts .csv filetypes

- **Project** — choose specific columns  
- **Filter** — row filtering based on condition
- **Join** — combine rows across tables 
- **Group_by** — group rows by key  
- **Aggregate** — numerical aggregations (sum, mean, min, max, count)

## Authors

Yonghoon Kim <br>
DaeYong Han <br>
WooSeob Sim <br>
