import streamlit as st
import sql_app

def homepage():
    st.title("DSCI 551 Project Demo")
    st.markdown("---")

    st.markdown("Select a demo to run:")

    col_sql, col_nosql = st.columns(2)

    with col_sql:
        if st.button("SQL demo"):
            st.session_state.current_page = "sql"

    with col_nosql:
        if st.button("NoSQL demo"):
            st.session_state.current_page = "nosql"


def show_sql_demo():
    st.button("Back to home", on_click=lambda: set_page("home"))

    if sql_app is None:
        st.error("SQL demo app (sql_app.py) not found.")
        return

    sql_app.main()


def show_nosql_demo():
    st.button("← Back to menu", on_click=lambda: set_page("home"))

    st.title("NoSQL Demo")
    st.write("NoSQL demo coming soon.")

def set_page(page_name: str):
    st.session_state.current_page = page_name

def main():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    current_page = st.session_state.current_page

    if current_page == "home":
        homepage()
    elif current_page == "sql":
        show_sql_demo()
    elif current_page == "nosql":
        show_nosql_demo()
    else:
        # Fallback
        homepage()


if __name__ == "__main__":
    main()