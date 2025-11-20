import streamlit as st
import sql_app
import nosql_app

def homepage():
    st.markdown("## DSCI 551 Project Demo")
    st.markdown("---")

    st.markdown("### Select a demo to run:")

    st.subheader("🏎️ SQL Demo 🏁")
    st.markdown("The SQL demo uses a second-hand car data!  \nA dummy dataset is also available for speedy testing!")

    st.button("Start SQL Demo", on_click = lambda: set_page("sql"))

    st.markdown("---")

    st.subheader("💰 NoSQL Demo 📈")
    st.markdown("The NoSQL demo uses salary data!  \nA dummy dataset is also available for speedy testing!")

    st.button("Start NoSQL Demo", on_click = lambda: set_page("nosql"))


def sql_demo():
    st.button("Back to home", on_click = lambda: set_page("home"))

    if sql_app is None:
        st.error("SQL demo app (sql_app.py) not found.")
        return

    sql_app.main()


def nosql_demo():
    st.button("Back to home", on_click = lambda: set_page("home"))

    if nosql_app is None:
        st.error("NoSQL demo app (nosql_app.py) not found.")
        return

    nosql_app.main()

def set_page(page_name: str):
    st.session_state.current_page = page_name

def main():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    current_page = st.session_state.current_page

    if current_page == "home":
        homepage()
    elif current_page == "sql":
        sql_demo()
    elif current_page == "nosql":
        nosql_demo()
    else:
        homepage()

if __name__ == "__main__":
    main()