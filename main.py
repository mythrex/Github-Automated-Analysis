import streamlit as st
import pandas as pd
import github_fetcher
import llm_analyser


def main():
    st.title("Analyze your github repos.")

    # Create a text box for user input
    input_text = st.text_input("Enter Github username")

    # Create a submit button
    if st.button("Submit"):
        # clone repose
        username = input_text
        st.write("Cloning github repos ....")
        st.write("This would take some time depending of size of your repos.")
        github_fetcher.main(username)
        st.write("Github repos cloned! Using ChatGPT for analysing repos")
        st.write("Using LLM to process repos")
        llm_analyser.run(username)
        st.write("Repos Analyzed")
        df = pd.read_csv(f"dumped/{username}.code_analysis.ranked.csv")
        st.dataframe(df)
        row = df.iloc[0]
        st.write(
            f"{username}'s most complex repo based on parameters given below is {row.repo}"
        )


if __name__ == "__main__":
    main()
