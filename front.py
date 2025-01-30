import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/summarize/"

# Set the title for the app
st.title("Text Summarizer")

# Input fields
text_input = st.text_area("Enter the text to summarize:", height=250)

method = st.selectbox("Choose Summarization Method", ["extractive", "abstractive"])

max_length = st.slider("Select Max Summary Length", min_value=50, max_value=500, value=150, step=10)

# When the user clicks the "Summarize" button
if st.button("Summarize"):
    if text_input.strip():
        # Prepare data to send to FastAPI
        payload = {
            "text": text_input,
            "method": method,
            "max_length": max_length
        }
        
        try:
            # Send a POST request to the FastAPI backend
            response = requests.post(API_URL, json=payload)
            response_data = response.json()

            # Check if the response contains an error
            if "error" in response_data:
                st.error(response_data["error"])
            else:
                # Display the generated summary
                st.subheader(f"Generated {method.capitalize()} Summary:")
                st.write(response_data["summary"])
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the backend: {e}")
    else:
        st.error("Text cannot be empty.")