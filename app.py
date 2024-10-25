import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


# Function to analyze sentiment of each word and the overall sentence
def analyze_sentiment(text):
    words = text.split()

    # Calculate overall sentiment for the sentence
    overall_sentiment = sia.polarity_scores(text)['compound']
    if overall_sentiment > 0:
        overall_sentiment_result = 'Positive'
    elif overall_sentiment < 0:
        overall_sentiment_result = 'Negative'
    else:
        overall_sentiment_result = 'Neutral'

    # Initialize lists to hold words by sentiment category
    positive_words = []
    neutral_words = []
    negative_words = []

    for word in words:
        sentiment = sia.polarity_scores(word)['compound']
        if sentiment > 0:
            positive_words.append(word)
        elif sentiment < 0:
            negative_words.append(word)
        else:
            neutral_words.append(word)

    return overall_sentiment_result, positive_words, neutral_words, negative_words


# Streamlit app layout
st.set_page_config(page_title="Sentiment Analysis Web App", page_icon="😊")
st.title("Sentiment Analysis Web App")

text = st.text_input("Enter a sentence for sentiment analysis:")
st.write("")  # Adding space after input

if st.button("Analyze"):
    st.write("")  # Adding space after the button
    if text:
        overall_sentiment, positive_words, neutral_words, negative_words = analyze_sentiment(text)

        # Display the overall sentiment with increased font size using markdown
        st.subheader("Overall Sentiment: " + overall_sentiment)

        # Add some space
        st.write("")  # Adding an empty line for spacing

        # Display word-wise results categorized with emojis
        st.subheader("Word-wise Result:")

        if positive_words:
            st.markdown(f"😊 Positive: {', '.join(positive_words)}")
        else:
            st.markdown("😊 Positive: None")

        if neutral_words:
            st.markdown(f"😐 Neutral: {', '.join(neutral_words)}")
        else:
            st.markdown("😐 Neutral: None")

        if negative_words:
            st.markdown(f"😢 Negative: {', '.join(negative_words)}")
        else:
            st.markdown("😢 Negative: None")
    else:
        st.write("Please enter some text to analyze.")
