import streamlit as st
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬")

# Title
st.title("🎬 IMDB Sentiment Classifier")
st.markdown("### Analyze movie reviews using AI 🤖")

# Input
review = st.text_area("✍️ Enter your movie review here:")

# Prediction
if st.button("🔍 Predict Sentiment"):
    if review:
        data = vectorizer.transform([review])
        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)

        st.subheader("Result:")

        if prediction == "positive":
            st.success("😊 Positive Review")
            st.write(f"Confidence: {prob[0][1]*100:.2f}%")
        else:
            st.error("😠 Negative Review")
            st.write(f"Confidence: {prob[0][0]*100:.2f}%")
    else:
        st.warning("⚠️ Please enter a review first")

# WordCloud
if st.button("📊 Show WordCloud"):
    if review:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black'
        ).generate(review)

        st.subheader("WordCloud Visualization")

        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")

        st.pyplot(fig)
    else:
        st.warning("⚠️ Enter a review first")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")