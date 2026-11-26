import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(page_title="Lab 8 | Feedback", layout="wide")

# ── Lesson 1: sample comments and sentiment labels ───────────────────────────
reviews = [
    "The bottle keeps my water cold all day.",
    "Great size and easy to carry to class.",
    "I love the simple design and sturdy lid.",
    "It leaks in my bag. Very disappointing.",
    "The cap is hard to open and the bottle feels heavy.",
    "Good value, but cleaning the narrow opening is difficult.",
    "Keeps drinks cold, though the paint scratches easily.",
    "The bottle is fine. Nothing special either way.",
    "I expected better insulation for this price.",
    "Lightweight, useful, and easy to wash.",
    "Not bad, but I would prefer a larger capacity.",
    "Excellent bottle. I use it every day.",
    "The lid broke after one week. Poor quality.",
    "It works as described and arrived on time.",
    "Nice color, but it does not keep drinks cold.",
    "Affordable and practical for carrying water.",
    "The bottle is okay; the handle is uncomfortable.",
    "Very happy with the purchase and the quality.",
]

# Assignment 4: one hand-assigned main topic per comment
topics = [
    "Insulation", "Portability", "Lid & cap", "Lid & cap", "Lid & cap", "Cleaning",
    "Durability", "General", "Insulation", "Cleaning", "Portability", "General",
    "Lid & cap", "General", "Insulation", "Price & value", "Portability", "General",
]

NEGATION_TEST = "The bottle is not easy to clean."

st.sidebar.header("Negation test")
negation_on = st.sidebar.checkbox(
    "Replace R06 with the negation test comment",
    help=f'Swaps R06 for "{NEGATION_TEST}" to see how VADER handles "not".',
)
if negation_on:
    reviews[5] = NEGATION_TEST

df = pd.DataFrame({"id": [f"R{i:02}" for i in range(1, 19)],
                   "text": reviews, "topic": topics})

analyzer = SentimentIntensityAnalyzer()


def label_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive", score
    if score <= -0.05:
        return "Negative", score
    return "Neutral", score


df[["sentiment", "compound"]] = df["text"].apply(
    lambda text: pd.Series(label_sentiment(text)))

# ── Lesson 2: dashboard ──────────────────────────────────────────────────────
st.title("Product Feedback: A Text Summary")
st.caption("Practice data only. Automated labels need human review.")

order = ["Positive", "Neutral", "Negative"]
counts = df["sentiment"].value_counts().reindex(order, fill_value=0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Comments", len(df))
m2.metric("Positive", counts["Positive"])
m3.metric("Neutral", counts["Neutral"])
m4.metric("Negative", counts["Negative"])

left, right = st.columns(2)
with left:
    st.subheader("Comment counts by sentiment")
    st.bar_chart(counts, horizontal=True, x_label="Number of comments", y_label="Sentiment")
    st.caption(
        "Each comment gets one label, so a comment that praises one feature and "
        "criticises another still counts once, in a single bar."
    )

with right:
    st.subheader("Words used in the comments")
    # Removed "day": it is frequent ("all day", "every day") but says nothing about the product.
    # Retained "not": the default list drops it, which hides negations like "does not keep drinks cold".
    stop_words = (set(STOPWORDS) | {"bottle", "drinks", "drink", "day"}) - {"not"}
    cloud = WordCloud(width=900, height=500, background_color="white",
                      colormap="viridis", stopwords=stop_words,
                      collocations=False, random_state=8).generate(" ".join(df["text"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "Word size shows how often a word appears (after removing filler words). "
        "Position and colour mean nothing, and size ignores context: \"cold\" is large, "
        "but one of its three uses is a complaint (\"does not keep drinks cold\")."
    )

st.subheader("Review the original comments")
st.dataframe(df[["id", "text", "topic", "sentiment", "compound"]],
             width="stretch", hide_index=True)
st.caption("A compound score is a model output, not a confidence rating.")

# ── Assignment 3: audit of six labels ────────────────────────────────────────
st.header("Audit: six labels checked by hand")
audit = pd.DataFrame([
    ("R02", "Reasonable", "Clear praise for size and portability."),
    ("R09", "Incorrect", "A complaint about insulation; VADER scores \"better\" as positive."),
    ("R14", "Reasonable", "Factual statement with no strong opinion."),
    ("R01", "Too low", "Keeping water cold all day is praise, but none of the words are in VADER's lexicon."),
    ("R13", "Reasonable", "Broken lid and \"poor quality\" are clearly negative."),
    ("R08", "Borderline", "\"Fine\" and \"nothing special\" read as neutral; VADER tips it slightly negative."),
], columns=["id", "verdict", "reason"])
audit = audit.merge(df[["id", "text", "sentiment", "compound"]], on="id")
st.dataframe(audit[["id", "text", "sentiment", "compound", "verdict", "reason"]],
             width="stretch", hide_index=True)

neg_scores = pd.DataFrame({
    "comment": ["The bottle is easy to clean.", NEGATION_TEST],
    "compound": [analyzer.polarity_scores(t)["compound"]
                 for t in ["The bottle is easy to clean.", NEGATION_TEST]],
})
neg_scores["sentiment"] = neg_scores["comment"].apply(lambda t: label_sentiment(t)[0])
st.subheader("Negation test")
st.dataframe(neg_scores, width="stretch", hide_index=True)
st.caption(
    "VADER handles this simple \"not\" correctly, flipping the score from positive to "
    "negative. It still misses R15 (\"Nice color, but it does not keep drinks cold\"), "
    "where \"nice\" outweighs the negated complaint. Use the sidebar to add the test "
    "comment to the dataset."
)

# ── Assignment 4: sentiment by topic ─────────────────────────────────────────
st.header("Sentiment by topic")
by_topic = (df.groupby(["topic", "sentiment"]).size().unstack(fill_value=0)
            .reindex(columns=order, fill_value=0))
by_topic = by_topic.loc[by_topic.sum(axis=1).sort_values(ascending=False).index]
st.bar_chart(by_topic, horizontal=True, stack=True,
             color=["#4C9F70", "#B8B8B8", "#D1495B"],
             x_label="Number of comments", y_label="Topic")
st.caption(
    "No topic has more than four comments, so this sample is too small for a fair "
    "comparison: one comment changes a topic's picture completely. Insulation also shows "
    "why labels need checking: VADER marks two of its three comments positive, yet two "
    "of the three are complaints."
)
