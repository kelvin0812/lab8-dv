import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import altair as alt
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

st.sidebar.subheader("Negation test")
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
INK = "#1F2A37"
SUBTLE = "#52606D"
order = ["Positive", "Neutral", "Negative"]
# One colour per sentiment, reused by every chart so the legend only has to be learnt once
SENTIMENT_COLORS = ["#59A14F", "#B8C1CC", "#D1495B"]
sentiment_scale = alt.Scale(domain=order, range=SENTIMENT_COLORS)
y_axis = alt.Axis(labelFontSize=13, labelColor=INK, ticks=False, domain=False)
# Band padding (not a fixed bar height) keeps bars inside their rows at any chart height
band = alt.Scale(paddingInner=0.35, paddingOuter=0.2)
x_axis = alt.Axis(grid=False, tickMinStep=1, labelColor=SUBTLE, titleColor=SUBTLE)

st.caption("TFB3133 DATA VISUALIZATION · LAB 8")
st.title("Product Feedback: A Text Summary")
st.markdown(
    f"<p style='color:{SUBTLE}; font-size:1.05rem; margin-top:-0.5rem'>"
    "18 short comments about a water bottle, labelled by VADER and checked by hand. "
    "Practice data only. Automated labels need human review.</p>",
    unsafe_allow_html=True,
)

counts = df["sentiment"].value_counts().reindex(order, fill_value=0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Comments", len(df), border=True)
m2.metric("Positive", counts["Positive"], border=True)
m3.metric("Neutral", counts["Neutral"], border=True)
m4.metric("Negative", counts["Negative"], border=True)

overview, comments, audit_tab, topic_tab = st.tabs(
    ["Overview", "Comments", "Label audit", "By topic"]
)

with overview:
    left, right = st.columns(2, gap="large")
    with left.container(border=True, height="stretch"):
        st.subheader("Comment counts by sentiment")
        count_df = counts.rename_axis("sentiment").reset_index(name="comments")
        base = alt.Chart(count_df).encode(
            y=alt.Y("sentiment:N", sort=order, title=None, axis=y_axis, scale=band),
            x=alt.X("comments:Q", title="Number of comments", axis=x_axis),
        )
        bars = base.mark_bar(cornerRadiusEnd=4).encode(
            color=alt.Color("sentiment:N", scale=sentiment_scale, legend=None),
            tooltip=["sentiment", "comments"],
        )
        labels = base.mark_text(align="left", dx=6, fontSize=13, fontWeight="bold",
                                color=SUBTLE).encode(text="comments:Q")
        st.altair_chart((bars + labels).properties(height=260).configure_view(stroke=None),
                        width="stretch")
        st.caption(
            "Each comment gets one label, so a comment that praises one feature and "
            "criticises another still counts once, in a single bar."
        )

    with right.container(border=True, height="stretch"):
        st.subheader("Words used in the comments")
        # Removed "day": it is frequent ("all day", "every day") but says nothing about the product.
        # Retained "not": the default list drops it, which hides negations like "does not keep drinks cold".
        stop_words = (set(STOPWORDS) | {"bottle", "drinks", "drink", "day"}) - {"not"}
        # Navy shades only: a multi-hue palette would invite readers to read colour as sentiment
        word_colors = ListedColormap(["#1F2A37", "#2F5D8A", "#4C78A8", "#6B8FB8"])
        cloud = WordCloud(width=1200, height=650, background_color="white",
                          colormap=word_colors, stopwords=stop_words, prefer_horizontal=0.95,
                          collocations=False, random_state=8).generate(" ".join(df["text"]))
        fig, ax = plt.subplots(figsize=(8, 4.3), dpi=150)
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        fig.tight_layout(pad=0)
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "Word size shows how often a word appears (after removing filler words). "
            "Position and colour mean nothing, and size ignores context: \"cold\" is large, "
            "but one of its three uses is a complaint (\"does not keep drinks cold\")."
        )

compound_col = st.column_config.NumberColumn(
    "compound", format="%+.3f", help="VADER score from -1 (negative) to +1 (positive)")

with comments:
    st.dataframe(df[["id", "text", "topic", "sentiment", "compound"]],
                 width="stretch", hide_index=True, height=665,
                 column_config={"text": st.column_config.TextColumn("comment", width="large"),
                                "compound": compound_col})
    st.caption("A compound score is a model output, not a confidence rating.")

# ── Assignment 3: audit of six labels ────────────────────────────────────────
with audit_tab:
    st.subheader("Six labels checked by hand")
    st.caption("Two from each label VADER assigned. Two of the six are wrong and one is borderline.")
    audit = pd.DataFrame([
        ("R02", "✅ Reasonable", "Clear praise for size and portability."),
        ("R09", "❌ Incorrect", "A complaint about insulation; VADER scores \"better\" as positive."),
        ("R14", "✅ Reasonable", "Factual statement with no strong opinion."),
        ("R01", "❌ Too low", "Keeping water cold all day is praise, but none of the words are in VADER's lexicon."),
        ("R13", "✅ Reasonable", "Broken lid and \"poor quality\" are clearly negative."),
        ("R08", "⚠️ Borderline", "\"Fine\" and \"nothing special\" read as neutral; VADER tips it slightly negative."),
    ], columns=["id", "verdict", "reason"])
    audit = audit.merge(df[["id", "text", "sentiment", "compound"]], on="id")
    st.dataframe(audit[["id", "text", "sentiment", "compound", "verdict", "reason"]],
                 width="stretch", hide_index=True,
                 column_config={"text": st.column_config.TextColumn("comment", width="large"),
                                "reason": st.column_config.TextColumn(width="large"),
                                "compound": compound_col})

    st.write("")
    st.subheader("Negation test")
    n1, n2 = st.columns(2)
    for col, text in zip((n1, n2), ["The bottle is easy to clean.", NEGATION_TEST]):
        label, score = label_sentiment(text)
        col.metric(f"“{text}”", f"{score:+.3f}", label, border=True,
                   delta_color="normal" if label == "Positive" else "inverse",
                   delta_arrow="off")
    st.caption(
        "VADER handles this simple \"not\" correctly, flipping the score from positive to "
        "negative. It still misses R15 (\"Nice color, but it does not keep drinks cold\"), "
        "where \"nice\" outweighs the negated complaint. Use the sidebar to add the test "
        "comment to the dataset."
    )

# ── Assignment 4: sentiment by topic ─────────────────────────────────────────
with topic_tab:
    st.subheader("Sentiment by topic")
    by_topic = df.groupby(["topic", "sentiment"]).size().reset_index(name="comments")
    by_topic["rank"] = by_topic["sentiment"].map({s: i for i, s in enumerate(order)})
    topic_order = df["topic"].value_counts().index.tolist()
    topic_chart = alt.Chart(by_topic).mark_bar().encode(
        y=alt.Y("topic:N", sort=topic_order, title=None, axis=y_axis, scale=band),
        x=alt.X("sum(comments):Q", title="Number of comments", axis=x_axis),
        color=alt.Color("sentiment:N", scale=sentiment_scale,
                        legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
        order=alt.Order("rank:Q"),
        tooltip=["topic", "sentiment", "comments"],
    ).properties(height=320).configure_view(stroke=None)
    with st.container(border=True):
        st.altair_chart(topic_chart, width="stretch")
    st.caption(
        "No topic has more than four comments, so this sample is too small for a fair "
        "comparison: one comment changes a topic's picture completely. Insulation also shows "
        "why labels need checking: VADER marks two of its three comments positive, yet two "
        "of the three are complaints."
    )
