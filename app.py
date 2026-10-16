"""Streamlit demo: try the lab's prompting techniques and applications in a browser.

Run with:  uv run streamlit run app.py
"""

import json
import time
import urllib.request

import pandas as pd
import streamlit as st

from promptlab import chains
from promptlab.llm import BASE_URL, MODEL, llm_model
from promptlab.review_analyzer import analyze_review, report_markdown, review_analysis_chain, summarize
from promptlab.techniques import EXAMPLES, self_consistency

st.set_page_config(page_title="Prompt Engineering Lab", page_icon="🧪", layout="wide")


# --- Helpers ---------------------------------------------------------------------

@st.cache_data(ttl=30)
def ollama_status() -> str:
    """Check that Ollama is running and has our model. Re-checked every 30 s."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=2) as response:
            installed = [m["name"] for m in json.load(response)["models"]]
    except Exception:
        return "offline"
    return "ready" if MODEL in installed else "model missing"


@st.cache_resource
def get_chain(name: str):
    """Build each chain once and reuse it on every click."""
    builders = {
        "summarization": chains.summarization_chain,
        "qa": chains.qa_chain,
        "classification": chains.classification_chain,
        "sql": chains.sql_chain,
        "roleplay": chains.roleplay_chain,
        "review": review_analysis_chain,
    }
    return builders[name]()


def timed(function, *args):
    """Run a function and also return how many seconds it took."""
    start = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - start


# --- Sidebar ----------------------------------------------------------------------

with st.sidebar:
    st.title("🧪 Prompt Engineering Lab")
    st.write(f"**Model:** `{MODEL}`  \nRunning locally with Ollama.")

    status = ollama_status()
    if status == "ready":
        st.success("Ollama is running")
    elif status == "model missing":
        st.warning(f"Model not found. Run:\n\n`ollama pull {MODEL}`")
    else:
        st.error("Ollama is not running. Start the Ollama app, then refresh this page.")

    st.divider()
    st.caption(
        "Based on IBM's *Develop Generative AI Applications* lab "
        "(Master Prompt Engineering and LangChain PromptTemplates), "
        "rebuilt to run locally and extended."
    )

tab_techniques, tab_apps, tab_reviews = st.tabs(
    ["Prompting techniques", "Applications", "Review analyzer"]
)


# --- Tab 1: Prompting techniques ------------------------------------------

with tab_techniques:
    technique = st.selectbox("Technique", list(EXAMPLES))
    example = EXAMPLES[technique]
    st.caption(example["description"])

    # Keys include the technique name, so switching technique resets the inputs.
    label = "Question" if technique == "Self-consistency" else "Prompt"
    prompt = st.text_area(label, example["prompt"], height=220, key=f"prompt_{technique}")

    col1, col2, col3 = st.columns(3)
    temperature = col1.slider("Temperature", 0.0, 1.0, example["params"]["temperature"], 0.1,
                              key=f"temp_{technique}")
    max_tokens = col2.number_input("Max new tokens", 5, 1024, example["params"]["max_new_tokens"],
                                   key=f"tokens_{technique}")
    top_k = col3.number_input("Top-k", 1, 100, example["params"]["top_k"], key=f"topk_{technique}")
    params = {"temperature": temperature, "max_new_tokens": int(max_tokens), "top_k": int(top_k)}

    if technique == "Self-consistency":
        num_samples = st.slider("Number of independent samples", 3, 9, 5, step=2)
        if st.button("Run", type="primary", key="run_sc"):
            with st.spinner(f"Generating {num_samples} independent answers..."):
                result, seconds = timed(self_consistency, prompt, num_samples, params)
            st.metric("Answer by majority vote", result["winner"],
                      f"{result['count']}/{num_samples} samples agree", delta_color="off")
            st.write("**Votes:**", result["votes"])
            for i, sample in enumerate(result["samples"], start=1):
                with st.expander(f"Sample {i}: answer {sample['answer']}"):
                    st.write(sample["reasoning"])
            st.caption(f"Took {seconds:.1f} s")
    else:
        if st.button("Run", type="primary", key="run_technique"):
            with st.spinner("Thinking..."):
                answer, seconds = timed(llm_model, prompt, params)
            st.markdown("**Response**")
            st.write(answer)
            st.caption(f"Took {seconds:.1f} s")


# --- Tab 2: Applications (LangChain prompt templates + LCEL chains) -----------

with tab_apps:
    app = st.radio(
        "Application",
        ["Summarization", "Question answering", "Classification", "SQL generation", "Role play"],
        horizontal=True,
    )

    if app == "Summarization":
        content = st.text_area("Text to summarize", height=180, value=(
            "The rapid advancement of technology in the 21st century has transformed healthcare, "
            "education, and transportation. AI-powered diagnostic tools improve the speed and "
            "accuracy of medical diagnoses, smart transportation systems reduce traffic congestion, "
            "and online learning platforms make education accessible around the world."))
        if st.button("Summarize", type="primary"):
            with st.spinner("Summarizing..."):
                st.write(get_chain("summarization").invoke({"content": content}))

    elif app == "Question answering":
        content = st.text_area("Context", height=150, value=(
            "The solar system consists of the Sun, eight planets, their moons, dwarf planets, and "
            "smaller objects. The inner planets (Mercury, Venus, Earth, and Mars) are rocky and solid. "
            "The outer planets (Jupiter, Saturn, Uranus, and Neptune) are much larger and gaseous."))
        question = st.text_input("Question", "Which planets in the solar system are rocky and solid?")
        st.caption("Tip: ask something the context doesn't cover. The model should say it's unsure.")
        if st.button("Answer", type="primary"):
            with st.spinner("Reading the context..."):
                st.write(get_chain("qa").invoke({"content": content, "question": question}))

    elif app == "Classification":
        text = st.text_area("Text", "The concert last night was an exhilarating experience "
                                    "with outstanding performances by all artists.")
        categories = st.text_input("Categories (comma-separated)",
                                   "Entertainment, Food and Dining, Technology, Literature, Music")
        if st.button("Classify", type="primary"):
            with st.spinner("Classifying..."):
                st.write(get_chain("classification").invoke({"text": text, "categories": categories}))

    elif app == "SQL generation":
        description = st.text_area("Describe the query", height=120, value=(
            "Retrieve the names and email addresses of all customers from the 'customers' table "
            "who have made a purchase in the last 30 days. The 'purchases' table has the columns "
            "'customer_id' and 'purchase_date'."))
        if st.button("Generate SQL", type="primary"):
            with st.spinner("Writing SQL..."):
                sql = get_chain("sql").invoke({"description": description})
            # Remove ``` fences if the model added them, then show as highlighted code.
            st.code(sql.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip(),
                    language="sql")

    else:  # Role play
        col1, col2 = st.columns(2)
        role = col1.text_input("Role", "Dungeons & Dragons game master")
        tone = col2.text_input("Tone", "engaging and immersive")
        st.caption("Like the lab's chatbot, each question is answered on its own (no memory).")

        # st.session_state survives between clicks, so the conversation stays on screen.
        if "chat" not in st.session_state:
            st.session_state.chat = []

        question = st.text_input("Your question", "Who are you?")
        if st.button("Ask", type="primary"):
            with st.spinner("The game master is thinking..."):
                answer = get_chain("roleplay").invoke({"role": role, "tone": tone, "question": question})
            st.session_state.chat.append((question, answer))

        for asked, answered in reversed(st.session_state.chat):  # newest first
            st.chat_message("user").write(asked)
            st.chat_message("assistant").write(answered)


# --- Tab 3: Review analyzer (structured output) ------------------------------------

with tab_reviews:
    st.write("Analyze a CSV file of product reviews. It needs a column named **review**.")

    uploaded = st.file_uploader("Upload a CSV", type="csv")
    reviews = pd.read_csv(uploaded) if uploaded else pd.read_csv("data/sample_reviews.csv")
    if not uploaded:
        st.caption("No file uploaded, so the 20 sample reviews are used.")

    if "review" not in reviews.columns:
        st.error("The file has no 'review' column.")
        st.stop()  # don't run the rest of the page

    limit = st.slider("Reviews to analyze", 1, len(reviews), min(5, len(reviews)))
    st.dataframe(reviews.head(limit), hide_index=True)

    if st.button("Analyze reviews", type="primary"):
        chain = get_chain("review")
        progress = st.progress(0.0, "Starting...")
        results, rows, failed = [], [], 0

        for i, row in enumerate(reviews.head(limit).to_dict("records"), start=1):
            try:
                analysis = analyze_review(chain, str(row["review"]))
                results.append(analysis)
                rows.append({**row, "sentiment": analysis.sentiment,
                             "features": "; ".join(analysis.features), "summary": analysis.summary})
            except Exception as error:
                failed += 1
                rows.append({**row, "sentiment": "error", "features": "", "summary": str(error)})
            progress.progress(i / limit, f"Analyzed {i}/{limit}")

        stats = summarize(results, failed)
        progress.empty()

        # Headline numbers
        cols = st.columns(4)
        cols[0].metric("Analyzed", stats["total"])
        for col, sentiment in zip(cols[1:], ["positive", "negative", "neutral"]):
            col.metric(sentiment.capitalize(), stats["sentiments"][sentiment])

        # Feature charts
        left, right = st.columns(2)
        for column, title, items in [
            (left, "Most mentioned features", stats["top_features"]),
            (right, "Most mentioned in negative reviews", stats["top_negative_features"]),
        ]:
            column.markdown(f"**{title}**")
            if items:
                column.bar_chart(pd.DataFrame(items, columns=["feature", "mentions"]).set_index("feature"),
                                 horizontal=True)
            else:
                column.caption("None found.")

        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, hide_index=True)

        st.download_button("Download results (CSV)", results_df.to_csv(index=False).encode("utf-8-sig"),
                           "review_results.csv", "text/csv")
        st.download_button("Download report (Markdown)", report_markdown(stats),
                           "review_report.md", "text/markdown")
