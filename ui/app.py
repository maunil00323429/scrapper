"""
Streamlit UI Application

Provides a user-friendly web interface for the Web Content Scraper
& Text Analyzer. Users enter a URL and receive visual analysis
results including keywords, topics, readability scores, and entities.

Run with: streamlit run ui/app.py
"""

import importlib
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

from src.scraper.extractor import WebContentExtractor
from src.nlp.processor import NLPProcessor
from src.analysis.keywords import KeywordExtractor
from src.analysis.topics import TopicDetector
from src.analysis.readability import ReadabilityAnalyzer
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.summarizer import TextSummarizer
import src.analysis.chatbot as _chatbot_module

importlib.reload(_chatbot_module)
ContentChatbot = _chatbot_module.ContentChatbot


def build_analysis_record(
    content,
    processed,
    keywords,
    topics,
    readability,
    sentiment,
    summary_result,
    nlp,
) -> dict:
    """Shape matching API / comparator expectations."""
    return {
        "metadata": {
            "url": content.url,
            "title": content.title,
            "author": content.author,
            "date": content.date,
            "word_count": content.word_count,
            "extraction_method": content.extraction_method,
        },
        "readability": readability.to_dict(),
        "keywords": keywords.to_dict(),
        "topics": topics.to_dict(),
        "sentiment": sentiment.to_dict(),
        "summary": summary_result.to_dict(),
        "entity_summary": nlp.get_entity_summary(processed),
    }


def readability_radar_fig(readability) -> go.Figure:
    """Spider chart: each metric scaled to 0–100 vs a typical upper bound."""
    sc = readability.to_dict()["scores"]
    labels = [
        "Flesch Ease",
        "FK Grade",
        "Gunning",
        "SMOG",
        "Coleman-Liau",
        "ARI",
        "Dale-Chall",
    ]
    keys = [
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "gunning_fog",
        "smog_index",
        "coleman_liau_index",
        "automated_readability_index",
        "dale_chall_score",
    ]
    caps = [100.0, 20.0, 20.0, 20.0, 20.0, 20.0, 10.0]
    vals = []
    for key, cap in zip(keys, caps):
        v = float(sc[key])
        vals.append(max(0.0, min(100.0, (v / cap) * 100.0)))
    vals_closed = vals + [vals[0]]
    labels_closed = labels + [labels[0]]
    fig = go.Figure(
        data=go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            name="Scaled score",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Readability metrics (scaled to 0–100 vs typical upper bound)",
        showlegend=False,
        height=500,
        margin=dict(t=60, b=40),
    )
    return fig


def fmt2(x) -> float:
    """Round a numeric value for display (max 2 decimal places)."""
    return round(float(x), 2)


def round_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with float columns rounded to 2 decimal places."""
    out = df.copy()
    for col in out.select_dtypes(include=["float64", "float32"]).columns:
        out[col] = out[col].round(2)
    return out


PRIMARY_ENTITY_LABELS: tuple[str, ...] = (
    "PERSON",
    "ORG",
    "GPE",
    "DATE",
    "WORK_OF_ART",
)
_ENTITY_SIDEBAR_LIMIT = 5


def _format_entity_sidebar_value(
    entities: list[str], limit: int = _ENTITY_SIDEBAR_LIMIT
) -> str:
    if not entities:
        return ""
    shown = entities[:limit]
    out = ", ".join(shown)
    extra = len(entities) - limit
    if extra > 0:
        out += f" … (+{extra} more)"
    return out


def _render_sidebar_entity_summary(ent_sidebar: dict[str, list[str]]) -> None:
    st.sidebar.markdown("---")
    st.sidebar.header("🏷 Entities")
    if not ent_sidebar:
        st.sidebar.caption("No named entities detected.")
        return
    for label in PRIMARY_ENTITY_LABELS:
        entities = ent_sidebar.get(label) or []
        if entities:
            st.sidebar.markdown(
                f"**{label}:** {_format_entity_sidebar_value(entities)}"
            )
    secondary = {
        k: v
        for k, v in ent_sidebar.items()
        if k not in PRIMARY_ENTITY_LABELS and v
    }
    if secondary:
        with st.sidebar.expander("Show all entity types"):
            for label in sorted(secondary.keys()):
                st.markdown(
                    f"**{label}:** {_format_entity_sidebar_value(secondary[label])}"
                )



def _chat_dialog_dismiss() -> None:
    st.session_state["chat_dialog_open"] = False


@st.dialog(
    "Ask about this page",
    width="large",
    on_dismiss=_chat_dialog_dismiss,
)
def _page_chat_dialog() -> None:
    """Modal Q&A over the scraped page (history in st.session_state)."""
    if not st.session_state.get("chat_dialog_open"):
        return

    _, _close_col = st.columns([0.78, 0.22])
    with _close_col:
        if st.button("Close", key="dialog_close_chat", use_container_width=True):
            st.session_state["chat_dialog_open"] = False
            st.rerun()

    if not ContentChatbot.is_configured():
        st.info("Set **OPENAI_API_KEY** in `.env` to enable Q&A.")
        return

    ctx = (st.session_state.get("chat_context_text") or "").strip()
    if not ctx:
        st.warning("No page text in memory. Run **Analyze** again.")
        return

    hist = st.session_state.setdefault("chat_history", [])
    for msg in hist:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input(
        "Question about the scraped content",
        key="dialog_page_chat_input",
    ):
        _title = st.session_state.get("chat_context_title")
        bot = ContentChatbot(ctx, page_title=_title)
        prior = list(hist)
        hist.append({"role": "user", "content": prompt})
        try:
            reply = bot.answer(prompt, prior)
        except Exception as exc:
            reply = f"**Error:** {exc}"
        hist.append({"role": "assistant", "content": reply})
        st.rerun()


# ── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="Web Content Analyzer",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Web Content Scraper & Text Analyzer")
st.markdown(
    "Enter a public web page URL to extract and analyze its content "
    "using NLP techniques."
)

# ── Sidebar Settings ────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
num_topics = st.sidebar.slider("Number of Topics", 2, 10, 3)
num_keywords = st.sidebar.slider("Top Keywords", 5, 30, 15)

# ── Main Input ──────────────────────────────────────────────────
url = st.text_input(
    "🌐 Enter URL to analyze",
    placeholder="https://en.wikipedia.org/wiki/Natural_language_processing",
)

_bundle = st.session_state.get("analysis_bundle")
_analyzed_url = (st.session_state.get("analyzed_url") or "").strip()
_url_matches = bool(_bundle and url.strip() and url.strip() == _analyzed_url)
if _url_matches:
    st.sidebar.markdown("---")
    st.sidebar.caption("Q&A (uses scraped text)")
    if st.sidebar.button(
        "💬 Ask about this page",
        use_container_width=True,
        key="sidebar_open_chat",
    ):
        st.session_state["chat_dialog_open"] = True
        st.rerun()


@st.cache_resource
def load_nlp_processor():
    """Load spaCy model once and cache it."""
    return NLPProcessor()


@st.cache_resource
def load_sentiment_analyzer():
    """Load VADER-backed analyzer once (downloads NLTK data on first use)."""
    return SentimentAnalyzer()


def run_analysis(target_url: str, topic_count: int, keyword_count: int):
    """
    Execute the full analysis pipeline on a URL.

    Returns:
        Tuple: content, processed, keywords, topics, readability, sentiment,
        summary_result, nlp.
    """
    scraper = WebContentExtractor()
    nlp = load_nlp_processor()
    kw_extractor = KeywordExtractor(top_n=keyword_count)
    t_detector = TopicDetector(num_topics=topic_count)
    r_analyzer = ReadabilityAnalyzer()
    sentiment_analyzer = load_sentiment_analyzer()
    summarizer = TextSummarizer()

    content = scraper.extract(target_url)
    processed = nlp.process(content.text)
    keywords = kw_extractor.extract(processed)
    topics = t_detector.detect(processed.sentences)
    readability = r_analyzer.analyze(content.text)
    sentiment = sentiment_analyzer.analyze(content.text)
    summary_result = summarizer.summarize(content.text)

    return (
        content,
        processed,
        keywords,
        topics,
        readability,
        sentiment,
        summary_result,
        nlp,
    )


# ── Run Analysis ────────────────────────────────────────────────
analyze_clicked = st.button("🚀 Analyze", type="primary")
if analyze_clicked and url:
    try:
        with st.spinner("Scraping and analyzing content..."):
            (
                content,
                processed,
                keywords,
                topics,
                readability,
                sentiment,
                summary_result,
                nlp,
            ) = run_analysis(url, num_topics, num_keywords)

        entity_summary = nlp.get_entity_summary(processed)
        st.session_state["analysis_bundle"] = {
            "content": content,
            "keywords": keywords,
            "topics": topics,
            "readability": readability,
            "sentiment": sentiment,
            "summary_result": summary_result,
        }
        st.session_state["analyzed_url"] = url.strip()
        st.session_state["chat_context_text"] = content.text
        st.session_state["chat_context_title"] = (content.title or "").strip() or None
        st.session_state["chat_history"] = []
        st.session_state["chat_dialog_open"] = False
        st.session_state["sidebar_text_stats"] = dict(readability.text_statistics)
        st.session_state["sidebar_entity_summary"] = entity_summary
        st.success("✅ Analysis complete!")

    except ValueError as exc:
        for _k in (
            "analysis_bundle",
            "analyzed_url",
            "chat_context_text",
            "chat_context_title",
            "chat_history",
            "chat_dialog_open",
            "sidebar_text_stats",
            "sidebar_entity_summary",
        ):
            st.session_state.pop(_k, None)
        st.error(f"❌ Extraction failed: {exc}")
    except Exception as exc:
        for _k in (
            "analysis_bundle",
            "analyzed_url",
            "chat_context_text",
            "chat_context_title",
            "chat_history",
            "chat_dialog_open",
            "sidebar_text_stats",
            "sidebar_entity_summary",
        ):
            st.session_state.pop(_k, None)
        st.error(f"❌ An error occurred: {exc}")

elif analyze_clicked and not url.strip():
    st.warning("Please enter a URL to analyze.")

bundle = st.session_state.get("analysis_bundle")
_analyzed = (st.session_state.get("analyzed_url") or "").strip()
if (
    bundle
    and url.strip()
    and url.strip() == _analyzed
):
    content = bundle["content"]
    keywords = bundle["keywords"]
    topics = bundle["topics"]
    readability = bundle["readability"]
    sentiment = bundle["sentiment"]
    summary_result = bundle["summary_result"]

    _, _ask_col = st.columns([0.86, 0.14])
    with _ask_col:
        if st.button(
            "💬 Ask",
            type="primary",
            key="main_open_chat",
            help="Open Q and A about this page",
        ):
            st.session_state["chat_dialog_open"] = True
            st.rerun()

    # ── Content Metadata ────────────────────────────────────
    st.header("📄 Extracted Content")
    col1, col2, col3 = st.columns(3)
    col1.metric("Word Count", content.word_count)
    col2.metric("Extraction Method", content.extraction_method)
    col3.metric(
        "Reading Time",
        f"{fmt2(readability.estimated_reading_time_minutes)} min",
    )

    if content.title:
        st.subheader(content.title)
    if content.author:
        st.caption(f"Author: {content.author}")
    if content.date:
        st.caption(f"Date: {content.date}")

    with st.expander("📝 View Extracted Text"):
        st.text(content.text[:3000] + ("..." if len(content.text) > 3000 else ""))

    # ── Summary ─────────────────────────────────────────────
    st.header("📋 Summary")
    st.caption(
        f"**Method:** `{summary_result.method}` · "
        f"Sentences: **{summary_result.sentence_count}**"
    )
    st.write(summary_result.summary)

    # ── Readability Analysis ────────────────────────────────
    st.header("📊 Readability Analysis")
    st.info(f"**Reading Level:** {readability.reading_level}")

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("Flesch Reading Ease", fmt2(readability.flesch_reading_ease))
    r_col2.metric("Flesch-Kincaid Grade", fmt2(readability.flesch_kincaid_grade))
    r_col3.metric("Gunning Fog Index", fmt2(readability.gunning_fog))
    r_col4.metric("SMOG Index", fmt2(readability.smog_index))

    st.plotly_chart(readability_radar_fig(readability), use_container_width=True)

    # ── Sentiment (VADER) ───────────────────────────────────
    st.header("📊 Sentiment (VADER)")
    o = sentiment.to_dict()["overall"]
    st.metric(
        label=f"Overall: {o['label']}",
        value=f"compound {fmt2(o['compound'])}",
        help="VADER compound in [-1, 1]; label uses ±0.05 thresholds.",
    )
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("Positive", fmt2(o["positive"]))
    s_col2.metric("Neutral", fmt2(o["neutral"]))
    s_col3.metric("Negative", fmt2(o["negative"]))

    dist = sentiment.distribution
    dist_df = pd.DataFrame(
        {
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Sentences": [
                dist.get("positive", 0),
                dist.get("neutral", 0),
                dist.get("negative", 0),
            ],
        }
    )
    st.subheader("Sentence-level distribution")
    pie = px.pie(
        dist_df,
        names="Sentiment",
        values="Sentences",
        title="Sentiment mix (sentence counts)",
    )
    pie.update_layout(height=360, margin=dict(t=50, b=20))
    st.plotly_chart(pie, use_container_width=True)

    # ── Keywords ────────────────────────────────────────────
    st.header("🔑 Keywords")

    if keywords.tfidf_keywords:
        kw_df = pd.DataFrame(keywords.tfidf_keywords)
        kw_df_display = kw_df.copy()
        kw_df_display["score"] = kw_df_display["score"].round(2)
        kc1, kc2 = st.columns(2)
        with kc1:
            st.bar_chart(kw_df.set_index("keyword")["score"])
        with kc2:
            freq = {
                row["keyword"]: float(row["score"])
                for _, row in kw_df.iterrows()
            }
            wc = WordCloud(
                width=900,
                height=400,
                background_color="white",
                colormap="viridis",
            ).generate_from_frequencies(freq)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title("Keyword word cloud (TF-IDF weights)")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        st.dataframe(kw_df_display, use_container_width=True)
    else:
        st.warning("No TF-IDF keywords extracted.")

    with st.expander("Named Entities"):
        if keywords.entity_keywords:
            ent_df = pd.DataFrame(keywords.entity_keywords)
            ent_disp = ent_df.copy()
            for col in ent_disp.select_dtypes(
                include=["float64", "float32"]
            ).columns:
                ent_disp[col] = ent_disp[col].round(2)
            st.dataframe(ent_disp, use_container_width=True)
        else:
            st.warning("No named entities found.")

    with st.expander("Noun Phrases"):
        if keywords.noun_phrase_keywords:
            for phrase in keywords.noun_phrase_keywords:
                st.markdown(f"- {phrase}")
        else:
            st.warning("No noun phrases extracted.")

    # ── Topics ──────────────────────────────────────────────
    st.header("📌 Topic Detection (LDA)")
    if topics.lda_topics:
        for topic in topics.lda_topics:
            st.markdown(f"**{topic['label']}**")
            st.markdown(", ".join(topic["words"]))
            st.markdown("---")
    else:
        st.warning("Topic detection produced no results.")


# ── Sidebar: text stats & entity summary (below Settings) ────────
stats_sidebar = st.session_state.get("sidebar_text_stats")
if stats_sidebar:
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Text Statistics")
    stat_rows = []
    for k, v in stats_sidebar.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            stat_rows.append({"Metric": label, "Value": fmt2(v)})
        else:
            stat_rows.append({"Metric": label, "Value": v})
    st.sidebar.dataframe(
        pd.DataFrame(stat_rows), use_container_width=True, hide_index=True
    )

ent_sidebar = st.session_state.get("sidebar_entity_summary")
if ent_sidebar is not None:
    _render_sidebar_entity_summary(ent_sidebar)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Powered by:** trafilatura, spaCy, scikit-learn, textstat, plotly"
)

if st.session_state.get("chat_dialog_open"):
    _page_chat_dialog()

# ── Compare URLs (optional module) ──────────────────────────────
st.divider()
st.header(f"🔀 Compare 2–3 URLs")
try:
    from src.analysis.comparator import ContentComparator
except ImportError:
    st.caption("Comparison module is not available.")
else:
    cmp_u1 = st.text_input("Comparison URL 1", key="cmp_u1")
    cmp_u2 = st.text_input("Comparison URL 2", key="cmp_u2")
    cmp_u3 = st.text_input("Comparison URL 3 (optional)", key="cmp_u3")
    if st.button("Run comparison", key="cmp_btn"):
        cmp_urls = [u.strip() for u in (cmp_u1, cmp_u2, cmp_u3) if u.strip()][:3]
        if len(cmp_urls) < 2:
            st.warning("Enter at least two URLs to compare.")
        else:
            comp = ContentComparator()
            payloads: list[dict] = []
            try:
                with st.spinner("Fetching and analyzing each URL..."):
                    for cu in cmp_urls:
                        (
                            c_content,
                            c_processed,
                            c_keywords,
                            c_topics,
                            c_readability,
                            c_sentiment,
                            c_summary,
                            c_nlp,
                        ) = run_analysis(cu, num_topics, num_keywords)
                        payloads.append(
                            build_analysis_record(
                                c_content,
                                c_processed,
                                c_keywords,
                                c_topics,
                                c_readability,
                                c_sentiment,
                                c_summary,
                                c_nlp,
                            )
                        )
                result = comp.compare(payloads)
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")
            else:
                st.subheader("Readability (side-by-side)")
                df_r = pd.DataFrame(result.readability)
                df_r_disp = round_numeric_df(df_r)
                st.dataframe(df_r_disp, use_container_width=True)
                metric_cols = [
                    c
                    for c in (
                        "flesch_reading_ease",
                        "gunning_fog",
                        "smog_index",
                    )
                    if c in df_r_disp.columns
                ]
                if metric_cols:
                    fig_cmp = px.bar(
                        df_r_disp,
                        x="url",
                        y=metric_cols,
                        barmode="group",
                        title="Readability metrics by URL",
                    )
                    fig_cmp.update_layout(
                        xaxis_title="URL",
                        yaxis_title="Score",
                        legend_title="Metric",
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

                st.subheader("Sentiment")
                df_s = pd.DataFrame(result.sentiment)
                df_s_disp = round_numeric_df(df_s)
                st.dataframe(df_s_disp, use_container_width=True)
                if not df_s_disp.empty and "compound" in df_s_disp.columns:
                    fig_s = px.bar(
                        df_s_disp,
                        x="url",
                        y="compound",
                        color="label",
                        title="Overall sentiment (compound) by URL",
                    )
                    st.plotly_chart(fig_s, use_container_width=True)

                st.subheader("Shared keywords")
                st.write(", ".join(result.common_keywords) or "(none)")

                st.subheader("Unique keywords by URL")
                for u, terms in result.unique_keywords_by_url.items():
                    st.markdown(f"**{u}:** {', '.join(terms[:25])}")

                st.subheader("Topic overlap")
                overlap = result.topic_overlap or {}
                shared = overlap.get("shared_topic_terms") or []
                if shared:
                    st.markdown("**Shared topic terms**")
                    st.caption(", ".join(shared))
                else:
                    st.caption("No shared topic terms across the selected pages.")

                pair_rows = overlap.get("pairwise") or []
                if pair_rows:
                    st.markdown("**Pairwise topic similarity**")
                    pair_df = pd.DataFrame(
                        [
                            {
                                "URL A": p.get("url_a", ""),
                                "URL B": p.get("url_b", ""),
                                "Jaccard Similarity": fmt2(
                                    float(p.get("topic_jaccard", 0) or 0)
                                ),
                                "Shared Terms Count": int(
                                    p.get("shared_term_count", 0) or 0
                                ),
                            }
                            for p in pair_rows
                        ]
                    )
                    st.dataframe(pair_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No pairwise topic comparisons available.")
