```python
import os
import datetime

import streamlit as st
import arxiv
import networkx as nx
import matplotlib.pyplot as plt

from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PaperPilot",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# GROQ API CLIENT
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Please add it to your Streamlit Cloud Secrets."
    )
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# GROQ HELPER FUNCTIONS
# ============================================================

def groq_summarize(text: str) -> str:
    """
    Summarize text using Groq.
    """

    response = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize the following academic text clearly and "
                    "concisely. Highlight the main objective, methodology, "
                    "important findings, and conclusions.\n\n"
                    f"{text}"
                )
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return response.choices[0].message.content.strip()


def groq_generate(text: str) -> str:
    """
    Generate text using Groq.
    """

    response = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": text
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return response.choices[0].message.content.strip()


# ============================================================
# ARXIV PAPER RETRIEVAL
# ============================================================

def retrieve_papers(query, max_results=5):
    """
    Retrieve academic papers from arXiv.

    Uses arxiv.Client() for reliable API requests and retries.
    """

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3.0,
        num_retries=5
    )

    papers = []

    try:

        for result in client.results(search):

            paper = {
                "title": result.title,
                "summary": result.summary,
                "url": result.pdf_url,
                "authors": [author.name for author in result.authors],
                "published": result.published
            }

            papers.append(paper)

    except arxiv.HTTPError as e:

        st.error(
            "arXiv is currently unavailable or has temporarily "
            "rejected the request. Please try again in a moment."
        )

        # Show technical information while debugging
        st.caption(f"Technical error: {str(e)}")

        return []

    except Exception as e:

        st.error(
            "Something went wrong while retrieving papers from arXiv."
        )

        # Useful during deployment/debugging
        st.caption(f"Technical error: {type(e).__name__}: {str(e)}")

        return []

    return papers


# ============================================================
# TEXT SUMMARIZATION
# ============================================================

def summarize_text(text):
    """
    Wrapper around Groq summarization.
    """

    return groq_summarize(text)


# ============================================================
# AUTHOR CONNECTION GRAPH
# ============================================================

def generate_concept_map(papers):
    """
    Create a graph where:
    - Each node represents a paper.
    - An edge means two papers share at least one author.
    """

    G = nx.Graph()

    # Add paper nodes
    for paper in papers:
        G.add_node(paper["title"])

    # Add connections based on shared authors
    for i in range(len(papers)):

        for j in range(i + 1, len(papers)):

            authors_i = set(papers[i]["authors"])
            authors_j = set(papers[j]["authors"])

            if authors_i & authors_j:

                G.add_edge(
                    papers[i]["title"],
                    papers[j]["title"]
                )

    return G


# ============================================================
# CITATION GENERATION
# ============================================================

def generate_citation(paper):
    """
    Generate a simple APA-style citation.
    """

    authors = ", ".join(paper["authors"])

    if isinstance(paper["published"], datetime.datetime):
        year = paper["published"].year
    else:
        year = "n.d."

    return (
        f"{authors} ({year}). "
        f"{paper['title']}. "
        f"Retrieved from {paper['url']}"
    )


# ============================================================
# RESEARCH PROPOSAL GENERATION
# ============================================================

def generate_proposal_suggestions(text):
    """
    Generate research directions based on a research summary.
    """

    prompt = f"""
You are an academic research assistant.

Based on the following research summary, suggest several
potential research directions.

For each direction, briefly explain:
1. The research problem
2. Why it is interesting
3. A possible methodology

Research summary:

{text}

Generate clear and practical research ideas.
"""

    return groq_generate(prompt)


# ============================================================
# SUMMARY CACHE
# ============================================================

def get_cached_summary(paper_id, text):
    """
    Generate a paper summary only once per session.
    """

    if "summaries" not in st.session_state:
        st.session_state.summaries = {}

    if paper_id not in st.session_state.summaries:

        st.session_state.summaries[paper_id] = summarize_text(text)

    return st.session_state.summaries[paper_id]


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "active_section" not in st.session_state:
    st.session_state.active_section = "none"

if "papers" not in st.session_state:
    st.session_state.papers = []

if "summaries" not in st.session_state:
    st.session_state.summaries = {}


# ============================================================
# MAIN TITLE
# ============================================================

st.title("📚 PaperPilot – Intelligent Academic Navigator")

st.write(
    """
PaperPilot is an intelligent academic navigator designed to simplify
your research workflow. With a single query, it fetches relevant
academic papers and provides tools to explore them in depth.

You can read paper abstracts, generate AI-powered summaries,
visualize author connections, create citations, and receive
suggestions for potential research directions.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔍 Search Parameters")

    query = st.text_input(
        "Research topic or question:",
        placeholder="e.g., machine learning in water quality"
    )

    if st.button("🚀 Find Articles"):

        if query.strip():

            with st.spinner("Searching arXiv..."):

                papers = retrieve_papers(
                    query.strip(),
                    max_results=5
                )

            if papers:

                st.session_state.papers = papers

                # Clear old summaries when a new search is performed
                st.session_state.summaries = {}

                # Clear old combined summary
                if "combined_summary" in st.session_state:
                    del st.session_state.combined_summary

                st.session_state.active_section = "articles"

                st.success(
                    f"Found {len(papers)} papers!"
                )

            else:

                st.warning(
                    "No papers were retrieved. "
                    "Try different keywords or try again later."
                )

        else:

            st.warning(
                "Please enter a research query."
            )


    # ========================================================
    # NAVIGATION
    # ========================================================

    if st.session_state.papers:

        st.header("🔀 Navigation")

        if st.button("📑 Show Articles"):

            st.session_state.active_section = "articles"

        if st.button("📚 Literature Review & Summary"):

            st.session_state.active_section = "review"

        if st.button("🔍 Author Connection Graph"):

            st.session_state.active_section = "graph"

        if st.button("📝 Formatted Citations"):

            st.session_state.active_section = "citations"

        if st.button("💡 Research Proposal"):

            st.session_state.active_section = "proposal"


# ============================================================
# MAIN CONTENT
# ============================================================

papers = st.session_state.papers


# ============================================================
# 1. RETRIEVED ARTICLES
# ============================================================

if papers and st.session_state.active_section == "articles":

    st.header("📑 Retrieved Papers")

    for idx, paper in enumerate(papers, 1):

        with st.expander(
            f"{idx}. {paper['title']}"
        ):

            st.markdown(
                f"**Authors:** "
                f"{', '.join(paper['authors'])}"
            )

            if isinstance(
                paper["published"],
                datetime.datetime
            ):

                pub_date = paper["published"].strftime(
                    "%Y-%m-%d"
                )

            else:

                pub_date = "n.d."

            st.markdown(
                f"**Published:** {pub_date}"
            )

            st.markdown(
                f"**Link:** [Open PDF]({paper['url']})"
            )

            st.markdown("**Abstract:**")

            st.write(
                paper["summary"]
            )


# ============================================================
# 2. LITERATURE REVIEW & SUMMARY
# ============================================================

elif papers and st.session_state.active_section == "review":

    st.header("📚 Literature Review & Summary")

    combined_summary = ""

    for idx, paper in enumerate(papers, 1):

        with st.expander(
            f"Summary: {paper['title']}"
        ):

            with st.spinner(
                f"Analyzing {paper['title']}..."
            ):

                paper_id = f"paper_{idx}"

                summary = get_cached_summary(
                    paper_id,
                    paper["summary"]
                )

                st.write(summary)

                combined_summary += (
                    summary + "\n\n"
                )

    st.session_state.combined_summary = combined_summary


# ============================================================
# 3. AUTHOR CONNECTION GRAPH
# ============================================================

elif papers and st.session_state.active_section == "graph":

    st.header("🔍 Author Connection Graph")

    st.write(
        """
Each node represents a paper. An edge connects two papers
when they have at least one author in common.
"""
    )

    with st.spinner(
        "Generating author connection graph..."
    ):

        G = generate_concept_map(papers)

        if G.nodes():

            fig, ax = plt.subplots(
                figsize=(12, 8)
            )

            pos = nx.spring_layout(
                G,
                k=0.5,
                seed=42
            )

            nx.draw_networkx_nodes(
                G,
                pos,
                node_color="skyblue",
                node_size=2000,
                ax=ax
            )

            nx.draw_networkx_edges(
                G,
                pos,
                edge_color="#666666",
                ax=ax
            )

            nx.draw_networkx_labels(
                G,
                pos,
                font_size=8,
                ax=ax
            )

            ax.axis("off")

            st.pyplot(
                fig,
                clear_figure=True
            )

        else:

            st.info(
                "No connections were found between the retrieved papers."
            )


# ============================================================
# 4. FORMATTED CITATIONS
# ============================================================

elif papers and st.session_state.active_section == "citations":

    st.header("📝 Formatted Citations")

    for paper in papers:

        st.markdown(
            f"- {generate_citation(paper)}"
        )


# ============================================================
# 5. RESEARCH PROPOSAL
# ============================================================

elif papers and st.session_state.active_section == "proposal":

    st.header("💡 Research Proposal Suggestions")

    # Create combined summary if it doesn't exist
    if "combined_summary" not in st.session_state:

        with st.spinner(
            "Synthesizing research overview..."
        ):

            full_text = "\n\n".join(
                [
                    paper["summary"]
                    for paper in papers
                ]
            )

            st.session_state.combined_summary = (
                summarize_text(full_text)
            )

    # Generate research ideas
    with st.spinner(
        "Generating innovative research ideas..."
    ):

        proposal = generate_proposal_suggestions(
            st.session_state.combined_summary[:4000]
        )

    st.write(proposal)


# ============================================================
# NO PAPERS / INITIAL STATE
# ============================================================

elif not papers:

    st.info(
        "Enter a research topic in the sidebar "
        "and click 'Find Articles' to get started."
    )


# ============================================================
# FOOTER
# ============================================================

st.caption("Built with ❤️ using AI")
```
