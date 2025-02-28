# app.py

import os
import streamlit as st
import arxiv
import networkx as nx
import matplotlib.pyplot as plt
import datetime

# -------------------------------
# Groq API Client
# -------------------------------
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# -------------------------------
# Helper Functions (Groq-based)
# -------------------------------
def groq_summarize(text: str) -> str:
    """
    Summarize the given text using Groq's chat completion API.
    Adjust the prompt or model as needed.
    """
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following text in detail:\n\n{text}"
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return response.choices[0].message.content.strip()

def groq_generate(text: str) -> str:
    """
    Generate text (e.g., research proposals) using Groq's chat completion API.
    Adjust the prompt or model as needed.
    """
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": text
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return response.choices[0].message.content.strip()

# -------------------------------
# Existing Helper Functions
# -------------------------------
def retrieve_papers(query, max_results=5):
    """Retrieve academic papers from arXiv."""
    search = arxiv.Search(query=query, max_results=max_results)
    papers = []
    for result in search.results():
        paper = {
            "title": result.title,
            "summary": result.summary,
            "url": result.pdf_url,
            "authors": [author.name for author in result.authors],
            "published": result.published
        }
        papers.append(paper)
    return papers

def summarize_text(text):
    """
    Wrap the groq_summarize function so it's easy to switch 
    implementations if needed.
    """
    return groq_summarize(text)

def generate_concept_map(papers):
    """Create a concept map (graph) based on author connections."""
    G = nx.Graph()
    for paper in papers:
        G.add_node(paper['title'])
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            if set(papers[i]['authors']) & set(papers[j]['authors']):
                G.add_edge(papers[i]['title'], papers[j]['title'])
    return G

def generate_citation(paper):
    """Generate APA-style citation for a paper."""
    authors = ", ".join(paper['authors'])
    if isinstance(paper['published'], datetime.datetime):
        year = paper['published'].year
    else:
        year = "n.d."
    return f"{authors} ({year}). {paper['title']}. Retrieved from {paper['url']}"

def generate_proposal_suggestions(text):
    """
    Generate novel research proposal suggestions based on text,
    wrapping the groq_generate function.
    """
    prompt = (
        f"Based on this research summary:\n\n{text}\n\n"
        "Propose novel research directions:"
    )
    return groq_generate(prompt)

def get_cached_summary(paper_id, text):
    """
    Retrieve or create a cached summary for a given paper.
    This ensures each paper's summary is generated only once.
    """
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {}
    if paper_id not in st.session_state.summaries:
        st.session_state.summaries[paper_id] = summarize_text(text)
    return st.session_state.summaries[paper_id]

# -------------------------------
# Streamlit Interface
# -------------------------------
st.title("📚 PaperPilot – Intelligent Academic Navigator")

# Add the Overview subheading
st.write("""
PaperPilot is an intelligent academic navigator designed to simplify your research workflow. 
With a single query, it fetches relevant academic papers and provides you with a 
comprehensive toolkit to explore them in depth. You can read a quick summary of each article, 
view a visual concept map to see how different papers are interlinked, generate properly 
formatted citations, and even receive suggestions for novel research proposals. By integrating 
state-of-the-art AI models, PaperPilot streamlines the entire literature review process—making 
it easier to stay organized, discover new insights, and advance your academic endeavors.
""")

# ---------------------------------
# Sidebar: Search & Navigation
# ---------------------------------
with st.sidebar:
    st.header("🔍 Search Parameters")
    query = st.text_input("Research topic or question:")
    
    if st.button("🚀 Find Articles"):
        if query.strip():
            with st.spinner("Searching arXiv..."):
                papers = retrieve_papers(query)
                if papers:
                    st.session_state.papers = papers
                    st.success(f"Found {len(papers)} papers!")
                    # Default to showing articles after retrieval
                    st.session_state.active_section = "articles"
                else:
                    st.error("No papers found. Try different keywords.")
        else:
            st.warning("Please enter a search query")

    # Navigation buttons (only relevant if we have papers in session)
    if 'papers' in st.session_state and st.session_state.papers:
        st.header("🔀 Navigation")
        if st.button("📑 Show Articles"):
            st.session_state.active_section = "articles"
        if st.button("📚 Literature Review & Summary"):
            st.session_state.active_section = "review"
        if st.button("🔍 Concept & Visual Graph"):
            st.session_state.active_section = "graph"
        if st.button("📝 Formatted Citations"):
            st.session_state.active_section = "citations"
        if st.button("💡 Research Proposal"):
            st.session_state.active_section = "proposal"

# ---------------------------------
# Main Content Area
# ---------------------------------
if 'active_section' not in st.session_state:
    st.session_state.active_section = "none"

if 'papers' in st.session_state and st.session_state.papers:
    papers = st.session_state.papers

    # ---------------------------------
    # 1) Show Articles
    # ---------------------------------
    if st.session_state.active_section == "articles":
        st.header("📑 Retrieved Papers")
        for idx, paper in enumerate(papers, 1):
            with st.expander(f"{idx}. {paper['title']}"):
                st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                if isinstance(paper['published'], datetime.datetime):
                    pub_date = paper['published'].strftime('%Y-%m-%d')
                else:
                    pub_date = "n.d."
                st.markdown(f"**Published:** {pub_date}")
                st.markdown(f"**Link:** [PDF Link]({paper['url']})")
                st.markdown("**Abstract:**")
                st.write(paper['summary'])

    # ---------------------------------
    # 2) Literature Review & Summary
    # ---------------------------------
    elif st.session_state.active_section == "review":
        st.header("📚 Literature Review & Summary")
        combined_summary = ""

        for idx, paper in enumerate(papers, 1):
            with st.expander(f"Summary: {paper['title']}", expanded=False):
                with st.spinner(f"Analyzing {paper['title']}..."):
                    paper_id = f"paper_{idx}"
                    summary = get_cached_summary(paper_id, paper['summary'])
                    st.write(summary)
                    combined_summary += summary + "\n\n"

        st.session_state.combined_summary = combined_summary

    # ---------------------------------
    # 3) Concept & Visual Graph
    # ---------------------------------
    elif st.session_state.active_section == "graph":
        st.header("🔍 Concept & Visual Graph")
        st.write(
            "Below is a concept map that visualizes how the authors are "
            "connected across the retrieved articles. Each node represents a paper, "
            "and edges indicate shared authors."
        )

        with st.spinner("Generating concept map..."):
            G = generate_concept_map(papers)
            if G.nodes():
                fig, ax = plt.subplots(figsize=(10, 8))
                pos = nx.spring_layout(G, k=0.5, seed=42)
                nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000, ax=ax)
                nx.draw_networkx_edges(G, pos, edge_color='#666666', ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("No significant connections found between papers.")

    # ---------------------------------
    # 4) Formatted Citations
    # ---------------------------------
    elif st.session_state.active_section == "citations":
        st.header("📝 Formatted Citations (APA Style)")
        for paper in papers:
            st.markdown(f"- {generate_citation(paper)}")

    # ---------------------------------
    # 5) Research Proposal
    # ---------------------------------
    elif st.session_state.active_section == "proposal":
        st.header("💡 Research Proposal Suggestions")

        # Make sure we have a combined summary for the proposals
        if 'combined_summary' not in st.session_state:
            with st.spinner("Synthesizing research overview..."):
                full_text = "\n".join([p['summary'] for p in papers])
                st.session_state.combined_summary = summarize_text(full_text)

        with st.spinner("Generating innovative ideas..."):
            proposal = generate_proposal_suggestions(st.session_state.combined_summary[:2000])
        st.write(proposal)

    else:
        st.info("Please select an option from the sidebar to begin.")
else:
    st.info("Enter a query in the sidebar and click 'Find Articles' to get started.")

st.caption("Built with ❤️ using AI")
