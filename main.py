import os
import re

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI HR Recruitment Agent",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Page styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        .hero {
            padding: 1.5rem 1.7rem;
            border-radius: 16px;
            background: linear-gradient(120deg, #14213d 0%, #234b75 100%);
            color: white;
            margin-bottom: 1.2rem;
        }
        .hero h1 {color: white; margin-bottom: .35rem;}
        .hero p {color: #e7eef8; margin-bottom: 0;}
        .result-card {
            border: 1px solid rgba(128,128,128,.28);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            min-height: 125px;
        }
        .result-label {font-size: .85rem; opacity: .75; margin-bottom: .35rem;}
        .result-value {font-size: 1.35rem; font-weight: 700;}
        .muted {opacity: .72; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------
class State(TypedDict, total=False):
    application: str
    experience_level: str
    skill_match: str
    response: str


def get_api_key():
    """Read the OpenAI key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("OPENAI_API_KEY", "")


@st.cache_resource(show_spinner=False)
def build_workflow(api_key: str):
    """Create and compile the recruitment workflow."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    def categorize_experience(state: State) -> State:
        prompt = ChatPromptTemplate.from_template(
            """
            Categorize the candidate's professional experience using the
            application below.

            Choose exactly one category:
            - Entry-level
            - Mid-level
            - Senior-level

            Return only the category name, with no explanation.

            Application:
            {application}
            """
        )
        result = (prompt | llm).invoke(
            {"application": state["application"]}
        ).content.strip()

        # Normalize the model response to one of the graph's expected values.
        normalized = result.lower()
        if "senior" in normalized:
            level = "Senior-level"
        elif "mid" in normalized:
            level = "Mid-level"
        elif "entry" in normalized:
            level = "Entry-level"
        else:
            level = "Unable to classify"

        return {"experience_level": level}

    def assess_skillset(state: State) -> State:
        prompt = ChatPromptTemplate.from_template(
            """
            Assess whether the candidate's application demonstrates skills
            suitable for a Python Developer role.

            Choose exactly one:
            - Match
            - No Match

            Return only the selected phrase, with no explanation.

            Application:
            {application}
            """
        )
        result = (prompt | llm).invoke(
            {"application": state["application"]}
        ).content.strip()

        normalized = result.lower()
        skill_match = "Match" if re.search(r"\bmatch\b", normalized) and "no match" not in normalized else "No Match"
        return {"skill_match": skill_match}

    def schedule_hr_interview(state: State) -> State:
        return {
            "response": "Candidate has been shortlisted for an HR interview."
        }

    def escalate_to_recruiter(state: State) -> State:
        return {
            "response": (
                "Candidate has senior-level experience but doesn't match "
                "the assessed job skills. Escalate to a recruiter for review."
            )
        }

    def reject_application(state: State) -> State:
        return {
            "response": "Candidate doesn't meet the assessed job requirements and is not shortlisted."
        }

    def route_app(state: State) -> str:
        if state.get("skill_match") == "Match":
            return "schedule_hr_interview"
        elif state.get("experience_level") == "Senior-level":
            return "escalate_to_recruiter"
        return "reject_application"

    workflow = StateGraph(State)

    workflow.add_node("categorize_experience", categorize_experience)
    workflow.add_node("assess_skillset", assess_skillset)
    workflow.add_node("schedule_hr_interview", schedule_hr_interview)
    workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
    workflow.add_node("reject_application", reject_application)

    workflow.add_edge(START, "categorize_experience")
    workflow.add_edge("categorize_experience", "assess_skillset")

    workflow.add_conditional_edges(
        "assess_skillset",
        route_app,
        {
            "schedule_hr_interview": "schedule_hr_interview",
            "escalate_to_recruiter": "escalate_to_recruiter",
            "reject_application": "reject_application",
        },
    )

    workflow.add_edge("schedule_hr_interview", END)
    workflow.add_edge("escalate_to_recruiter", END)
    workflow.add_edge("reject_application", END)

    return workflow.compile()


def run_candidate_screening(graph, application: str):
    """Run the candidate application through the LangGraph workflow."""
    return graph.invoke({"application": application})


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧑‍💼 AI HR Recruitment Agent</h1>
        <p>Screen candidate applications, assess experience and Python skills,
        and generate a recruitment recommendation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About this application")
    st.write(
        "This application uses an OpenAI language model and a LangGraph "
        "workflow to assess candidate applications."
    )
    st.divider()
    st.subheader("Screening workflow")
    st.markdown(
        """
        1. Categorize experience
        2. Assess Python Developer skill match
        3. Route to the appropriate outcome
        """
    )
    st.divider()
    st.caption("Model: GPT-4o-mini")
    st.caption("Workflow: LangGraph")
    st.caption("Interface: Streamlit")

st.subheader("Candidate application")

st.caption(
    "Paste the candidate's resume text, application summary or relevant "
    "professional experience below."
)

with st.form("candidate_screening_form"):
    application_text = st.text_area(
        "Candidate resume / application",
        height=250,
        placeholder=(
            "Example: I have 5 years of experience in software development "
            "with expertise in Python, Django, REST APIs and SQL..."
        ),
    )
    submitted = st.form_submit_button(
        "🔍 Analyse Candidate",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not application_text.strip():
        st.warning("Please enter the candidate's application before proceeding.")
    else:
        api_key = get_api_key()

        if not api_key:
            st.error(
                "OpenAI API key not found. Add OPENAI_API_KEY to your "
                "Streamlit secrets or environment variables."
            )
        else:
            try:
                with st.spinner(
                    "AI agent is reviewing the application and evaluating the candidate..."
                ):
                    graph = build_workflow(api_key)
                    results = run_candidate_screening(
                        graph, application_text.strip()
                    )

                st.success("Candidate assessment completed.")

                st.subheader("Screening results")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">Experience classification</div>
                            <div class="result-value">{results.get("experience_level", "Not available")}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">Python skill assessment</div>
                            <div class="result-value">{results.get("skill_match", "Not available")}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.write("")
                st.subheader("Recruitment recommendation")

                recommendation = results.get("response", "No recommendation returned.")
                if "shortlisted" in recommendation.lower():
                    st.success("✅ " + recommendation)
                elif "escalate" in recommendation.lower():
                    st.warning("⚠️ " + recommendation)
                else:
                    st.error("❌ " + recommendation)

                with st.expander("View submitted application"):
                    st.write(application_text)

                st.download_button(
                    "⬇️ Download assessment",
                    data=(
                        "AI HR Recruitment Agent - Candidate Assessment\n\n"
                        f"Application:\n{application_text}\n\n"
                        f"Experience level: {results.get('experience_level')}\n"
                        f"Python skill match: {results.get('skill_match')}\n"
                        f"Recommendation: {recommendation}\n"
                    ),
                    file_name="candidate_assessment.txt",
                    mime="text/plain",
                )

                st.caption(
                    "Important: This is an AI-assisted screening recommendation, "
                    "not a final hiring decision. A qualified recruiter should "
                    "review the application and assessment before taking action."
                )

            except Exception as exc:
                st.error(
                    "The screening process could not be completed. "
                    "Check your API key, internet connection and package versions."
                )
                with st.expander("Technical error details"):
                    st.exception(exc)

st.divider()
st.markdown(
    '<p class="muted">AI HR Recruitment Agent | Candidate screening prototype</p>',
    unsafe_allow_html=True,
)
