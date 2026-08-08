"""
AI Resume Analyzer — Multi-Agent System (built with OpenAI Swarm)
------------------------------------------------------------------------
Objective:
  Analyze a resume and provide improvement suggestions.

Agents (each hands off to the next):
  1. Resume Parser     -> extracts structured info from the raw resume text
  2. ATS Score Agent    -> scores how well the resume would pass an
                           Applicant Tracking System
  3. Grammar Checker    -> flags grammar/phrasing issues
  4. Career Advisor     -> suggests missing skills + improvements

Output:
  - ATS score
  - Missing skills
  - Resume improvement suggestions

Tech Stack:
  Python, OpenAI Swarm, OpenAI API

BEFORE YOU RUN THIS:
  1. Install dependencies:
       pip install git+https://github.com/openai/swarm.git
       pip install python-dotenv pypdf
  2. Get an OpenAI API key: platform.openai.com/api-keys
  3. Create a ".env" file in this folder with:
       OPENAI_API_KEY=your_key_here
  4. Have your resume ready as a .txt or .pdf file
"""

import os
from dotenv import load_dotenv
from swarm import Swarm, Agent

load_dotenv()

client = Swarm()

MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Helper: read resume text (plain .txt, or .pdf via pypdf)
# ---------------------------------------------------------------------------

def load_resume_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


# ---------------------------------------------------------------------------
# HANDOFF FUNCTIONS
# ---------------------------------------------------------------------------

def handoff_to_ats_scorer():
    """Call once the resume has been parsed, to pass it to the ATS Score Agent."""
    return ats_score_agent

def handoff_to_grammar_checker():
    """Call once the ATS score is ready, to pass everything to the Grammar Checker."""
    return grammar_checker_agent

def handoff_to_career_advisor():
    """Call once grammar review is complete, to pass everything to the Career Advisor."""
    return career_advisor_agent


# ---------------------------------------------------------------------------
# AGENTS
# ---------------------------------------------------------------------------

resume_parser_agent = Agent(
    name="Resume Parser",
    model=MODEL,
    instructions=(
        "You are precise and detail-oriented. The user message contains a "
        "target role and raw resume text. Extract and organize it into clear "
        "sections: contact info, summary, skills, experience, education, "
        "projects. Once done, call handoff_to_ats_scorer to continue."
    ),
    functions=[handoff_to_ats_scorer],
)

ats_score_agent = Agent(
    name="ATS Score Agent",
    model=MODEL,
    instructions=(
        "You've reviewed thousands of resumes against Applicant Tracking "
        "Systems. Using the parsed resume above and the target role, give a "
        "numeric ATS compatibility score (0-100) with a short explanation of "
        "what helps or hurts it (formatting, keyword usage, structure). Once "
        "done, call handoff_to_grammar_checker to continue."
    ),
    functions=[handoff_to_grammar_checker],
)

grammar_checker_agent = Agent(
    name="Grammar Checker",
    model=MODEL,
    instructions=(
        "You are a sharp-eyed editor. Review the original resume text for "
        "grammar issues, inconsistent verb tenses, and awkward phrasing. "
        "List specific issues with suggested fixes. Once done, call "
        "handoff_to_career_advisor to continue."
    ),
    functions=[handoff_to_career_advisor],
)

career_advisor_agent = Agent(
    name="Career Advisor",
    model=MODEL,
    instructions=(
        "You are an experienced career coach. Using everything gathered so "
        "far, identify: 1) skills commonly expected for the target role that "
        "are missing from this resume, 2) 3-5 concrete, specific suggestions "
        "to strengthen it. Then present a final combined summary: ATS score, "
        "missing skills, and improvement suggestions. This is the final "
        "step — do not hand off to anyone else."
    ),
    functions=[],
)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_resume_analyzer(resume_text: str, target_role: str) -> str:
    user_message = (
        f"Target role: {target_role}\n\n"
        f"Resume text:\n{resume_text}"
    )
    messages = [{"role": "user", "content": user_message}]

    response = client.run(
        agent=resume_parser_agent,   # pipeline always starts at the Resume Parser
        messages=messages,
    )

    return response.messages[-1]["content"]


if __name__ == "__main__":
    resume_path = input("Enter path to your resume (.txt or .pdf): ").strip()
    target_role = input("Enter the role you're targeting (e.g. 'Data Analyst'): ").strip()

    resume_text = load_resume_text(resume_path)

    print("\nAgents are analyzing your resume...\n")
    result = run_resume_analyzer(resume_text, target_role)

    print("\n" + "=" * 70)
    print("RESUME ANALYSIS")
    print("=" * 70)
    print(result)
