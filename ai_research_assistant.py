"""
AI Research Assistant — Multi-Agent System (built with OpenAI Swarm)
------------------------------------------------------------------------
Objective:
  The user asks a question. Four agents collaborate — each one handing
  off control to the next — to produce a structured, fact-checked report.

    1. Research Agent      -> finds information on the topic
    2. Summarizer Agent    -> condenses findings into key points
    3. Fact Checker Agent  -> verifies the summary is consistent
    4. Report Generator    -> writes the final structured report

Tech Stack:
  Python, OpenAI Swarm, OpenAI API

A NOTE ON SWARM:
  Swarm is OpenAI's lightweight, educational multi-agent framework, built
  on two ideas: Agents (instructions + functions) and handoffs (a function
  that returns another Agent to transfer control to). It's officially
  deprecated in favor of OpenAI's newer Agents SDK, but it's still widely
  used for learning multi-agent patterns since it's so small and readable
  — which is exactly why it fits a project like this.

BEFORE YOU RUN THIS:
  1. Install dependencies:
       pip install git+https://github.com/openai/swarm.git
       pip install python-dotenv
  2. Get an OpenAI API key: platform.openai.com/api-keys
  3. Create a ".env" file in this folder with:
       OPENAI_API_KEY=your_key_here
"""

import os
from dotenv import load_dotenv
from swarm import Swarm, Agent

load_dotenv()

client = Swarm()

MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# HANDOFF FUNCTIONS
# Each function returns the *next* agent, transferring control to it.
# ---------------------------------------------------------------------------

def handoff_to_summarizer():
    """Call this once research is complete, to pass findings to the Summarizer Agent."""
    return summarizer_agent

def handoff_to_fact_checker():
    """Call this once the summary is ready, to pass it to the Fact Checker Agent."""
    return fact_checker_agent

def handoff_to_report_generator():
    """Call this once fact-checking is complete, to pass everything to the Report Generator."""
    return report_generator_agent


# ---------------------------------------------------------------------------
# AGENTS
# ---------------------------------------------------------------------------

research_agent = Agent(
    name="Research Agent",
    model=MODEL,
    instructions=(
        "You are a meticulous researcher. Given the user's question, gather "
        "key facts, relevant context, and important details needed to answer "
        "it thoroughly. Be specific and organized. Once your research is "
        "complete, call handoff_to_summarizer to pass your findings along."
    ),
    functions=[handoff_to_summarizer],
)

summarizer_agent = Agent(
    name="Summarizer Agent",
    model=MODEL,
    instructions=(
        "You specialize in distilling research into clear, concise summaries. "
        "Take the research findings above and condense them into 5-8 clear "
        "bullet points, organized by sub-topic. Once done, call "
        "handoff_to_fact_checker to pass the summary along for verification."
    ),
    functions=[handoff_to_fact_checker],
)

fact_checker_agent = Agent(
    name="Fact Checker Agent",
    model=MODEL,
    instructions=(
        "You are a skeptical editor. Review the summary above against the "
        "original research. Flag anything unsupported, contradictory, or "
        "exaggerated, and correct it if needed. Once done, call "
        "handoff_to_report_generator to pass the verified summary along."
    ),
    functions=[handoff_to_report_generator],
)

report_generator_agent = Agent(
    name="Report Generator Agent",
    model=MODEL,
    instructions=(
        "You are a professional technical writer. Using the fact-checked "
        "summary above, write a final structured report in Markdown with: "
        "1) A short executive summary (2-3 sentences), 2) Key findings under "
        "clear headers, 3) A short conclusion. This is the final step — do "
        "not hand off to anyone else."
    ),
    functions=[],
)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_research_assistant(user_question: str) -> str:
    messages = [{"role": "user", "content": user_question}]

    response = client.run(
        agent=research_agent,   # pipeline always starts at the Research Agent
        messages=messages,
    )

    return response.messages[-1]["content"]


if __name__ == "__main__":
    question = input("What would you like me to research? ")
    print("\nAgents are working on your report...\n")

    final_report = run_research_assistant(question)

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(final_report)
