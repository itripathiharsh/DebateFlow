# nodes/judge_node.py

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import DebateState
from utils import log_node_event, log_judgment, log_run_end
from nodes.llm_utils import get_judge_llm

# Initialize LLM using Gemini
llm = get_judge_llm()

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial, highly logical debate judge with expertise in both scientific reasoning "
    "and philosophical ethics. Your role is to evaluate a structured 8-round debate and declare a "
    "clear winner based on argument quality—not personal opinion."
)

JUDGE_HUMAN_PROMPT = """
Debate Topic: "{topic}"

Full Debate Transcript:
{transcript}

Evaluation Criteria:
- Logical coherence and internal consistency
- Use of evidence, principles, or sound reasoning
- Relevance to the topic
- Depth and originality of insight
- Responsiveness to the opponent's points (where applicable)

Instructions:
1. Write a 3–4 sentence neutral summary of the debate.
2. Declare a winner: either "Scientist" or "Philosopher".
3. Provide a 2–3 sentence justification that references specific strengths.

Format your response EXACTLY as follows:

[Summary]
<your summary here>

[Winner]
<Scientist or Philosopher>

[Reason]
<your justification here>
"""

def format_transcript(history):
    lines = []
    for entry in history:
        lines.append(f"[Round {entry['round']}] {entry['agent']}: {entry['argument']}")
    return "\n".join(lines)

def parse_judge_output(raw_output: str):
    """Parse the structured output from the judge LLM."""
    try:
        summary = raw_output.split("[Summary]")[1].split("[Winner]")[0].strip()
        winner = raw_output.split("[Winner]")[1].split("[Reason]")[0].strip()
        reason = raw_output.split("[Reason]")[1].strip()
        return summary, winner, reason
    except Exception as e:
        # Fallback if parsing fails
        log_node_event("Judge", f"Output parsing failed: {e}")
        # Default fallback
        return "Summary unavailable.", "Scientist", "Fallback decision due to parsing error."

def judge_node(state: DebateState) -> Dict[str, Any]:
    """
    Judge node: evaluates the full debate and declares a winner.
    Triggered after round 8.
    """
    log_node_event("Judge", "Evaluating complete debate")
    
    topic = state.topic
    transcript = format_transcript(state.history)

    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", JUDGE_HUMAN_PROMPT)
    ])

    chain = prompt | llm | StrOutputParser()
    raw_response = chain.invoke({
        "topic": topic,
        "transcript": transcript
    })

    summary, winner, reason = parse_judge_output(raw_response)

    # Enforce valid winner
    if winner not in ["Scientist", "Philosopher"]:
        winner = "Scientist"  # default fallback
        reason = "Winner label invalid; defaulted to Scientist."

    # Log final output
    log_judgment(summary, winner, reason)
    log_node_event("Judge", "Evaluation complete")
    log_run_end()  # End the run logging

    return {
        "final_summary": summary,
        "winner": winner
    }