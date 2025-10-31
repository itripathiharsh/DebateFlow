# nodes/memory_node.py

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import DebateState
from utils import log_node_event, log_memory_update
from nodes.llm_utils import get_summary_llm

# Initialize LLM for summarization using Gemini
llm = get_summary_llm()

SUMMARIZER_PROMPT = """
You are summarizing an ongoing structured debate between a Scientist and a Philosopher.

Debate Topic: "{topic}"

Full Transcript So Far:
{transcript}

Instructions:
- Condense the entire debate into 2–3 clear, neutral sentences.
- Preserve key claims from both sides.
- Do NOT add new opinions or judgments.
- Focus on what was argued, not who won.

Summary:
"""

def format_transcript(history):
    """Convert history list into readable transcript."""
    lines = []
    for entry in history:
        lines.append(f"[Round {entry['round']}] {entry['agent']}: {entry['argument']}")
    return "\n".join(lines)

def memory_node(state: DebateState) -> Dict[str, Any]:
    """
    Memory node: updates the debate summary after each agent speaks.
    Triggered after Agent A or Agent B generates an argument.
    """
    log_node_event("Memory", "Generating incremental summary")
    
    topic = state.topic
    transcript = format_transcript(state.history)

    if not state.history:
        # Edge case: no history yet
        new_summary = "Debate has not started."
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("human", SUMMARIZER_PROMPT)
        ])
        chain = prompt | llm | StrOutputParser()
        new_summary = chain.invoke({
            "topic": topic,
            "transcript": transcript
        }).strip()

    # Create persona views for logging
    persona_views = {
        "Scientist": "Focusing on empirical evidence and practical consequences",
        "Philosopher": "Emphasizing ethical principles and human values"
    }

    log_memory_update(new_summary, persona_views)
    log_node_event("Memory", "Summary updated successfully")

    return {
        "memory_summary": new_summary
    }