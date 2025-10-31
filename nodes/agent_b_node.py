# nodes/agent_b_node.py

import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import DebateState
from utils import log_node_event, log_round_entry, log_validation, is_argument_repeated
from nodes.llm_utils import get_debate_llm

# Initialize LLM using Gemini
llm = get_debate_llm()

AGENT_B_SYSTEM_PROMPT = (
    "You are a thoughtful Philosopher. You focus on ethics, human autonomy, "
    "societal evolution, freedom of inquiry, and long-term civilizational values. "
    "Your arguments should be principled, nuanced, and avoid technical jargon. "
    "Do not repeat prior points. Offer original philosophical insight."
)

def agent_b_node(state: DebateState) -> Dict[str, Any]:
    """
    Agent B (Philosopher) generates an argument for the current round.
    Only runs on even-numbered rounds (2, 4, 6, 8).
    """
    current_round = state.current_round
    topic = state.topic
    memory_summary = state.memory_summary

    # Validate turn: Agent B speaks only on even rounds 2–8
    if current_round % 2 != 0 or current_round > 8 or current_round < 2:
        log_validation("TurnCheck", f"Agent B should not speak in round {current_round}", False)
        raise ValueError(f"Agent B should not speak in round {current_round}")
    else:
        log_validation("TurnCheck", f"Agent B validated for round {current_round}", True)

    log_node_event("AgentB", f"Generating argument for round {current_round}")

    # Build prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", AGENT_B_SYSTEM_PROMPT),
        ("human", (
            f"Debate Topic: '{topic}'\n\n"
            f"Debate Summary So Far:\n{memory_summary}\n\n"
            "Provide your next original philosophical argument (1–2 sentences). "
            "Do not repeat prior points. Focus on ethics, freedom, or societal implications."
        ))
    ])

    chain = prompt_template | llm | StrOutputParser()
    argument = chain.invoke({})

    # Check for repetition with logging
    if is_argument_repeated(argument, state.history):
        log_validation("Repetition", f"Possible repeated argument detected in round {current_round}", False)
    else:
        log_validation("Repetition", f"Argument is original in round {current_round}", True)

    # Log to file and console
    log_round_entry(current_round, "Philosopher", argument)
    log_node_event("AgentB", f"Argument generated for round {current_round}")

    # Return updated state (LangGraph merges this)
    return {
        "history": state.history + [{
            "agent": "Philosopher",
            "round": current_round,
            "argument": argument.strip()
        }],
        "current_round": current_round + 1
    }