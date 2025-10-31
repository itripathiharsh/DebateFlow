# nodes/agent_a_node.py

import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import DebateState
from utils import log_node_event, log_round_entry, log_validation, is_argument_repeated
from nodes.llm_utils import get_debate_llm

# Initialize LLM using Gemini
llm = get_debate_llm()

AGENT_A_SYSTEM_PROMPT = (
    "You are a rigorous Scientist. You base your arguments on empirical evidence, "
    "risk assessment, public safety, and real-world consequences. "
    "Your responses must be concise, logical, and directly address the debate topic. "
    "Do not repeat previous points. Focus on new, grounded reasoning."
)

def agent_a_node(state: DebateState) -> Dict[str, Any]:
    """
    Agent A (Scientist) generates an argument for the current round.
    Only runs on odd-numbered rounds (1, 3, 5, 7).
    """
    current_round = state.current_round
    topic = state.topic
    memory_summary = state.memory_summary

    # Validate turn with logging
    if current_round % 2 == 0 or current_round > 8:
        log_validation("TurnCheck", f"Agent A should not speak in round {current_round}", False)
        raise ValueError(f"Agent A should not speak in round {current_round}")
    else:
        log_validation("TurnCheck", f"Agent A validated for round {current_round}", True)

    log_node_event("AgentA", f"Generating argument for round {current_round}")

    # Build prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", AGENT_A_SYSTEM_PROMPT),
        ("human", (
            f"Debate Topic: '{topic}'\n\n"
            f"Debate Summary So Far:\n{memory_summary}\n\n"
            "Provide your next original argument (1–2 sentences). "
            "Do not repeat prior points. Be specific and evidence-based."
        ))
    ])

    chain = prompt_template | llm | StrOutputParser()
    argument = chain.invoke({})

    # Validate for repetition with logging
    if is_argument_repeated(argument, state.history):
        log_validation("Repetition", f"Possible repeated argument detected in round {current_round}", False)
    else:
        log_validation("Repetition", f"Argument is original in round {current_round}", True)

    # Log round entry
    log_round_entry(current_round, "Scientist", argument)
    log_node_event("AgentA", f"Argument generated for round {current_round}")

    # Return updated state (LangGraph merges this)
    return {
        "history": state.history + [{
            "agent": "Scientist",
            "round": current_round,
            "argument": argument.strip()
        }],
        "current_round": current_round + 1
    }