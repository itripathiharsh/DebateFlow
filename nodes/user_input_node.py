# nodes/user_input_node.py

from typing import Dict, Any
from state import DebateState
from utils import log_node_event, log_run_start

def user_input_node(state: DebateState) -> Dict[str, Any]:
    """
    User Input Node: prompts the user for a debate topic and initializes the state.
    This is the entry point of the graph.
    """
    print("\nATG Multi-Agent Debate System")
    print("-" * 40)
    
    # Get topic from user
    topic = input("Enter topic for debate: ").strip()
    
    # Validate input - use consistent topic for reproducible runs
    if not topic:
        topic = "Should AI development be regulated like nuclear technology?"  # Consistent topic for assignment
        print(f"[Using assignment topic: {topic}]")

    # Log and print start message
    log_node_event("UserInput", "Prompting user for debate topic")
    log_run_start(topic)  # Start the run logging with timestamp
    log_node_event("UserInput", f"Topic received: {topic}")
    print(f"Starting debate between Scientist and Philosopher...\n")

    # Return ALL required initial state updates
    return {
        "topic": topic,
        "current_round": 1,      # Agent A (Scientist) speaks first
        "history": [],           # Empty at start
        "memory_summary": "",    # Will be populated after Round 1
        "winner": None,          # Initialize winner field
        "final_summary": None    # Initialize final_summary field
    }