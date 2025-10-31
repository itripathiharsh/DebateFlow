from dotenv import load_dotenv
load_dotenv()

import os
from langgraph.graph import StateGraph, END
from state import DebateState
from nodes.user_input_node import user_input_node
from nodes.agent_a_node import agent_a_node
from nodes.agent_b_node import agent_b_node
from nodes.memory_node import memory_node
from nodes.judge_node import judge_node

def route_agent(state: DebateState) -> str:
    """
    Route to next node based on current round:
    - Rounds 1–8: alternate between Agent A and Agent B
    - Round > 8: go to Judge
    """
    if state.current_round > 8:
        return "judge"
    elif state.current_round % 2 == 1:
        return "agent_a"   
    else:
        return "agent_b"  


def build_debate_graph():
    workflow = StateGraph(DebateState)

    def debug_node(node_func, node_name):
        def wrapper(state):
            print(f" Entering {node_name}...")
            result = node_func(state)
            print(f" {node_name} returned: {list(result.keys()) if result else 'NOTHING'}")
            return result
        return wrapper

    workflow.add_node("user_input", debug_node(user_input_node, "user_input"))
    workflow.add_node("agent_a", debug_node(agent_a_node, "agent_a"))
    workflow.add_node("agent_b", debug_node(agent_b_node, "agent_b"))
    workflow.add_node("memory", debug_node(memory_node, "memory"))
    workflow.add_node("judge", debug_node(judge_node, "judge"))

    workflow.set_entry_point("user_input")

    workflow.add_edge("user_input", "agent_a")

    workflow.add_edge("agent_a", "memory")
    workflow.add_edge("agent_b", "memory")

    workflow.add_conditional_edges(
        "memory",
        route_agent,
        {
            "agent_a": "agent_a",
            "agent_b": "agent_b",
            "judge": "judge"
        }
    )

    workflow.add_edge("judge", END)

    return workflow.compile()

def main():
    print(" Starting ATG Multi-Agent Debate System...\n")

    app = build_debate_graph()

    try:
        graph_image = app.get_graph().draw_mermaid_png()
        with open("dag_diagram.png", "wb") as f:
            f.write(graph_image)
        print(" DAG diagram saved as 'dag_diagram.png'")
    except Exception as e:
        print(f" Could not generate DAG diagram (requires internet): {e}")
        try:
            mermaid_code = app.get_graph().draw_mermaid()
            with open("dag_diagram.mmd", "w", encoding="utf-8") as f:
                f.write(mermaid_code)
            print(" Fallback: DAG saved as 'dag_diagram.mmd' (use https://mermaid.live to view)")
        except Exception as fallback_e:
            print(f"  Could not save fallback diagram: {fallback_e}")

    try:
        print("\n Starting debate execution...")
        initial_state = {
            "topic": "",
            "current_round": 0,
            "history": [],
            "memory_summary": "",
            "winner": None,
            "final_summary": None
        }
        final_state = app.invoke(initial_state)
        
        print("\n" + "="*60)
        print("[Judge] Summary of debate:")
        print(final_state.get("final_summary", "No summary available"))
        print(f"\n[Judge] Winner: {final_state.get('winner', 'No winner declared')}")
        print("="*60)
        
        print("\n Debate complete! Full log saved in 'logs/debate_log.txt'")
        
    except KeyboardInterrupt:
        print("\n Debate interrupted by user.")
    except Exception as e:
        print(f"\n Error during debate: {e}")
        raise


if __name__ == "__main__":
    main()