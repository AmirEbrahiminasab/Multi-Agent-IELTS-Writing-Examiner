from langgraph.graph import StateGraph, START, END


def workflow_fn(State, task_response, lexical_resource, grammatical_range_and_accuracy, coherence_and_cohesion, aggregator):
    workflow = StateGraph(State)

    workflow.add_node("Task Response", task_response)
    workflow.add_node("Lexical", lexical_resource)
    workflow.add_node("Grammar", grammatical_range_and_accuracy)
    workflow.add_node("CC", coherence_and_cohesion)
    workflow.add_node("Aggregator", aggregator)

    workflow.add_edge(START, "Task Response")
    workflow.add_edge(START, "Lexical")
    workflow.add_edge(START, "Grammar")
    workflow.add_edge(START, "CC")
    workflow.add_edge("Task Response", "Aggregator")
    workflow.add_edge("CC", "Aggregator")
    workflow.add_edge("Grammar", "Aggregator")
    workflow.add_edge("Lexical", "Aggregator")
    workflow.add_edge("Aggregator", END)

    workflow = workflow.compile()

    with open("Workflow.md", "w") as f:
        mermaid_code = workflow.get_graph().draw_mermaid()
        f.write(f"```mermaid\n{mermaid_code}\n```")

    return workflow
