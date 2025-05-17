import re
import networkx as nx
import matplotlib.pyplot as plt
from typing import TypedDict, Union, Any
from langchain_core.messages import HumanMessage, AIMessage

MessageType = Union[HumanMessage, AIMessage]
from langgraph.graph import StateGraph

END = "END"

class KGState(TypedDict):
    topic: str
    raw_text: str
    entities: list[str]
    relations: list[tuple[str, str, str]]
    resolved_relations: list[tuple[str, str, str]]
    graph: nx.DiGraph  # type: ignore
    validation: dict[str, int | bool]
    messages: list[MessageType]
    current_agent: str

def data_gatherer(state: KGState) -> KGState:
    """
    Searches for information about the given topic and adds it to the state.

    Args:
        state (KGState): The current state of the knowledge graph.

    Returns:
        KGState: The updated state with the collected raw text.
    """
    topic = state["topic"]
    print(f"💻 Data Gatherer: Searching for information about '{topic}'")
    collected_text = f"{topic} is an important concept. It relates to various entities like EntityA, EntityB, and EntityC. EntityA influences EntityB. EntityC is a type of EntityB."
    state["messages"].append(AIMessage(content=f"Collected raw text about {topic}"))
    state["raw_text"] = collected_text
    state["current_agent"] = "entity_extractor"
    return state

def entity_extractor(state: KGState) -> KGState:
    """
    Extracts entities from the raw text in the state and updates the state with the identified entities.

    Args:
        state (KGState): The current state of the knowledge graph.

    Returns:
        KGState: The updated state containing the extracted entities.

    The function identifies entities in the raw text by searching for patterns
    that match the format "EntityX" where X is an uppercase letter. It also 
    includes the topic as an entity. The identified entities are stored in 
    the state, and a message indicating the extracted entities is appended 
    to the state messages. The current agent is updated to "relation_extractor".
    """

    print("🔍 Entity Extractor: Identifying entities in the text")
    text = state["raw_text"]
   
    entities = re.findall(r"Entity[A-Z]", text
    entities = [state["topic"]] + entities
    state["entities"] = list(set(entities))
    state["messages"].append(AIMessage(content=f"Extracted entities: {state['entities']}"))
    print(f"   Found entities: {state['entities']}")
    state["current_agent"] = "relation_extractor"
    return state

def relation_extractor(state: KGState) -> KGState:
    """
    Extracts relationships between entities from the raw text in the state and updates the state with the identified relations.

    Args:
        state (KGState): The current state of the knowledge graph.

    Returns:
        KGState: The updated state containing the extracted relations.

    The function identifies relationships between entities in the raw text by searching for patterns 
    that match specific relationship types such as "relates to", "influences", and "is a type of". 
    It iterates over pairs of entities and applies regular expression patterns to detect relationships. 
    The identified relations are stored in the state, and a message indicating the extracted relations 
    is appended to the state messages. The current agent is updated to "entity_resolver".
    """

    print("🔗 Relation Extractor: Identifying relationships between entities")
    text = state["raw_text"]
    entities = state["entities"]
    relations = []

    relation_patterns = [
        (r"([A-Za-z]+) relates to ([A-Za-z]+)", "relates_to"),
        (r"([A-Za-z]+) influences ([A-Za-z]+)", "influences"),
        (r"([A-Za-z]+) is a type of ([A-Za-z]+)", "is_type_of")
    ]

    for e1 in entities:
        for e2 in entities:
            if e1 != e2:
                for pattern, rel_type in relation_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        subj, obj = match.group(1), match.group(2)
                        if subj == e1 and obj == e2:
                            relations.append((e1, rel_type, e2))

    state["relations"] = relations
    state["messages"].append(AIMessage(content=f"Extracted relations: {relations}"))
    print(f"   Found relations: {relations}")

    state["current_agent"] = "entity_resolver"

    return state

def entity_resolver(state: KGState) -> KGState:
    """
    Resolves duplicate entities in the state by creating a mapping of canonical names and updating the relations with the resolved names.

    Args:
        state (KGState): The current state of the knowledge graph.

    Returns:
        KGState: The updated state containing the resolved relations.

    The function iterates over the entities in the state and creates a mapping of canonical names by lowercasing and replacing spaces with underscores. 
    It then iterates over the relations and resolves the subject and object names using the mapping. 
    The resolved relations are stored in the state, and a message indicating the resolved relations is appended to the state messages. 
    The current agent is updated to "graph_integrator".
    """
    
    print("🔄 Entity Resolver: Resolving duplicate entities")
   
    entity_map = {}
    for entity in state["entities"]:
        canonical_name = entity.lower().replace(" ", "_")
        entity_map[entity] = canonical_name
   
    resolved_relations = []
    for s, p, o in state["relations"]:
        s_resolved = entity_map.get(s, s)
        o_resolved = entity_map.get(o, o)
        resolved_relations.append((s_resolved, p, o_resolved))
   
    state["resolved_relations"] = resolved_relations
    state["messages"].append(AIMessage(content=f"Resolved relations: {resolved_relations}"))
   
    state["current_agent"] = "graph_integrator"
   
    return state

def graph_integrator(state: KGState) -> KGState:
    """
    Builds a knowledge graph from the resolved relations in the state.

    Args:
        state (KGState): The current state of the knowledge graph.

    Returns:
        KGState: The updated state containing the built graph.

    The function builds a directed graph with the resolved relations and stores it in the state. 
    It also appends a message with the number of nodes and edges in the graph to the state messages. 
    The current agent is updated to "graph_validator".
    """
    
    
    
    print("📊 Graph Integrator: Building the knowledge graph")
    G: nx.DiGraph[Any] = nx.DiGraph()
   
    for s, p, o in state["resolved_relations"]:
        if not G.has_node(s):
            G.add_node(s)
        if not G.has_node(o):
            G.add_node(o)
        G.add_edge(s, o, relation=p)
   
    state["graph"] = G
    state["messages"].append(AIMessage(content=f"Built graph with {len(G.nodes)} nodes and {len(G.edges)} edges"))
   
    state["current_agent"] = "graph_validator"
   
    return state

def graph_validator(state: KGState) -> KGState:
    """
    Validates the knowledge graph in the current state.

    Args:
        state (KGState): The current state of the knowledge graph.

    Returns:
        KGState: The updated state containing the validation report.

    The function performs validation checks on the directed graph stored in the state, 
    including counting nodes and edges, checking if the graph is weakly connected, 
    and determining if it has cycles. The results are stored in the state's validation 
    field, and a validation report message is appended to the state messages. 
    The current agent is updated to signal the end of the workflow.
    """

    
    print("✅ Graph Validator: Validating knowledge graph")
    G = state["graph"]
   
    validation_report = {
        "num_nodes": len(G.nodes),
        "num_edges": len(G.edges),
        "is_connected": nx.is_weakly_connected(G) if G.nodes else False,
        "has_cycles": not nx.is_directed_acyclic_graph(G) if G.nodes else False
    }
   
    state["validation"] = validation_report
    state["messages"].append(AIMessage(content=f"Validation report: {validation_report}"))
    print(f"   Validation report: {validation_report}")
   
    state["current_agent"] = END
   
    return state

def router(state: KGState) -> str:
    """
    Returns the current agent in the given state.

    Args:
        state (KGState): The current state of the knowledge graph.

    Returns:
        str: The current agent in the state.
    """
    
    
    return state["current_agent"]


def visualize_graph(graph):
/*************  ✨ Windsurf Command ⭐  *************/
    """
    Visualizes a directed graph using Matplotlib.

    The function uses a spring layout to position the nodes and displays the
    nodes, edges, and edge labels on a plot. Each node is displayed with a sky
    blue color.

    Args:
        graph (nx.DiGraph): The directed graph to visualize.
    """

/***
****  ed22690a-3eb9-48f7-a1bc-5138fac2d78c  *******/
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(graph)
   
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=10)
   
    edge_labels = nx.get_edge_attributes(graph, 'relation')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
   
    plt.title("Knowledge Graph")
    plt.tight_layout()
    plt.show()

def build_kg_graph():
    """
    Builds the knowledge graph workflow.

    This function builds a state graph defining the steps and their dependencies
    involved in the knowledge graph creation process. The steps and their
    dependencies are as follows:

    1. Data Gathering
    2. Entity Extraction
    3. Relation Extraction
    4. Entity Resolution
    5. Graph Integration
    6. Graph Validation

    The workflow starts at the Data Gathering step and ends at the Graph
    Validation step. The workflow automatically transitions between steps
    based on the current state of the knowledge graph.

    Returns:
        The compiled state graph.
    """
    workflow = StateGraph(KGState)
   
    workflow.add_node("data_gatherer", data_gatherer)
    workflow.add_node("entity_extractor", entity_extractor)
    workflow.add_node("relation_extractor", relation_extractor)
    workflow.add_node("entity_resolver", entity_resolver)
    workflow.add_node("graph_integrator", graph_integrator)
    workflow.add_node("graph_validator", graph_validator)
   
    workflow.add_conditional_edges("data_gatherer", router,
                                {"entity_extractor": "entity_extractor"})
    workflow.add_conditional_edges("entity_extractor", router,
                                {"relation_extractor": "relation_extractor"})
    workflow.add_conditional_edges("relation_extractor", router,
                                {"entity_resolver": "entity_resolver"})
    workflow.add_conditional_edges("entity_resolver", router,
                                {"graph_integrator": "graph_integrator"})
    workflow.add_conditional_edges("graph_integrator", router,
                                {"graph_validator": "graph_validator"})
    # Remove the edge to END since it is not a node in the graph
    # This indicates termination of the workflow
    workflow.add_conditional_edges("graph_validator", router,
                                {})
   
    workflow.set_entry_point("data_gatherer")
   
    return workflow.compile()

def run_knowledge_graph_pipeline(topic):
    """
    Runs the knowledge graph pipeline for a given topic.

    Args:
        topic (str): The topic to build the knowledge graph about.

    Returns:
        dict: The final state of the pipeline, which should contain the constructed knowledge graph.
    """
    
    print(f"🚀 Starting knowledge graph pipeline for: {topic}")
   
    initial_state = {
        "topic": topic,
        "raw_text": "",
        "entities": [],
        "relations": [],
        "resolved_relations": [],
        "graph": None,
        "validation": {},
        "messages": [HumanMessage(content=f"Build a knowledge graph about {topic}")],
        "current_agent": "data_gatherer"
    }
   
    kg_app = build_kg_graph()
    final_state = kg_app.invoke(initial_state)
   
    print(f"✨ Knowledge graph construction complete for: {topic}")
   
    return final_state

if __name__ == "__main__":
    topic = "Artificial Intelligence"
    result = run_knowledge_graph_pipeline(topic)
   
    visualize_graph(result["graph"])



