# Auto-KG Knowledge Graph Pipeline

This project implements a knowledge graph construction pipeline using Python. It extracts entities and relations from raw text, resolves entities, integrates them into a directed graph, validates the graph, and visualizes the result.

## Features

- Data gathering for a given topic
- Entity extraction using regex patterns
- Relation extraction with predefined relation types
- Entity resolution to canonical forms
- Knowledge graph construction using NetworkX directed graphs
- Graph validation for connectivity and cycles
- Visualization of the knowledge graph using Matplotlib
- Workflow orchestration using `langgraph` StateGraph

## Requirements

- Python 3.10+
- NetworkX
- Matplotlib
- langchain_core (for message types)
- langgraph (for workflow orchestration)

Install dependencies using uv:

```bash
uv install networkx matplotlib langchain_core langgraph
```

## Usage

Run the main script to build and visualize a knowledge graph for a given topic using uv:

```bash
uv run main.py
```

The default topic is "Artificial Intelligence". You can modify the `topic` variable in `main.py` to analyze other topics.

## Code Structure

- `main.py`: Contains the full pipeline implementation including:
  - TypedDict `KGState` for state management
  - Functions for each pipeline step: data gathering, entity extraction, relation extraction, entity resolution, graph integration, and validation
  - Workflow setup using `StateGraph`
  - Visualization function for the graph

## Notes

- The relation extraction uses simple regex patterns and may need enhancement for complex texts.
- The graph validation checks for connectivity and cycles.
- The visualization uses a spring layout for node positioning.

## License

This project is provided as-is without warranty. Use and modify as needed.
