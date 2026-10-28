"""Writes the LangGraph structure for the report: graph.mmd (Mermaid text, always) and graph.png (needs internet:
LangGraph renders the PNG through mermaid.ink). Usage: python draw_graph.py"""
from agent import SummaryAgent
from llm import HeuristicBackend
from ontology_store import OntologyStore

ag = SummaryAgent(OntologyStore("infoiag_ontology.owl"), HeuristicBackend(), [], "x")
g = ag.graph.get_graph()
open("graph.mmd", "w").write(g.draw_mermaid())
print("wrote graph.mmd")
try:
    open("graph.png", "wb").write(g.draw_mermaid_png())
    print("wrote graph.png")
except Exception as e:
    print("PNG skipped (", type(e).__name__, ") - paste graph.mmd into https://mermaid.live instead")
