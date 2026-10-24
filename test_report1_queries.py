"""Shows each of the six Report-1 queries firing on a deliberately bad summary claim (needs Java for Pellet).
    python test_report1_queries.py
Expected: Q1, Q2, Q3, Q5, Q6 flag their bad claim; Q4 reports per-topic group counts; clean claims pass."""
from llm import HeuristicBackend
from ontology_store import OntologyStore
from report1_checks import Report1Checker

store = OntologyStore("infoiag_ontology.owl")
chk = Report1Checker(store)
b = HeuristicBackend()
cases = {
    "Q1 underage marriage": ["Jane Smith is 15 years old. Tom Smith married Jane Smith."],
    "Q2 implausible age": ["Old Man is 150 years old."],
    "Q3 diet conflict": ["Anna de Vries ate bread."],
    "Q5 player-team": ["Tom Brady plays for the Seattle Seahawks."],
    "Q6 event cycle": ["Match A happened before Match B.", "Match B happened before Match A."],
    "clean claims": ["Several tweets praise the New England Patriots.", "Tom Brady plays for the New England Patriots."],
}
for name, sents in cases.items():
    claims = [{"sentence": x, "team": None} for x in sents]
    vs, st = chk.check(claims, b.extract_facts(sents))
    print(f"{name:22s} -> {[v.kind for v in vs] or 'no violation'}")
print("Q4 source counts:", chk.q4_source() or "(load tweets first: run main.py)")
