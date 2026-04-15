import json
import time
from app import chat

print(" Evaluation started...\n")

start = time.time()

test_questions = [

    {"q": "What are the core values mentioned in the code of conduct?", "expected": "C-LIFE"},
    {"q": "What does integrity mean in the company context?", "expected": "transparency"},
    {"q": "What should employees do if they see a violation?", "expected": "report"},
    {"q": "Who can employees contact for ethical concerns?", "expected": "Office of Integrity"},

    {"q": "What is the purpose of the travel policy?", "expected": "guidelines"},
    {"q": "Who approves travel arrangements?", "expected": "supervisor"},
    {"q": "Are luxury accommodations allowed?", "expected": "avoid"},
    {"q": "What must employees submit for reimbursement?", "expected": "receipts"},

    {"q": "What does POSH policy aim to prevent?", "expected": "sexual harassment"},
    {"q": "Within how many months should a complaint be filed?", "expected": "3 months"},
    {"q": "What committee handles harassment complaints?", "expected": "ICC"},
    {"q": "What is the maximum time for inquiry completion?", "expected": "90 days"},


    {"q": "How much reimbursement is provided per head for one event?", "expected": "1500"},
    {"q": "What expenses are covered under this policy?", "expected": "costume"},
    {"q": "What must employees submit for reimbursement?", "expected": "bills"},
    {"q": "Who verifies reimbursement claims?", "expected": "Finance"},

    {"q": "What is considered a bribe?", "expected": "anything of value"},
    {"q": "Is sharing confidential information allowed?", "expected": "not"},
    {"q": "What is insider trading?", "expected": "non-public"},
    {"q": "Can employees engage in conflicts of interest?", "expected": "avoid"},
]

results = []

correct = 0
hallucinated = 0
source_hits = 0


for t in test_questions:
    print(f"🔹 Processing: {t['q']}")

    try:
        res = chat(t["q"])
    except Exception as e:
        print(f"Error on question: {t['q']}")
        print(e)
        continue

    answer = res["answer"].lower()
    expected = t["expected"].lower()

    
    if "don't know" in answer:
        hallucinated += 1
    elif expected in answer:
        correct += 1

  
    if len(res["sources"]) > 0:
        source_hits += 1

    results.append({
        "question": t["q"],
        "answer": res["answer"],
        "sources": res["sources"],
        "expected": t["expected"]
    })


# -------------------------
# MULTI-TURN EVALUATION
# -------------------------
print("\n Running multi-turn test...")

multi_turn_results = []
conversation_id = "test_user"

multi_turn_questions = [
    "What is leave policy?",
    "How many sick leaves?",
    "Can unused leaves be carried forward?"
]

for q in multi_turn_questions:
    try:
        res = chat(q, conversation_id=conversation_id)
    except Exception as e:
        print(f"Error in multi-turn: {q}")
        print(e)
        continue

    multi_turn_results.append({
        "question": q,
        "answer": res["answer"],
        "sources": res["sources"]
    })


total = len(test_questions)

accuracy = correct / total
hallucination_rate = hallucinated / total
avg_sources = sum(len(r["sources"]) for r in results) / total
source_coverage = source_hits / total

summary = {
    "accuracy": accuracy,
    "hallucination_rate": hallucination_rate,
    "avg_sources_per_answer": avg_sources,
    "source_coverage": source_coverage
}

print("\n Evaluation Summary:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Hallucination Rate: {hallucination_rate:.2f}")
print(f"Avg Sources per Answer: {avg_sources:.2f}")
print(f"Source Coverage: {source_coverage:.2f}")

print(f"\n⏱ Total time: {time.time() - start:.2f}s")


with open("results.json", "w") as f:
    json.dump({
        "summary": summary,
        "single_turn_results": results,
        "multi_turn_results": multi_turn_results
    }, f, indent=4)

print("\n Evaluation complete! Results saved to results.json")