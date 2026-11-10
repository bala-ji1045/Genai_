"""Knowledge source 1 of 2: the plain Python list/dict from Lab 1. Exact-name match, no retrieval model."""

KNOWLEDGE_BASE = {
    "zylarian communicator": "A compact interstellar comm device with a 3,200 mAh battery, released in a fictional product line for this course.",
    "kelvorian outpost": "Located in the fictional Kelvorian mountain range, established (fictionally) in the year 2089.",
    "mount everlore": "8,850 meters tall, located in the fictional Kelvorian range.",
    "orbital relay station": "Has 340 units of fuel reserve, consuming 28 units per day of normal operation.",
}


def lookup(query: str) -> str:
    q = query.lower().strip().replace("_", " ")
    for name, fact in KNOWLEDGE_BASE.items():
        if name in q or q in name:
            return f"{name}: {fact}"
    return f"No entry for '{query}'. Known entries: {', '.join(KNOWLEDGE_BASE)}."   # a helpful error, not an exception
