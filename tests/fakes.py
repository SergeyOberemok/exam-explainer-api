"""In-memory stand-ins for the PyMongo async collections the API uses.

They implement only the operators the code sends, with MongoDB's semantics for them
(including array fields and `a.<n>` index paths). `tests/test_game_mongo.py` runs the same
repository calls against a real MongoDB when one is configured, to keep these honest.
"""

import copy

from bson import ObjectId

MISSING = object()


def get_path(doc, path):
    for part in path.split("."):
        if isinstance(doc, dict) and part in doc:
            doc = doc[part]
        elif isinstance(doc, list) and part.isdigit() and int(part) < len(doc):
            doc = doc[int(part)]
        else:
            return MISSING
    return doc


def set_path(doc, path, value):
    *parents, last = path.split(".")
    for part in parents:
        doc = doc.setdefault(part, {})
    doc[last] = value


def _values(value):
    """What an operator compares against: an array field matches by any element."""
    if value is MISSING:
        return []
    return value if isinstance(value, list) else [value]


def matches(doc, query):
    for path, cond in query.items():
        if path == "$text":
            continue
        value = get_path(doc, path)
        if not isinstance(cond, dict):
            if cond is None and value is MISSING:
                continue  # null matches a missing field
            if not (value == cond or cond in _values(value)):
                return False
            continue
        for op, arg in cond.items():
            if op == "$exists":
                if (value is not MISSING) != arg:
                    return False
            elif op == "$gt":
                if value is MISSING or not value > arg:
                    return False
            elif op == "$ne":
                if value == arg or arg in _values(value) or (arg is None and value is MISSING):
                    return False
            elif op == "$in":
                if not any(v in arg for v in _values(value)):
                    return False
            elif op == "$nin":
                if any(v in arg for v in _values(value)):
                    return False
            elif op == "$size":
                if not isinstance(value, list) or len(value) != arg:
                    return False
            else:
                raise NotImplementedError(op)
    return True


class FakeCursor:
    def __init__(self, docs):
        self.docs = docs
        self.limit_value = None

    def sort(self, *args, **kwargs):
        return self

    def skip(self, n):
        self.docs = self.docs[n:]
        return self

    def limit(self, n):
        self.limit_value = n
        return self

    async def to_list(self):
        return self.docs[: self.limit_value] if self.limit_value else list(self.docs)


class FakeCollection:
    """`examination` / `examination_simplified`. `$sample` keeps the stored order (and its
    duplicates, as `$sample` may), so tests are deterministic."""

    def __init__(self, docs):
        self.docs = list(docs)
        self.pipelines = []

    async def aggregate(self, pipeline):
        self.pipelines.append(pipeline)
        docs = list(self.docs)
        for stage in pipeline:
            if "$match" in stage:
                docs = [d for d in docs if matches(d, stage["$match"])]
            elif "$sample" in stage:
                docs = docs[: stage["$sample"]["size"]]
            elif "$limit" in stage:
                docs = docs[: stage["$limit"]]
        return FakeCursor(docs)

    async def count_documents(self, query):
        return len({str(d["_id"]) for d in self.docs if matches(d, query)})

    async def find_one(self, query, projection=None):
        return next((d for d in self.docs if matches(d, query)), None)

    def find(self, query=None, projection=None):
        return FakeCursor([d for d in self.docs if matches(d, query or {})])

    async def distinct(self, field):
        return sorted({d[field] for d in self.docs})


class FakeRounds:
    """`game_rounds`; projections are ignored (the code reads what it projects)."""

    def __init__(self):
        self.docs = {}

    async def insert_one(self, doc):
        doc = {"_id": ObjectId(), **copy.deepcopy(doc)}
        self.docs[doc["_id"]] = doc

        class Result:
            inserted_id = doc["_id"]

        return Result()

    async def find_one(self, query, projection=None):
        doc = self.docs.get(query["_id"])
        return copy.deepcopy(doc) if doc and matches(doc, query) else None

    async def find_one_and_update(self, query, update, projection=None, return_document=False):
        doc = self.docs.get(query["_id"])
        if doc is None or not matches(doc, query):
            return None
        before = copy.deepcopy(doc)
        for path, value in update.get("$set", {}).items():
            set_path(doc, path, value)
        for path, value in update.get("$inc", {}).items():
            set_path(doc, path, get_path(doc, path) + value)
        for path, value in update.get("$push", {}).items():
            items = value["$each"] if isinstance(value, dict) and "$each" in value else [value]
            current = get_path(doc, path)
            set_path(doc, path, (current if current is not MISSING else []) + list(items))
        return copy.deepcopy(doc) if return_document else before
