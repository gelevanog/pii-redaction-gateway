"""The hand-labeled gold set: inline markup source -> JSONL with character offsets.

Source files (data/gold/source/*.txt) hold documents like

    ### sup-001 | support | en | basic
    Hi, this is [[Anna Petrova|PERSON]], email [[anna.petrova@gmail.com|EMAIL]].

Each document is written and labeled individually against the guidelines in data/gold/README.md; the
compiler only strips the markup and computes offsets. No LLM API or generation pipeline produced the
documents or the labels, and no detector output was copied into them.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from pii_shield.entities import EntityType

_HEADER = re.compile(r"^### (?P<id>[\w-]+)\s*\|\s*(?P<domain>[\w-]+)\s*\|\s*(?P<lang>[\w-]+)\s*(?:\|\s*(?P<tags>.*))?$")
_MARK = re.compile(r"\[\[(?P<value>.+?)\|(?P<type>[A-Z_]+)\]\]", re.DOTALL)


class GoldEntity(BaseModel):
    type: EntityType
    start: int
    end: int
    value: str


class GoldDoc(BaseModel):
    id: str
    domain: str
    lang: str
    tags: list[str] = Field(default_factory=list)
    text: str
    entities: list[GoldEntity] = Field(default_factory=list)


class GoldFormatError(ValueError):
    pass


def parse_markup(doc_id: str, marked: str) -> tuple[str, list[GoldEntity]]:
    text_parts: list[str] = []
    entities: list[GoldEntity] = []
    cursor = 0
    length = 0
    for match in _MARK.finditer(marked):
        before = marked[cursor : match.start()]
        text_parts.append(before)
        length += len(before)
        value, kind = match.group("value"), match.group("type")
        if kind not in EntityType.__members__:
            raise GoldFormatError(f"{doc_id}: unknown entity type {kind!r}")
        if "[[" in value:
            raise GoldFormatError(f"{doc_id}: nested markup in {value!r}")
        entities.append(GoldEntity(type=EntityType(kind), start=length, end=length + len(value), value=value))
        text_parts.append(value)
        length += len(value)
        cursor = match.end()
    text_parts.append(marked[cursor:])
    text = "".join(text_parts)
    if "[[" in text or "]]" in text:
        raise GoldFormatError(f"{doc_id}: unbalanced markup")
    for entity in entities:
        assert text[entity.start : entity.end] == entity.value
    return text, entities


def parse_source(path: Path) -> list[GoldDoc]:
    docs: list[GoldDoc] = []
    header: re.Match[str] | None = None
    lines: list[str] = []

    def flush() -> None:
        if header is None:
            return
        marked = "\n".join(lines).strip("\n")
        text, entities = parse_markup(header["id"], marked)
        tags = [t.strip() for t in (header["tags"] or "").split(",") if t.strip()]
        docs.append(
            GoldDoc(
                id=header["id"], domain=header["domain"], lang=header["lang"], tags=tags, text=text, entities=entities
            )
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            flush()
            header = _HEADER.match(line)
            if header is None:
                raise GoldFormatError(f"{path.name}: bad header {line!r}")
            lines = []
        elif header is None:
            continue  # file preamble (comments)
        else:
            lines.append(line)
    flush()
    return docs


def build_gold(source_dir: Path) -> list[GoldDoc]:
    docs = [doc for path in sorted(source_dir.glob("*.txt")) for doc in parse_source(path)]
    ids = Counter(doc.id for doc in docs)
    duplicates = [doc_id for doc_id, count in ids.items() if count > 1]
    if duplicates:
        raise GoldFormatError(f"duplicate document ids: {', '.join(duplicates)}")
    return docs


def write_gold(docs: list[GoldDoc], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(doc.model_dump_json() + "\n")


def load_gold(path: Path) -> list[GoldDoc]:
    return [
        GoldDoc.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def gold_stats(docs: list[GoldDoc]) -> dict[str, object]:
    by_type = Counter(entity.type.value for doc in docs for entity in doc.entities)
    return {
        "documents": len(docs),
        "entities": sum(by_type.values()),
        "negatives": sum(1 for doc in docs if not doc.entities),
        "by_type": dict(by_type.most_common()),
        "by_domain": dict(Counter(doc.domain for doc in docs).most_common()),
        "by_language": dict(Counter(doc.lang for doc in docs).most_common()),
    }
