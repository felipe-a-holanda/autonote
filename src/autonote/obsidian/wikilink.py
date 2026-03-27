"""
Injects Obsidian [[wikilinks]] into a markdown file.
Wraps known people, products, and Jira ticket IDs in [[double brackets]].
Skips YAML frontmatter, code blocks, and already-linked text.
"""
import re
from pathlib import Path
from autonote.logger import log_info, log_error

def load_entities(entities_file: Path) -> dict[str, list[str]]:
    entities: dict[str, list[str]] = {"people": [], "products": []}
    if not entities_file.exists(): return entities
    current_section = None
    for line in entities_file.read_text(encoding="utf-8").splitlines():
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith("#"): continue
        if line_stripped.endswith(":") and not line_stripped.startswith("-"):
            section = line_stripped[:-1].lower()
            current_section = section if section in entities else None
        elif line_stripped.startswith("- ") and current_section:
            val = line_stripped[2:].strip().strip("\"'")
            if val: entities[current_section].append(val)
    return entities

def split_sections(content: str) -> tuple[str, list[tuple[str, bool]]]:
    frontmatter = ""
    body = content
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            frontmatter = content[: end + 4]
            body = content[end + 4:]
    pattern = re.compile(r"(```[\s\S]*?```|`[^`]+`|\[\[.*?\]\])", re.DOTALL)
    segments: list[tuple[str, bool]] = []
    last = 0
    for m in pattern.finditer(body):
        if m.start() > last: segments.append((body[last : m.start()], False))
        segments.append((m.group(), True))
        last = m.end()
    if last < len(body): segments.append((body[last:], False))
    return frontmatter, segments

def make_pattern(entity: str) -> re.Pattern:
    escaped = re.escape(entity)
    return re.compile(r"(?<!\[)\b" + escaped + r"\b(?!\])", re.IGNORECASE)

def inject_wikilinks(content: str, entities: dict[str, list[str]]) -> str:
    all_names: list[str] = []
    for group in ("people", "products"): all_names.extend(entities.get(group, []))
    all_names.sort(key=lambda x: -len(x))
    jira_pattern = re.compile(r"(?<!\[)\b([A-Z]{2,10}-\d+)\b(?!\])")
    frontmatter, segments = split_sections(content)
    new_segments = []
    for text, protected in segments:
        if protected:
            new_segments.append(text)
            continue
        text = jira_pattern.sub(r"[[\1]]", text)
        for name in all_names:
            pattern = make_pattern(name)
            text = pattern.sub(f"[[{name}]]", text)
        new_segments.append(text)
    return frontmatter + "".join(new_segments)

def run_wikilinks(file: str, entities: str):
    file_path = Path(file)
    if not file_path.exists(): raise FileNotFoundError(f"Markdown file not found: {file}")
    entities_path = Path(entities)
    ents = load_entities(entities_path)
    if not any(ents.values()): log_error(f"Warning: no entities loaded from {entities}")
    content = file_path.read_text(encoding="utf-8")
    updated = inject_wikilinks(content, ents)
    if updated != content:
        file_path.write_text(updated, encoding="utf-8")
        log_info(f"Wikilinks injected: {file_path}")
    else:
        log_info(f"No wikilink changes needed: {file_path}")
    return str(file_path)
