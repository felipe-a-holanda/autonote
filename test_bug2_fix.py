#!/usr/bin/env python3
"""Test script to verify Bug 2 fix - single meeting should create single folder"""

import re
from pathlib import Path

def parse_existing_frontmatter(content: str):
    """Simplified version of frontmatter parser"""
    if not content.startswith('---'):
        return {}, content
    
    parts = content.split('---', 2)
    if len(parts) < 3:
        return {}, content
    
    fm_text = parts[1]
    body = parts[2]
    
    # Simple YAML parsing for title field
    fm = {}
    for line in fm_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            fm[key.strip()] = value.strip().strip("'\"")
    
    return fm, body

def _resolve_vault_title(summary_file: str | None, time: str) -> str:
    """Fixed version of _resolve_vault_title with expanded BOILERPLATE"""
    BOILERPLATE = {
        "meeting summary", "action items", "summary",
        "overview", "meeting overview", "participants", 
        "key discussion points", "discussion points",
        "decisions made", "decisions", "key warnings", 
        "things to remember", "warnings", "next steps", 
        "follow-up", "notes", "main tasks", "tasks",
        "content analysis", "formatting guidelines",
        "additional notes", "key sections", "conclusion",
        "key takeaways", "takeaways", "per-person updates",
        "architecture clarification", "blockers", "agenda",
        "background", "context", "objectives", "goals"
    }
    if summary_file:
        path = Path(summary_file)
        if path.exists():
            content = path.read_text(encoding="utf-8")
            fm, body = parse_existing_frontmatter(content)
            user_tag = (fm.get("title") or "").strip()
            inferred = ""
            for line in body.splitlines():
                m = re.match(r"^#{1,3} (.+)", line)
                if m:
                    heading = m.group(1).strip()
                    # Normalize heading: remove emojis, numbered prefixes, and extra whitespace
                    normalized = re.sub(r'^[\d\.\s]*', '', heading)  # Remove leading numbers and dots
                    # Remove all emoji characters including compound emojis and variation selectors
                    normalized = re.sub(r'[\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F]+', '', normalized)
                    normalized = normalized.strip()
                    if normalized.lower() not in BOILERPLATE:
                        inferred = heading
                        break
            if user_tag:
                return user_tag
            if inferred:
                return inferred
    return time

# Test cases
test_files = [
    ("/home/felipe/Dropbox/Software/autonote/recordings/20260323/meeting_20260323_121157/meeting_20260323_121157_summary.md", "12:11", "ew-setup"),
    ("/home/felipe/Dropbox/Software/autonote/recordings/20260327/meeting_20260327_113024/meeting_20260327_113024_summary.md", "11:30", "ew-retro"),
]

print("Testing Bug 2 fix - _resolve_vault_title function")
print("=" * 70)

for summary_file, time_part, expected_title in test_files:
    if Path(summary_file).exists():
        result = _resolve_vault_title(summary_file, time_part)
        status = "✓ PASS" if result == expected_title else "✗ FAIL"
        print(f"\n{status}")
        print(f"  File: {Path(summary_file).name}")
        print(f"  Expected: {expected_title}")
        print(f"  Got:      {result}")
    else:
        print(f"\n⚠ SKIP - File not found: {summary_file}")

print("\n" + "=" * 70)
print("Test complete!")
