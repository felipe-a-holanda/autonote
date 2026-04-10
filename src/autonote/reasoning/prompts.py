"""Prompt templates for LLM reasoning tasks.

Ported directly from meeting-copilot/backend/reasoning/prompts.py.
"""

PROGRESSIVE_SUMMARY = """\
You are a meeting assistant. You maintain a running summary of a meeting.

Current summary (what has been discussed so far):
{current_summary}

New transcript segments since last update:
{new_segments}

Update the summary to incorporate the new information. Rules:
- Keep the summary concise (max 300 words)
- Preserve key decisions and important points from the existing summary
- Add new topics, decisions, and important statements
- Use bullet points grouped by topic
- Note who said what when relevant
- Write in the same language as the transcript
- If the meeting language is Portuguese, write the summary in Portuguese

Updated summary:"""

ACTION_ITEMS = """\
You are a meeting assistant that extracts action items and decisions.

Meeting context:
{full_context}

Recent transcript:
{recent_transcript}

Existing action items:
{existing_items}

Extract any NEW action items or decisions from the recent transcript. For each:
- description: What needs to be done or what was decided
- assignee: Who is responsible (use speaker label if clear, "TBD" if not)
- type: "action" or "decision"

Also check if any existing items should be marked as "completed" or "updated".

Respond in JSON format:
{{
  "new_items": [
    {{"description": "...", "assignee": "...", "type": "action"}}
  ],
  "updated_items": [
    {{"id": "...", "status": "completed", "note": "..."}}
  ]
}}"""

CONTRADICTION_DETECTION = """\
You are a meeting analyst detecting contradictions and inconsistencies.

Meeting summary so far:
{current_summary}

Recent transcript (last 2 minutes):
{recent_transcript}

Identify any contradictions where a speaker says something that conflicts with:
1. Something they said earlier
2. Something another speaker said
3. A decision that was already made

Only flag CLEAR contradictions, not minor clarifications or evolving discussions.

If contradictions found, respond in JSON:
{{
  "contradictions": [
    {{
      "description": "Brief description of the contradiction",
      "statement_a": "Earlier statement",
      "statement_b": "Contradicting statement",
      "severity": "low|medium|high"
    }}
  ]
}}

If no contradictions, respond: {{"contradictions": []}}"""

REPLY_SUGGESTION = """\
You are a meeting copilot helping the user participate more effectively.

Meeting context:
{full_context}

The user wants help responding to the current discussion.
{context_hint}

Generate 2-3 short reply suggestions the user could say. Consider:
- What was just discussed
- Any open questions that need answering
- Opportunities to clarify or add value
- The overall tone of the meeting

Respond in JSON:
{{
  "suggestions": [
    "Suggestion 1 — direct and concise",
    "Suggestion 2 — alternative angle",
    "Suggestion 3 — diplomatic/cautious option"
  ],
  "context": "Brief note on what triggered these suggestions"
}}"""

CUSTOM_PROMPT_TEMPLATE = """\
You are a meeting copilot with full context of the ongoing meeting.

Meeting context:
{full_context}

The user asks: {user_prompt}

Respond helpfully based on the meeting context. Be concise and actionable."""

COACH_PROMPT = """\
You are a real-time meeting coach. You have been briefed with a mission.

== YOUR MISSION ==
{mission_brief}

== CURRENT MEETING STATE ==
{full_context}

== MOST RECENT EXCHANGE ==
{recent_transcript}

Based on the discussion, decide if this is the right moment to act.

Respond in JSON:
{{
  "should_speak": true,
  "suggestion": "What to say or do right now (or 'Stay quiet and listen')",
  "argument_used": "Which argument from your brief is relevant, or null",
  "reasoning": "Why now (or why not)",
  "confidence": "high|medium|low"
}}"""

ADHOC_REPLY_PROMPT = """\
You are an expert meeting strategist providing a single, powerful reply suggestion.

{mission_section}

== FULL MEETING TRANSCRIPT ==
{full_context}

The user needs to respond RIGHT NOW. Based on everything discussed so far, \
craft the ideal reply.

Requirements:
- One concrete, ready-to-speak reply (not multiple options)
- Tailored to the full conversation arc, not just the last exchange
- If a mission brief is provided, align with the stated goal and available arguments
- Be specific and actionable — the user should be able to say this verbatim
- Write in the same language as the transcript

Respond in JSON:
{{
  "reply": "The exact words to say",
  "reasoning": "Why this reply is strategically effective right now"
}}"""

# Map task names to prompt templates for easy lookup by the dispatcher
PROMPT_MAP: dict[str, str] = {
    "summary": PROGRESSIVE_SUMMARY,
    "action_items": ACTION_ITEMS,
    "contradictions": CONTRADICTION_DETECTION,
    "reply": REPLY_SUGGESTION,
    "custom": CUSTOM_PROMPT_TEMPLATE,
    "coach": COACH_PROMPT,
    "adhoc_reply": ADHOC_REPLY_PROMPT,
}
