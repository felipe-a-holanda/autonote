# Meetings Folder — Bug Report for Code Agent

This document describes two distinct bugs found in the output produced by the meeting note generation code. Both bugs affect how folders are created inside the `meetings/` directory.

---

## Bug 1: Duplicate Numbered Folders from Re-runs

### What is happening

When the code processes the same meeting recording more than once (e.g. due to a sync triggering re-processing), it creates a new folder with a `(2)`, `(3)`, etc. suffix appended to the original folder name, instead of overwriting or skipping the existing output.

### Examples in the wild

```

/home/felipe/ew/vault/EnrollWise/meetings/
  2026-03-23 📋 Meeting Overview Enrolwise1/          ← original run
  2026-03-23 📋 Meeting Overview Enrolwise1 (2)/      ← second run

  2026-03-23 📋 Meeting Overview ew admin/            ← original
  2026-03-23 📋 Meeting Overview ew admin (2)/        ← second run
  2026-03-23 📋 Meeting Overview ew admin (3)/        ← third run

  2026-03-23 📋 Meeting Overview EW sprint/           ← original
  2026-03-23 📋 Meeting Overview EW sprint (2)/       ← second run

  2026-03-23 📋 Overview ew onboarding/               ← original
  2026-03-23 📋 Overview ew onboarding (2)/           ← second run
```

### Key observations

- The files **inside** the `(2)` / `(3)` folders use the **original name** (without the number suffix). For example, `2026-03-23 📋 Meeting Overview Enrolwise1 (2)/` contains a file named `2026-03-23 📋 Meeting Overview Enrolwise1.md`. This confirms the suffix is being added at the **folder creation** level, not at the file level.
- The content of the duplicate folders is **not identical** — the AI-generated summaries differ slightly between runs (different tags, slightly different wording, different participant lists). This means the code is re-running the full LLM summarisation pipeline on each run, rather than detecting that a note for this meeting already exists and skipping it.
- The `(2)` numbering pattern is characteristic of OS-level folder creation (e.g. macOS or a sync tool like iCloud/Dropbox auto-renaming a conflicting folder). The bug may be that the code does not check for an existing folder before attempting to create one, causing the filesystem or sync layer to auto-increment the name.

### Expected behaviour

Before creating a meeting folder, the code should check whether a folder for that meeting (matched by date + meeting title) already exists. If it does, it should either:
- Skip processing entirely, or
- Overwrite/update the existing folder in place.

---

## Bug 2: Single Meeting Split Into Multiple Folders

### What is happening

Instead of producing **one folder per meeting** containing all sections of the meeting note, the code creates **one folder per section** of the meeting note. Each section (e.g. Overview, Participants, Meeting Summary) becomes its own top-level folder inside `meetings/`, all with the same date and meeting title in the name.

### Examples in the wild

**ew setup meeting (3 folders instead of 1):**
```
meetings/
  2026-03-23 1. 📋 Meeting Summary ew setup/
      2026-03-23 1. 📋 Meeting Summary ew setup.md
      2026-03-23 1. 📋 Meeting Summary ew setup - transcript.md

  2026-03-23 Overview ew setup/
      2026-03-23 Overview ew setup.md
      2026-03-23 Overview ew setup - transcript.md

  2026-03-23 Participants ew setup/
      2026-03-23 Participants ew setup.md
      2026-03-23 Participants ew setup - transcript.md
```

**ew retro meeting (3 folders instead of 1):**
```
meetings/
  2026-03-27 1. Meeting Summary ew retro/
  2026-03-27 📋 Meeting Overview ew retro/
  2026-03-27 📋 Overview ew retro/
```

**beon tech meeting (2 folders instead of 1):**
```
meetings/
  2026-03-23 📋 Meeting Overview beon tech/
  2026-03-23 📋 Overview beon tech/
```

**EW sprint meeting (2 folders instead of 1):**
```
meetings/
  2026-03-23 📋 Meeting Overview EW sprint/
  2026-03-23 Participants EW sprint/
```

**ew system overview and ew tugboat (2 folders each instead of 1):**
```
meetings/
  2026-03-24 Overview ew system overview/
  2026-03-24 Participants ew system overview/

  2026-03-24 Overview ew tugboat/
  2026-03-24 Participants ew tugboat/
```

### Key observations

- The folder name is being constructed by concatenating the **section heading** (e.g. `Overview`, `Participants`, `1. 📋 Meeting Summary`) with the meeting title, rather than using just the meeting title.
- All fragmented folders for the same meeting share the **same `date` and `time` in their frontmatter**, confirming they originate from the same source recording.
- Each fragmented folder contains both a main `.md` file and a `-transcript.md` file, which means the **transcript is being duplicated** across every section folder as well.
- Some section names appear inconsistently across meetings (e.g. `📋 Meeting Overview` vs `📋 Overview` vs `Overview`), suggesting the section headings in the generated markdown are not stable and the folder-naming logic is sensitive to this variation.

### Expected behaviour

The code should produce **one folder per meeting**, named using only the date and meeting title:

```
meetings/
  2026-03-23 ew setup/
      2026-03-23 ew setup.md          ← full meeting note with all sections
      2026-03-23 ew setup - transcript.md
```

All sections (Summary, Overview, Participants, Action Items, Warnings, etc.) should live as headings **within a single `.md` file**, not as separate folders. The transcript file should appear once per meeting.

---

## Summary Table

| Bug | Root Cause (likely) | Effect |
|-----|---------------------|--------|
| Duplicate `(n)` folders | No idempotency check before folder creation; re-runs create new folders instead of skipping or overwriting | Multiple near-identical versions of the same meeting note |
| Section-per-folder splitting | Folder name is derived from the section heading inside the generated markdown, not from the meeting title alone | Each meeting produces 2–3 folders instead of 1; transcript duplicated per section |
