---
id: 0002
title: Update constitution with RAG chatbot requirements
stage: constitution
date: 2025-12-21
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: khubaib
command: /sp.constitution
labels: ["constitution", "rag-chatbot", "architecture", "governance"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - None (constitutional governance change)
---

## Prompt

Here is a **short, clean, Spec-Kit Plus–compliant constitution**, keeping only what's essential for your project:

---

## /sp.constitution

**Project:**
AI-Spec-Driven Book with Embedded RAG Chatbot

---

### Core Principles

* **Spec-first development** using Spec-Kit Plus
* **Technical accuracy** aligned with official documentation
* **Clarity for developers** with CS/AI background
* **Explainability** of systems, not just outcomes

---

### Key Standards

* Book written in **Docusaurus (MDX)** and deployed to **GitHub Pages**
* AI-assisted writing via **Claude Code**, strictly following specs
* All technical claims must be **verifiable**
* Code examples must be **correct, minimal, and explained**

---

### RAG Chatbot Requirements

* Answers questions **only from book content**
* Supports Q&A based on **user-selected text**
* Uses:

  * OpenAI Agents / ChatKit SDKs
  * FastAPI backend
  * Neon Serverless Postgres
  * Qdrant Cloud (Free Tier)
* Must clearly respond when information is **not found**

---

### Constraints

* No hallucinated APIs or features
* No proprietary dependencies beyond free tiers
* Secure handling of environment variables
* Clear separation between content, retrieval, and UI layers

---

### Success Criteria

* Book builds and deploys successfully
* RAG chatbot retrieves and answers accurately
* All specs satisfied without manual correction
* System is reproducible and extensible

---

If you want, I can make an **ultra-minimal version** (one screen) or align this **exactly to Panaversity's official Spec-Kit style**.

## Response snapshot

Constitution updated from version 1.0.0 to 1.1.0 with the following changes:

**Modified Principles:**
- Principle III: "Beginner-Friendly Clarity" reframed to "Clarity for Developers" targeting CS/AI audience with focus on systems understanding and explainability
- Principle VI: "Docusaurus Compatibility" expanded to "Platform Compatibility and Architecture" including comprehensive RAG chatbot technical requirements

**New Sections Added:**
- RAG Chatbot Requirements (major addition):
  - 8 Functional Requirements (FR-RAG-001 through FR-RAG-008)
  - 6 Architectural Requirements (AR-RAG-001 through AR-RAG-006)
  - 6 Success Criteria specific to RAG chatbot (SC-RAG-001 through SC-RAG-006)

**Technical Stack Codified:**
- OpenAI Agents / ChatKit SDKs for agent orchestration
- FastAPI for backend API layer
- Neon Serverless Postgres (free tier) for metadata/state
- Qdrant Cloud (free tier) for vector embeddings
- Three-layer architecture: content, retrieval, UI

**Version Bump Rationale:**
MINOR version (1.1.0) - Added major new section with significant architectural constraints and technical requirements. Materially expands project scope and governance without breaking existing principles.

**Template Consistency:**
All dependent templates (.specify/templates/plan-template.md, spec-template.md, tasks-template.md) reviewed and confirmed compatible with new constitution constraints.

## Outcome

- ✅ Impact: Constitution successfully updated to v1.1.0 with comprehensive RAG chatbot requirements. Clear architectural constraints established for free-tier operation. Target audience reframed from general beginners to CS/AI developers.
- 🧪 Tests: No tests required (governance document update)
- 📁 Files: Modified .specify/memory/constitution.md with complete RAG architecture specification
- 🔁 Next prompts: Create feature specification for RAG chatbot using `/sp.specify`, then generate implementation plan using `/sp.plan`
- 🧠 Reflection: Constitution now provides clear guardrails for both book content creation and RAG chatbot development. Free-tier constraints and architectural separation principles will prevent scope creep and cost overruns.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Manual review PASS - all placeholders filled, version correctly incremented, dates updated, no conflicting principles
- Prompt variant (if applicable): N/A (user-provided constitution content)
- Next experiment (smallest change to try): Test constitution enforcement during `/sp.plan` execution to ensure RAG chatbot architectural constraints are properly validated
