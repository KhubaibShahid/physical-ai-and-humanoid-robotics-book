<!--
Sync Impact Report:
- Version change: 1.1.0 → 2.0.0
- Modified principles:
  - All principles streamlined and simplified while preserving core intent
  - Principle VI: Expanded title from "Platform Compatibility and Architecture" to include explicit RAG requirements
- Added sections: None (restructured existing content)
- Removed sections:
  - Quality Standards (content merged into Core Principles)
  - Content Constraints (content merged into Core Principles and Success Criteria)
  - RAG Chatbot Requirements (content merged into Principle VI and Success Criteria)
- Templates requiring updates:
  ✅ .specify/templates/plan-template.md (reviewed - constitution check compatible with streamlined principles)
  ✅ .specify/templates/spec-template.md (reviewed - requirements alignment compatible)
  ✅ .specify/templates/tasks-template.md (reviewed - task categorization compatible)
- Follow-up TODOs: None
- Rationale for MAJOR version bump:
  * Backward-incompatible governance restructuring - removed major sections (Quality Standards, Content Constraints, detailed RAG Requirements)
  * Significant simplification changes enforcement expectations and compliance checking
  * Templates and workflows expecting detailed sections may need adjustment
  * Philosophy shift from comprehensive detailed guidance to essential principles only
-->

# Physical AI and Humanoid Robotics Book Constitution

**Project:**
AI-Spec-Driven Book with Embedded RAG Chatbot

---

## Core Principles

### I. Spec-First Development

**Spec-first development** using Spec-Kit Plus. Every chapter, section, code example, and feature MUST originate from an approved specification before any implementation begins.

**Rationale**: Ensures consistency, reproducibility, and alignment with project goals. Prevents scope creep and maintains clear traceability from requirements to deliverables.

---

### II. Technical Accuracy

All technical information, APIs, code examples, architectural decisions, and references MUST be **verifiable and correct**. No hallucinated facts, invented APIs, or unverified claims permitted.

**Non-negotiable rules**:
- Code examples must be syntactically correct and follow best practices
- API references must link to official documentation
- Technical claims require citation or verification from official sources
- All code must be **correct, minimal, and explained**

**Rationale**: Maintains reader trust and educational value. Technical accuracy is the foundation of credibility.

---

### III. Clarity for Developers

Content MUST be accessible and clear for developers with **CS/AI background**. Technical terminology should be precise. Complex concepts require clear **explanation of systems, not just outcomes**.

**Non-negotiable rules**:
- Use precise technical terminology appropriate for CS/AI audience
- Explain system architectures and design decisions
- Provide context for "why" decisions are made, not just "what" to do
- Focus on **explainability** of systems and tradeoffs

**Rationale**: The book serves developers and CS/AI students who need to understand underlying systems and make informed architectural decisions.

---

### IV. AI-Native Authorship

This project is built using **AI-assisted writing via Claude Code**, strictly following specs. All content generation follows reproducible, documented processes with human oversight.

**Non-negotiable rules**:
- All prompts and AI interactions documented in Prompt History Records (PHRs)
- Specifications drive content; AI generates within defined constraints
- Human review required before publishing
- Architectural decisions documented in ADRs (Architecture Decision Records)

**Rationale**: Transparency in authorship, reproducibility, and continuous improvement of AI-assisted workflows.

---

### V. Zero Plagiarism

All content MUST be original. External sources require proper attribution. Licensing compliance is mandatory.

**Rationale**: Ethical authorship, legal compliance, and value creation through original synthesis.

---

### VI. Platform Compatibility and Architecture

Book written in **Docusaurus (MDX)** and deployed to **GitHub Pages**. The RAG chatbot system MUST use specified free-tier services and maintain clear architectural separation.

**Non-negotiable rules for Book**:
- Valid Markdown/MDX syntax
- Frontmatter follows Docusaurus conventions
- Build succeeds without errors

**Non-negotiable rules for RAG Chatbot**:
- Answers questions **only from book content**
- Supports Q&A based on **user-selected text**
- Uses:
  - OpenAI Agents / ChatKit SDKs
  - FastAPI backend
  - Neon Serverless Postgres (free tier)
  - Qdrant Cloud (free tier)
- Must clearly respond when information is **not found**
- Secure handling of environment variables
- Clear separation between content, retrieval, and UI layers

**Rationale**: Ensures publishability, maintainability, and cost-effective operation within free-tier constraints.

---

## Constraints

* **No hallucinated APIs or features** - all technical claims must be verifiable
* **No proprietary dependencies** beyond free tiers specified above
* **Secure handling of environment variables** - no secrets in code
* **Clear separation** between content, retrieval, and UI layers

---

## Success Criteria

### Book Quality

* Book builds and deploys successfully
* All technical claims are verifiable
* Code examples are correct, minimal, and explained
* All specs satisfied without manual correction

### RAG Chatbot Quality

* RAG chatbot retrieves and answers accurately from book content only
* Clearly responds when information is not found
* Operates within free-tier limits
* System is reproducible and extensible

---

## Governance

### Amendment Process

This constitution supersedes all other development practices. Amendments require:

1. Documented justification for the change
2. Impact analysis on existing content and workflows
3. Update to version number following semantic versioning
4. Approval from project maintainer

### Compliance

- All pull requests MUST verify compliance with these principles
- Complexity or deviations MUST be justified in writing and documented
- Use `CLAUDE.md` for runtime development guidance
- Constitution violations require explicit sign-off and documentation

### Versioning

Version numbers follow MAJOR.MINOR.PATCH format:

- **MAJOR**: Backward-incompatible principle changes, removals, or redefinitions
- **MINOR**: New principles added or materially expanded guidance
- **PATCH**: Clarifications, wording improvements, typo fixes

**Version**: 2.0.0 | **Ratified**: 2025-12-20 | **Last Amended**: 2025-12-21
