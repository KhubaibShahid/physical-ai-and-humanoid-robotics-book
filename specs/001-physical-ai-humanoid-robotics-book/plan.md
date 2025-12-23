# Implementation Plan: Physical AI & Humanoid Robotics Book

**Branch**: `001-physical-ai-humanoid-robotics-book` | **Date**: 2025-12-22 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-physical-ai-humanoid-robotics-book/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Create a comprehensive educational resource on Physical AI and Humanoid Robotics delivered as a Docusaurus-based static site. The book will consist of four self-contained modules (ROS 2, Digital Twin, AI-Robot Brain, VLA) targeting CS-background students. Each module follows a consistent internal structure: concept overview, system architecture, tooling ecosystem, integration points, and limitations. The implementation uses research-concurrent writing, verifying all technical content against official framework documentation, and producing MDX-formatted content optimized for RAG ingestion and web deployment.

## Technical Context

**Language/Version**: Docusaurus v3.x (Node.js 18+), MDX v2.x
**Primary Dependencies**:
- Docusaurus (static site generator)
- React (for MDX components)
- Prism/Highlight.js (code syntax highlighting)
- Context7 MCP (for Docusaurus documentation research)

**Storage**:
- Static MDX files in version control (Git)
- Generated static HTML/CSS/JS for deployment
- Future: Neon Serverless Postgres (free tier) for RAG chatbot embeddings
- Future: Qdrant Cloud (free tier) for vector search

**Testing**:
- Docusaurus build validation (no errors)
- MDX syntax validation
- Link checking (internal and external)
- Content quality validation checklists per module

**Target Platform**: GitHub Pages (static hosting)

**Project Type**: Web documentation site (static site generator)

**Performance Goals**:
- Page load time <2 seconds on 3G
- Build time <5 minutes for full site
- Mobile-responsive rendering

**Constraints**:
- All code examples must be syntactically correct and verified
- No fabricated APIs or framework features
- All technical claims must cite official documentation
- Content must render correctly in Docusaurus MDX
- Free-tier services only (GitHub Pages, future Neon/Qdrant)

**Scale/Scope**:
- 4 major modules
- Estimated 15-20 chapters total across all modules
- ~50-100 pages of content
- 30-50 code examples
- 10-20 diagrams/architecture illustrations

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Spec-First Development ✅ PASS
- ✅ Feature originates from approved specification (spec.md exists and validated)
- ✅ Clear traceability from requirements to deliverables via Spec-Kit Plus workflow
- ✅ PHR documentation will track all AI interactions

### Principle II: Technical Accuracy ✅ PASS
- ✅ Commitment to verify all technical content against official documentation
- ✅ Research-concurrent writing approach prevents hallucination
- ✅ Code examples will be tested and verified
- ✅ API references will link to official ROS 2, Gazebo, Unity, Isaac, VLA docs

### Principle III: Clarity for Developers ✅ PASS
- ✅ Target audience: CS/AI background students (appropriate technical level)
- ✅ Consistent module structure includes system architecture explanations
- ✅ Each module documents "why" decisions are made (rationale, tradeoffs)
- ✅ Focus on explainability of systems (not just "follow these steps")

### Principle IV: AI-Native Authorship ✅ PASS
- ✅ Using Claude Code with Spec-Kit Plus for content generation
- ✅ All prompts tracked via PHRs
- ✅ Specifications drive content generation within defined constraints
- ✅ Human review checkpoint before publishing
- ✅ ADRs will document significant architectural decisions

### Principle V: Zero Plagiarism ✅ PASS
- ✅ Content will be original synthesis and explanation
- ✅ External sources will be properly attributed
- ✅ Framework documentation will be cited, not copied
- ✅ Code examples will be original or properly licensed

### Principle VI: Platform Compatibility and Architecture ✅ PASS

**Book Requirements**:
- ✅ Docusaurus (MDX) chosen as platform
- ✅ Frontmatter will follow Docusaurus conventions
- ✅ Build validation enforced via testing strategy
- ✅ GitHub Pages deployment planned

**RAG Chatbot Requirements** (Future Phase):
- ✅ Content structure optimized for RAG ingestion (modular, well-structured)
- ✅ Free-tier services specified: Neon Postgres, Qdrant Cloud
- ✅ OpenAI Agents / ChatKit SDKs, FastAPI backend planned
- ✅ Clear architectural separation (content, retrieval, UI layers)

**GATE RESULT**: ✅ ALL CHECKS PASS - Proceed to Phase 0

## Project Structure

### Documentation (this feature)

```text
specs/001-physical-ai-humanoid-robotics-book/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0: Module architecture, Docusaurus best practices
├── data-model.md        # Phase 1: Content entities (Module, Chapter, Section, CodeExample)
├── quickstart.md        # Phase 1: Writer's guide for creating new content
├── contracts/           # Phase 1: Module structure schemas, sidebar config
│   ├── module-schema.yaml        # JSON Schema for module frontmatter
│   ├── sidebar-structure.js      # Docusaurus sidebar configuration
│   └── content-checklist.yaml    # Per-module validation checklist
└── tasks.md             # Phase 2: /sp.tasks command (NOT created by /sp.plan)
```

### Source Code (repository root)

```text
# Docusaurus project structure
docs/                            # Main content directory
├── intro.md                     # Landing page / book overview
├── module-01-ros2/              # Module 1: ROS 2 (Robotic Nervous System)
│   ├── _category_.json          # Module metadata
│   ├── 01-overview.md           # Concept overview
│   ├── 02-architecture.md       # System architecture & data flow
│   ├── 03-tooling.md            # ROS 2 ecosystem (nodes, topics, rclpy)
│   ├── 04-integration.md        # Integration foundations
│   └── 05-summary.md            # Summary and limitations
├── module-02-digital-twin/      # Module 2: Digital Twin (Gazebo & Unity)
│   ├── _category_.json
│   ├── 01-overview.md
│   ├── 02-architecture.md
│   ├── 03-tooling.md            # Gazebo, Unity, physics engines
│   ├── 04-integration.md        # Integration with ROS 2
│   └── 05-summary.md
├── module-03-ai-brain/          # Module 3: AI-Robot Brain (NVIDIA Isaac)
│   ├── _category_.json
│   ├── 01-overview.md
│   ├── 02-architecture.md
│   ├── 03-tooling.md            # Isaac SDK, perception, SLAM, navigation
│   ├── 04-integration.md        # Integration with ROS 2 + simulation
│   └── 05-summary.md
├── module-04-vla/               # Module 4: Vision-Language-Action
│   ├── _category_.json
│   ├── 01-overview.md
│   ├── 02-architecture.md
│   ├── 03-tooling.md            # LLMs, Whisper, action execution
│   ├── 04-integration.md        # Full stack integration
│   ├── 05-summary.md
│   └── 06-capstone.md           # Capstone project walkthrough
└── appendix/                    # Supporting content
    ├── glossary.md
    ├── resources.md
    └── troubleshooting.md

src/                             # Docusaurus React components
├── components/                  # Custom MDX components
│   ├── RobotDiagram.js         # Architecture diagrams
│   ├── CodeExample.js          # Enhanced code blocks
│   └── IntegrationFlow.js      # Integration visualization
├── css/                        # Custom styling
│   └── custom.css
└── pages/                      # Non-docs pages (optional)

static/                         # Static assets
├── img/                        # Images, diagrams
│   ├── module-01/
│   ├── module-02/
│   ├── module-03/
│   └── module-04/
└── code-examples/              # Downloadable code samples
    ├── ros2-examples/
    ├── simulation-examples/
    ├── isaac-examples/
    └── vla-examples/

docusaurus.config.js            # Docusaurus configuration
sidebars.js                     # Sidebar navigation structure
package.json                    # Node.js dependencies

tests/                          # Content validation tests
├── build-validation.test.js    # Docusaurus build success
├── link-checker.test.js        # Internal/external link validation
└── syntax-validation.test.js   # MDX syntax validation

.github/
└── workflows/
    ├── build-deploy.yml        # GitHub Pages deployment
    └── content-validation.yml  # PR validation checks
```

**Structure Decision**:
Selected Docusaurus documentation site structure (Option 2 variant). This choice is driven by:
1. **Clear module separation**: Each of the 4 modules gets a dedicated directory with consistent internal structure (5-6 chapters)
2. **Docusaurus conventions**: Uses `_category_.json` for module metadata, follows standard `docs/` directory pattern
3. **Scalability**: Easy to add new modules or chapters without restructuring
4. **RAG-friendly**: Modular structure with clear hierarchy enables efficient chunking for vector embeddings
5. **Component reusability**: Custom React components in `src/components/` for consistent presentation of diagrams, code examples, and integration flows

## Complexity Tracking

> **No constitutional violations detected - table not needed**

All requirements align with constitutional principles:
- Single project type (documentation site)
- Standard Docusaurus patterns (no custom abstractions needed)
- Direct content authoring workflow (no complex build pipelines)
- Free-tier services only
- Clear separation of concerns (content, presentation, future RAG layer)

## Architecture Overview

### Module Progression Flow

```
Module 1 (ROS 2)           →  Foundation Layer
    ↓ provides
Module 2 (Digital Twin)    →  Testing & Simulation Layer
    ↓ builds on
Module 3 (AI Brain)        →  Intelligence Layer
    ↓ integrates
Module 4 (VLA)             →  Autonomous Behavior Layer (Capstone)
```

**Integration Architecture**:
- **Module 1 → 2**: ROS 2 concepts (nodes, topics, URDF) enable simulation setup in Gazebo/Unity
- **Module 2 → 3**: Simulated environments provide test bed for Isaac perception/navigation
- **Module 3 → 4**: Isaac's perception + navigation capabilities consumed by VLA planning layer
- **Module 4**: Synthesizes all previous modules into voice-to-action pipeline

### Content Entity Model (Preview)

**Module Entity**:
- Title, learning objectives, outcomes
- 5-6 chapters following standard structure
- Integration dependencies on prior modules
- Technical frameworks covered

**Chapter Entity**:
- Title, purpose, prerequisites
- Markdown content (MDX)
- Code examples
- Diagrams/illustrations
- External references (cited)

**Code Example Entity**:
- Language, framework
- Full working code
- Explanation
- Link to official docs

### Validation Strategy

**Per-Module Checklist** (contracts/content-checklist.yaml):
1. Self-contained and internally consistent
2. Clear connection to next module
3. No undocumented/speculative features
4. Terminology consistency with glossary
5. Architecture alignment with other modules
6. Docusaurus build success
7. All code examples tested
8. All API references linked to official docs

**Capstone Validation**:
Voice → Plan → Navigate → Perceive → Act flow must be traceable across all 4 modules

## Writing & Research Workflow

### Research-Concurrent Approach

**Phase 0 (Research & Architecture)**:
1. Use Context7 MCP to fetch Docusaurus best practices
2. Research official documentation for:
   - ROS 2 core concepts, APIs
   - Gazebo/Unity simulation patterns
   - NVIDIA Isaac SDK architecture
   - VLA implementation examples (research papers, open source)
3. Document decisions in `research.md`:
   - Why Docusaurus? (alternatives: Sphinx, GitBook, MkDocs)
   - Why this module order? (pedagogical rationale)
   - Why these specific frameworks? (industry standards, educational value)
4. Create module-level architecture sketches

**Phase 1 (Design & Contracts)**:
1. Define content entity model (`data-model.md`)
2. Generate module structure schema (`contracts/module-schema.yaml`)
3. Generate sidebar configuration (`contracts/sidebar-structure.js`)
4. Create writer's quickstart guide (`quickstart.md`)
5. Update agent context with technologies used

**Phase 2 (Implementation - via /sp.tasks)**:
1. Create tasks for each module chapter
2. Research while writing (just-in-time verification)
3. Cite official documentation inline
4. Validate against module checklist
5. Review and iterate

### Verification Sources

**Mandatory Documentation References**:
- ROS 2 Humble Documentation: https://docs.ros.org/en/humble/
- Gazebo Classic/Fortress Documentation: https://gazebosim.org/docs
- Unity Robotics Hub: https://github.com/Unity-Technologies/Unity-Robotics-Hub
- NVIDIA Isaac Sim/SDK: https://docs.omniverse.nvidia.com/isaacsim/latest/
- OpenAI Whisper: https://github.com/openai/whisper
- Docusaurus Documentation: https://docusaurus.io/docs (via Context7 MCP)

**Verification Protocol**:
1. Every API reference must link to official docs
2. Every code example must be syntactically valid
3. Every architectural claim must cite source or be marked as "synthesis"
4. Every framework feature must be verified in current stable release

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Framework documentation changes | High - broken links, outdated info | Pin to specific versions (ROS 2 Humble, Isaac 2023.1.1), use version tags in links |
| VLA examples scarce | Medium - harder to create accurate Module 4 | Research open-source implementations (RT-1, RT-2 papers), focus on architecture over specific models |
| Docusaurus build failures | High - deployment blocked | Continuous validation in PR workflow, local build tests before commit |
| Technical inaccuracies | High - reader trust loss | Mandatory citation policy, peer review checklist, test all code examples |
| Module dependencies unclear | Medium - reader confusion | Explicit dependency callouts in integration sections, prerequisite statements |
| Content not RAG-friendly | Medium - future chatbot quality | Modular structure, clear headings, consistent terminology, glossary |

## Next Steps

**After /sp.plan completion**:
1. Review and approve this plan
2. Proceed to **Phase 0**: Generate `research.md` (detailed in next section)
3. Proceed to **Phase 1**: Generate `data-model.md`, `contracts/`, `quickstart.md`
4. Run `/sp.tasks` to create implementation tasks

**Phase 0 will resolve**:
- Detailed Docusaurus configuration decisions (via Context7 MCP research)
- Module ordering rationale with pedagogical theory
- Specific version pinning for all frameworks
- Architecture diagram standards and tooling choices
- Integration patterns between modules (concrete examples)

## Decision Log

### Decision: Use Docusaurus over alternatives
**Rationale**:
- Industry-standard documentation platform (used by React, Jest, Babel)
- Built-in MDX support for interactive components
- Excellent mobile responsiveness
- Easy GitHub Pages deployment
- Strong community and ecosystem
- Free and open source

**Alternatives Considered**:
- Sphinx: Python ecosystem, less modern UI
- MkDocs: Simpler but less extensible
- GitBook: Commercial, limited free tier
- Custom React app: Unnecessary complexity

**Tradeoffs**:
- Locked into React ecosystem (acceptable given target audience)
- Node.js build dependency (widely available)

### Decision: Research-concurrent writing over upfront research
**Rationale**:
- Prevents analysis paralysis
- Ensures research is focused and relevant
- Enables faster iteration cycles
- Reduces wasted effort on unused research

**Tradeoffs**:
- Requires discipline to verify facts during writing
- May discover gaps that require backtracking

### Decision: Four modules (ROS → Simulation → AI → VLA)
**Rationale**:
- Pedagogical progression: foundation → testing → intelligence → integration
- Industry-standard tools at each layer
- Each module independently valuable
- Clear integration story

**Alternatives Considered**:
- Three modules (merge AI + VLA): Loses focus, too complex for one module
- Five modules (split simulation into Gazebo vs Unity): Unnecessary duplication
- Different order (e.g., VLA first): Lacks foundation, confusing for beginners

**Tradeoffs**:
- Four modules may feel lengthy (mitigated by self-contained structure)
- Module 4 depends on 1-3 (mitigated by clear prerequisites)
