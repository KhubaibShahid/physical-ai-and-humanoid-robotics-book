# Research & Architecture Decisions

**Feature**: Physical AI & Humanoid Robotics Book
**Date**: 2025-12-22
**Phase**: 0 (Research & Architecture)

## Purpose

This document captures research findings and architectural decisions for the Physical AI & Humanoid Robotics educational content. All unknowns from Technical Context have been resolved through research of official documentation and best practices.

## Executive Summary

**Decision**: Build a 4-module Docusaurus-based educational resource following research-concurrent writing methodology with strict technical verification.

**Key Technologies**:
- Docusaurus v3.x for content platform
- ROS 2 Humble (LTS) for robotics framework
- Gazebo Fortress + Unity 2022 LTS for simulation
- NVIDIA Isaac Sim 2023.1.1 for AI/perception
- Open-source VLA implementations for capstone

**Module Architecture**: Foundation (ROS 2) → Testing (Simulation) → Intelligence (Isaac) → Integration (VLA)

---

## Research Areas

### 1. Docusaurus Configuration & Best Practices

**Research Question**: How should we configure Docusaurus for optimal educational content delivery and RAG ingestion?

**Sources**:
- Docusaurus Documentation: https://docusaurus.io/docs
- Best practices from React, Jest, and Babel documentation sites
- RAG-optimized content structure patterns

**Findings**:

**Docusaurus Version Selection**:
- **Decision**: Use Docusaurus v3.x (latest stable)
- **Rationale**: Modern MDX v2 support, improved performance, active development
- **Alternatives**: v2.x (older, less MDX features), custom React app (too complex)

**Key Configuration Decisions**:

1. **Sidebar Structure**:
   ```javascript
   // sidebars.js pattern
   module.exports = {
     tutorialSidebar: [
       'intro',
       {
         type: 'category',
         label: 'Module 1: ROS 2',
         items: ['module-01-ros2/01-overview', '...'],
       },
       // ... modules 2-4
     ],
   };
   ```
   - **Rationale**: Category-based organization matches module structure
   - **RAG Benefit**: Clear hierarchy for chunk boundaries

2. **Frontmatter Standards**:
   ```yaml
   ---
   id: unique-page-id
   title: Page Title
   sidebar_label: Short Label
   sidebar_position: 1
   description: SEO and RAG-friendly description
   keywords: [ros2, robotics, simulation]
   ---
   ```
   - **Rationale**: Consistent metadata enables better search and RAG retrieval
   - **Required Fields**: id, title, description, keywords

3. **MDX Component Strategy**:
   - Custom components for: diagrams, code examples, integration flows
   - **Rationale**: Consistent presentation, reusability, interactive learning
   - **Components to create**:
     - `<RobotDiagram />`: System architecture visualizations
     - `<CodeExample />`: Enhanced code blocks with copy, syntax highlight, annotations
     - `<IntegrationFlow />`: Module integration diagrams
     - `<Callout />`: Important notes, warnings, tips

4. **Code Block Configuration**:
   - Use Prism with language support: Python, YAML, XML (URDF), C++, JavaScript
   - Enable line numbering, syntax highlighting, copy button
   - **Rationale**: Matches frameworks used (ROS 2 = Python/C++, configs = YAML)

5. **Build & Deployment**:
   - GitHub Actions workflow for CI/CD
   - Deploy to GitHub Pages on main branch
   - **Validation steps**: MDX syntax check, link check, build success

**Best Practices Applied**:
- **Versioning**: Use Docusaurus versioning for future updates (not immediately needed)
- **Search**: Algolia DocSearch (free for open source) or local search plugin
- **Mobile**: Default Docusaurus theme is mobile-responsive (no custom work needed)
- **Accessibility**: Use semantic HTML, alt text for images, ARIA labels where needed

---

### 2. Module Ordering & Pedagogical Rationale

**Research Question**: What is the optimal order for teaching Physical AI & Humanoid Robotics concepts?

**Sources**:
- Pedagogical theory: Bloom's Taxonomy, Constructivism
- Existing robotics curricula (MIT, Stanford, Carnegie Mellon)
- Industry onboarding practices (ROS Industrial, NVIDIA developer programs)

**Findings**:

**Chosen Order**: ROS 2 → Digital Twin → AI Brain → VLA

**Pedagogical Justification**:

1. **Module 1: ROS 2 (Foundation)**
   - **Bloom's Level**: Remember, Understand
   - **Constructivist Principle**: Build mental model of robot software communication
   - **Why First**: Students need vocabulary and concepts before simulation/AI
   - **Learning Progression**: Nodes → Topics → Services → URDF (simple to complex)
   - **Prerequisites**: CS background (programming), basic AI knowledge
   - **Deliverable**: Student can explain pub-sub pattern and robot structure

2. **Module 2: Digital Twin (Application)**
   - **Bloom's Level**: Apply, Analyze
   - **Constructivist Principle**: Practice ROS 2 concepts in safe environment
   - **Why Second**: Builds on Module 1, enables experimentation without hardware
   - **Learning Progression**: Simulation basics → Physics → Sensor modeling → ROS integration
   - **Prerequisites**: Module 1 (ROS 2 concepts)
   - **Deliverable**: Student can create and test robot behaviors in simulation

3. **Module 3: AI Brain (Analysis & Synthesis)**
   - **Bloom's Level**: Analyze, Evaluate
   - **Constructivist Principle**: Understand how AI enables autonomy
   - **Why Third**: Requires foundation (Module 1) and testing capability (Module 2)
   - **Learning Progression**: Perception → SLAM → Navigation → Integration with ROS + Sim
   - **Prerequisites**: Modules 1-2, AI/ML basics (neural nets, training)
   - **Deliverable**: Student can explain autonomous navigation pipeline

4. **Module 4: VLA (Integration & Capstone)**
   - **Bloom's Level**: Create, Evaluate
   - **Constructivist Principle**: Synthesize all prior learning into complete system
   - **Why Last**: Integrates all previous modules, demonstrates full stack
   - **Learning Progression**: Voice input → LLM planning → Action execution → Full capstone
   - **Prerequisites**: Modules 1-3
   - **Deliverable**: Student can trace voice-to-action flow across entire stack

**Alternatives Considered**:
- **Top-Down (VLA → ROS 2)**: Exciting but lacks foundation, confusing for beginners
- **Concurrent (All modules together)**: Overwhelming, no clear progression
- **AI-First (Isaac → ROS 2)**: Disconnects from practical implementation

**Supporting Evidence**:
- MIT 6.4212 (Robotic Manipulation): Teaches kinematics before planning
- Stanford CS237B: Covers fundamentals (coordinate frames) before perception
- NVIDIA Isaac tutorials: Assume ROS knowledge before Isaac SDK

---

### 3. Framework Version Selection & Compatibility

**Research Question**: Which specific versions of ROS 2, Gazebo, Unity, Isaac, and VLA tools should we use?

**Sources**:
- Official framework documentation
- LTS (Long-Term Support) schedules
- Community adoption metrics (GitHub stars, forum activity)

**Findings**:

**Version Matrix**:

| Framework | Version | Release Date | EOL Date | Rationale |
|-----------|---------|--------------|----------|-----------|
| ROS 2 | Humble Hawksbill | May 2022 | May 2027 | LTS release, 5-year support, widely adopted |
| Gazebo | Fortress (LTS) | Sep 2021 | Sep 2026 | Stable LTS, good ROS 2 integration |
| Unity | 2022 LTS | Nov 2022 | Nov 2024 (then 2023 LTS) | Current LTS, Robotics packages supported |
| NVIDIA Isaac Sim | 2023.1.1 | Q2 2023 | Active | Latest stable, comprehensive docs |
| Python | 3.10+ | Oct 2021 | Oct 2026 | ROS 2 Humble requirement |
| Node.js | 18 LTS | Apr 2022 | Apr 2025 | Docusaurus 3.x requirement |

**Decision Rationale**:
- **LTS Priority**: Chose long-term support versions to maximize content lifespan
- **ROS 2 Humble**: Most widely adopted LTS, excellent documentation, 5-year support
- **Gazebo Fortress vs Harmonic**: Fortress more stable, better tutorials available
- **Unity vs Unreal**: Unity has better ROS integration packages, lower learning curve
- **Isaac Sim 2023.1.1**: Latest stable with comprehensive Omniverse integration

**Compatibility Matrix**:
- ✅ ROS 2 Humble + Gazebo Fortress: Official support via `ros_gz` packages
- ✅ ROS 2 Humble + Unity 2022 LTS: Via Unity Robotics Hub packages
- ✅ ROS 2 Humble + Isaac Sim 2023.1.1: Native ROS 2 bridge included
- ✅ Python 3.10 + All frameworks: Supported by all

**Documentation Links to Include**:
- ROS 2 Humble: `https://docs.ros.org/en/humble/`
- Gazebo Fortress: `https://gazebosim.org/docs/fortress`
- Unity Robotics Hub: `https://github.com/Unity-Technologies/Unity-Robotics-Hub`
- Isaac Sim: `https://docs.omniverse.nvidia.com/isaacsim/latest/`

---

### 4. VLA (Vision-Language-Action) Implementation Approach

**Research Question**: How should we teach VLA systems given limited production implementations?

**Sources**:
- Research papers: RT-1 (Google), RT-2 (Google), PaLM-E, SayCan
- Open-source implementations: RT-X, Open-VLA
- Industry examples: Boston Dynamics, Figure AI

**Findings**:

**Challenge**: VLA is cutting-edge with limited open-source production code

**Decision**: Focus on architecture and integration patterns, not specific model training

**Module 4 Structure**:

1. **Voice Input Layer**:
   - **Technology**: OpenAI Whisper (open source, well-documented)
   - **Rationale**: Industry-standard speech recognition, easy to integrate
   - **Link**: `https://github.com/openai/whisper`

2. **Language Planning Layer**:
   - **Technology**: LLM APIs (OpenAI GPT-4, or open models like LLaMA)
   - **Approach**: Teach prompt engineering for task decomposition
   - **Example**: "Pick up red block" → [navigate_to(red_block), grasp(red_block), navigate_to(table), place(table)]

3. **Action Execution Layer**:
   - **Technology**: ROS 2 Action servers (from Module 1)
   - **Integration**: LLM output → ROS 2 actions (navigation, manipulation)
   - **Rationale**: Connects to all prior modules

4. **Capstone Project**:
   - **Scenario**: Voice-commanded household robot
   - **Flow**: Voice → Whisper → LLM → ROS Actions → Isaac Navigation → Gazebo/Unity Execution
   - **Focus**: System architecture, not model training

**What to Avoid**:
- ❌ Training custom VLA models (out of scope, requires significant compute)
- ❌ Claiming specific accuracy metrics without empirical evidence
- ❌ Presenting research prototypes as production-ready

**What to Include**:
- ✅ RT-1/RT-2 architecture diagrams (cited from papers)
- ✅ Explanation of vision-language grounding
- ✅ Practical integration using available tools (Whisper + LLM APIs + ROS)
- ✅ Discussion of limitations and research directions

**Research Citations**:
- RT-1 Paper: `https://arxiv.org/abs/2212.06817`
- RT-2 Paper: `https://arxiv.org/abs/2307.15818`
- SayCan Paper: `https://arxiv.org/abs/2204.01691`

---

### 5. Architecture Diagram Standards & Tooling

**Research Question**: How should we create and maintain consistent architecture diagrams?

**Sources**:
- Technical documentation best practices (C4 model, UML)
- Diagram-as-code tools (Mermaid, PlantUML, Excalidraw)
- Docusaurus diagram plugin ecosystem

**Findings**:

**Decision**: Use Mermaid for simple diagrams, Excalidraw for complex architecture diagrams

**Rationale**:

1. **Mermaid**:
   - **Pros**: Text-based (version control friendly), Docusaurus plugin available, simple syntax
   - **Use Cases**: Flow charts, sequence diagrams, simple component diagrams
   - **Integration**: `@docusaurus/plugin-content-docs` supports Mermaid in code blocks
   - **Example**:
     ```mermaid
     graph TD
       A[Voice Input] --> B[Whisper STT]
       B --> C[LLM Planner]
       C --> D[ROS Actions]
       D --> E[Robot Execution]
     ```

2. **Excalidraw**:
   - **Pros**: Visual, easy to create complex diagrams, export to SVG/PNG
   - **Use Cases**: System architecture, robot diagrams, integration flows
   - **Workflow**: Create in Excalidraw → Export SVG → Include in MDX
   - **Rationale**: Better for detailed technical diagrams than Mermaid

3. **Custom React Components**:
   - **Component**: `<RobotDiagram>` for interactive visualizations
   - **Use Case**: When interactivity needed (hover states, clickable components)
   - **Rationale**: Enhances learning for complex systems

**Diagram Standards**:
- **Color Scheme**: Consistent across all diagrams (e.g., blue=data flow, green=components, red=errors)
- **Labels**: Clear, concise, use consistent terminology from glossary
- **Complexity**: Keep diagrams focused (max 7-10 components per diagram)
- **Accessibility**: Include alt text describing diagram content

**Storage**:
- Mermaid: Inline in MDX files
- SVG/PNG: `static/img/module-XX/diagram-name.svg`
- Excalidraw sources: `static/diagrams/source/` (for future editing)

---

### 6. Integration Patterns Between Modules

**Research Question**: How do we clearly show integration between modules?

**Findings**:

**Integration Pattern Template** (for each module's integration chapter):

```markdown
## Integration with Previous Modules

### Prerequisites
- List specific concepts from prior modules needed
- Link to relevant chapters

### Integration Architecture
[Diagram showing how this module connects to previous]

### Concrete Example
- Scenario: [Specific use case]
- Module X provides: [Specific component/data]
- This module uses: [How it's consumed]
- Code Example: [Working integration code]

### Common Integration Patterns
- Pattern 1: [Name and description]
- Pattern 2: [Name and description]

### Troubleshooting Integration
- Issue 1: [Common problem and solution]
- Issue 2: [Common problem and solution]
```

**Specific Integration Examples**:

1. **Module 1 → 2 (ROS to Simulation)**:
   - **Integration Point**: URDF robot model
   - **Pattern**: Export URDF from ROS, import to Gazebo/Unity
   - **Example**: TurtleBot3 URDF in Gazebo with ROS 2 control

2. **Module 2 → 3 (Simulation to Isaac)**:
   - **Integration Point**: Sensor data streams
   - **Pattern**: Gazebo publishes camera/LIDAR → Isaac consumes for SLAM
   - **Example**: Isaac Visual SLAM consuming Gazebo camera feed

3. **Module 3 → 4 (Isaac to VLA)**:
   - **Integration Point**: Navigation goals from LLM planning
   - **Pattern**: LLM generates waypoints → Isaac navigation stack executes
   - **Example**: "Go to kitchen" → LLM maps to coordinates → Isaac navigates

---

## Resolved Unknowns from Technical Context

All "NEEDS CLARIFICATION" items from plan.md Technical Context have been resolved:

| Original Unknown | Resolution |
|------------------|------------|
| Language/Version | Docusaurus v3.x, Node.js 18+, Python 3.10+ (for code examples) |
| Primary Dependencies | Docusaurus, React, Prism, Context7 MCP (research), framework-specific tools (ROS/Gazebo/Unity/Isaac) |
| Storage | Git for content, GitHub Pages for deployment, future Neon/Qdrant for RAG |
| Testing | Docusaurus build, MDX syntax, link checking, content validation checklists |
| Performance Goals | <2s page load, <5min build, mobile-responsive |
| Constraints | Verified code, cited sources, free-tier only, Docusaurus MDX format |
| Scale/Scope | 4 modules, 15-20 chapters, ~50-100 pages, 30-50 code examples |

---

## Summary of Key Decisions

1. ✅ **Docusaurus v3.x** with MDX v2 for content platform
2. ✅ **Four-module progression**: ROS 2 → Simulation → Isaac → VLA
3. ✅ **LTS versions**: ROS 2 Humble, Gazebo Fortress, Unity 2022 LTS, Isaac 2023.1.1
4. ✅ **Research-concurrent writing** with mandatory citation policy
5. ✅ **Mermaid + Excalidraw** for diagrams, custom React components for interactive content
6. ✅ **Integration pattern template** for consistent module connections
7. ✅ **VLA focus on architecture** over model training (using Whisper + LLM APIs + ROS)

---

## Next Phase: Design & Contracts (Phase 1)

With research complete, Phase 1 will generate:
- `data-model.md`: Content entities (Module, Chapter, CodeExample, Diagram)
- `contracts/module-schema.yaml`: Frontmatter validation schema
- `contracts/sidebar-structure.js`: Docusaurus sidebar configuration
- `contracts/content-checklist.yaml`: Per-module quality validation
- `quickstart.md`: Writer's guide for creating new content
