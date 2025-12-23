# Implementation Tasks: Physical AI & Humanoid Robotics Book

**Feature**: Physical AI & Humanoid Robotics Book
**Branch**: `001-physical-ai-humanoid-robotics-book`
**Date**: 2025-12-22
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

## Overview

This document contains actionable implementation tasks organized by user story for the Physical AI & Humanoid Robotics educational content. Each phase corresponds to a user story from the specification, ensuring independent delivery and testing.

**Total Tasks**: 68
**Parallelization Opportunities**: 45 tasks can run in parallel within their phases
**MVP Scope**: Phase 3 (User Story 1 - Module 1: ROS 2)

---

## Implementation Strategy

**Approach**: Incremental delivery by user story (module)
- Each module is independently testable
- Modules build on each other but can be delivered sequentially
- MVP = Module 1 (ROS 2 Foundation)
- Each module follows consistent structure: Overview → Architecture → Tooling → Integration → Summary

**Dependencies**:
1. **Phase 1** (Setup) → Must complete before all user stories
2. **Phase 2** (Foundational) → Must complete before all user stories
3. **Phase 3** (US1 - Module 1) → MVP, no dependencies on other stories
4. **Phase 4** (US2 - Module 2) → Depends on US1 concepts (ROS 2 basics)
5. **Phase 5** (US3 - Module 3) → Depends on US1 (ROS 2), soft dependency on US2 (simulation testing)
6. **Phase 6** (US4 - Module 4) → Depends on US1, US2, US3 (integrates all modules)
7. **Phase 7** (Polish) → After all user stories complete

---

## User Story Mapping

| User Story | Module | Priority | Chapters | Independent Test |
|------------|--------|----------|----------|------------------|
| US1 | Module 1: ROS 2 | P1 | 5 chapters | Explain pub-sub, create ROS node, describe URDF |
| US2 | Module 2: Digital Twin | P2 | 5 chapters | Create simulated environment, verify sensor data |
| US3 | Module 3: AI Brain (Isaac) | P2 | 5 chapters | Explain VSLAM, configure Isaac navigation |
| US4 | Module 4: VLA | P3 | 6 chapters (+ capstone) | Voice-to-action demo with LLM planning |

---

## Phase 1: Project Setup

**Goal**: Initialize Docusaurus project structure and core infrastructure
**Independent Test**: `npm run build` succeeds, project structure matches plan.md

### Tasks

- [ ] T001 Initialize Docusaurus v3.x project with `npx create-docusaurus@latest physical-ai-humanoid-robotics classic`
- [ ] T002 [P] Configure Node.js version (18+) in package.json engines field and .nvmrc file
- [ ] T003 [P] Install core dependencies: Docusaurus plugins, React, MDX processor in package.json
- [ ] T004 [P] Create project directory structure per plan.md: docs/, src/components/, static/img/, static/code-examples/
- [ ] T005 [P] Create module directories: docs/module-01-ros2/, docs/module-02-digital-twin/, docs/module-03-ai-brain/, docs/module-04-vla/
- [ ] T006 [P] Create appendix directory: docs/appendix/ with placeholder files
- [ ] T007 [P] Create static asset directories: static/img/module-01/, static/img/module-02/, static/img/module-03/, static/img/module-04/
- [ ] T008 [P] Create code examples directories: static/code-examples/ros2-examples/, static/code-examples/simulation-examples/, static/code-examples/isaac-examples/, static/code-examples/vla-examples/
- [ ] T009 Configure docusaurus.config.js with project metadata: title, tagline, URL, baseUrl, GitHub org/repo
- [ ] T010 [P] Configure Docusaurus plugins in docusaurus.config.js: docs, blog (disabled), pages, syntax highlighting (Prism)
- [ ] T011 Copy sidebar structure from specs/001-physical-ai-humanoid-robotics-book/contracts/sidebar-structure.js to sidebars.js
- [ ] T012 [P] Configure Prism language support in docusaurus.config.js: Python, C++, YAML, XML, Bash, JavaScript
- [ ] T013 [P] Create custom CSS file: src/css/custom.css with base styling and theme colors
- [ ] T014 [P] Set up Git ignore patterns in .gitignore: node_modules/, .docusaurus/, build/, .DS_Store
- [ ] T015 Verify build with `npm run build` and fix any configuration errors

**Parallel Execution**: T002-T008, T010, T012-T014 can run in parallel after T001

---

## Phase 2: Foundational Components

**Goal**: Create shared infrastructure used by all modules
**Independent Test**: Custom components render in MDX, frontmatter validates against schema

### Tasks

- [ ] T016 [P] Create RobotDiagram React component in src/components/RobotDiagram.js for architecture visualizations
- [ ] T017 [P] Create CodeExample React component in src/components/CodeExample.js with syntax highlighting and copy button
- [ ] T018 [P] Create IntegrationFlow React component in src/components/IntegrationFlow.js for module integration diagrams
- [ ] T019 [P] Create Callout component in src/components/Callout.js for notes, warnings, tips
- [ ] T020 [P] Create intro page: docs/intro.md with book overview, target audience, prerequisites, how to use
- [ ] T021 [P] Create appendix/glossary.md with placeholder structure for technical terms
- [ ] T022 [P] Create appendix/resources.md with categories: official docs, tutorials, tools, communities
- [ ] T023 [P] Create appendix/troubleshooting.md with placeholder structure for common issues
- [ ] T024 Set up GitHub Actions workflow: .github/workflows/build-deploy.yml for GitHub Pages deployment
- [ ] T025 [P] Set up content validation workflow: .github/workflows/content-validation.yml for PR checks (build, link check, MDX syntax)
- [ ] T026 [P] Create build validation test: tests/build-validation.test.js to ensure `npm run build` succeeds
- [ ] T027 [P] Create link checker test: tests/link-checker.test.js for internal and external link validation
- [ ] T028 [P] Create MDX syntax validation test: tests/syntax-validation.test.js to catch syntax errors

**Parallel Execution**: T016-T019 (components), T020-T023 (base content), T025-T028 (testing) can run in parallel

---

## Phase 3: User Story 1 - Module 1: ROS 2 (Foundation) [P1]

**Goal**: Create Module 1 content teaching ROS 2 fundamentals (nodes, topics, services, URDF)
**Independent Test**: Student can explain pub-sub pattern, create ROS 2 node, describe URDF kinematics
**Why First**: Foundation for all subsequent modules; ROS 2 is prerequisite for simulation, AI, and VLA

### Tasks

#### Module 1 Setup

- [ ] T029 [P] [US1] Create module category config: docs/module-01-ros2/_category_.json with label, position, description
- [ ] T030 [P] [US1] Create image directory structure: static/img/module-01/ for diagrams and screenshots

#### Chapter 1.1: Overview

- [ ] T031 [US1] Create docs/module-01-ros2/01-overview.md with frontmatter (id, title, sidebar_label, position, description, keywords)
- [ ] T032 [P] [US1] Write concept introduction section explaining what ROS 2 is and why it matters for humanoid robotics
- [ ] T033 [P] [US1] Write section on ROS 2 vs ROS 1 differences (DDS middleware, real-time, multi-robot)
- [ ] T034 [P] [US1] Write learning objectives section listing what students will learn in Module 1
- [ ] T035 [P] [US1] Create Mermaid diagram showing ROS 2 architecture layers (application, client libraries, middleware)
- [ ] T036 [P] [US1] Add references section citing ROS 2 Humble documentation

#### Chapter 1.2: Architecture

- [ ] T037 [US1] Create docs/module-01-ros2/02-architecture.md with frontmatter
- [ ] T038 [P] [US1] Write system architecture section explaining ROS 2 graph concept (nodes as processes)
- [ ] T039 [P] [US1] Write data flow section explaining publish-subscribe pattern with concrete examples
- [ ] T040 [P] [US1] Write component interactions section covering topics, services, actions
- [ ] T041 [P] [US1] Create architecture diagram: static/img/module-01/ros2-graph-architecture.svg showing nodes, topics, services
- [ ] T042 [P] [US1] Create Mermaid sequence diagram showing message flow: publisher → topic → subscriber
- [ ] T043 [P] [US1] Add references to ROS 2 concepts documentation

#### Chapter 1.3: Tooling

- [ ] T044 [US1] Create docs/module-01-ros2/03-tooling.md with frontmatter
- [ ] T045 [P] [US1] Write section on ROS 2 nodes: lifecycle, creation, execution
- [ ] T046 [P] [US1] Write section on topics: pub-sub pattern, message types, quality of service (QoS)
- [ ] T047 [P] [US1] Write section on services: request-response pattern, synchronous communication
- [ ] T048 [P] [US1] Write section on actions: long-running tasks, feedback, preemption
- [ ] T049 [P] [US1] Write section on rclpy Python library: basic usage, node creation, publisher/subscriber
- [ ] T050 [P] [US1] Write section on URDF: robot description format, joints, links, kinematic chains
- [ ] T051 [P] [US1] Create code example: simple ROS 2 publisher in Python (static/code-examples/ros2-examples/simple_publisher.py)
- [ ] T052 [P] [US1] Create code example: simple ROS 2 subscriber in Python (static/code-examples/ros2-examples/simple_subscriber.py)
- [ ] T053 [P] [US1] Create code example: basic URDF for simple robot (static/code-examples/ros2-examples/simple_robot.urdf)
- [ ] T054 [P] [US1] Write best practices section: naming conventions, QoS tuning, error handling
- [ ] T055 [P] [US1] Write common pitfalls section: timing issues, message serialization, lifecycle management
- [ ] T056 [P] [US1] Add code example explanations with links to ROS 2 Humble API documentation

#### Chapter 1.4: Integration

- [ ] T057 [US1] Create docs/module-01-ros2/04-integration.md with frontmatter
- [ ] T058 [P] [US1] Write section on ROS 2 as foundation for simulation (preview Module 2)
- [ ] T059 [P] [US1] Write section on ROS 2 as foundation for AI integration (preview Module 3)
- [ ] T060 [P] [US1] Write section on ROS 2 as foundation for VLA systems (preview Module 4)
- [ ] T061 [P] [US1] Create integration architecture diagram showing ROS 2 at the base of the stack
- [ ] T062 [P] [US1] Write concrete example: how a humanoid robot uses ROS 2 nodes for arm control

#### Chapter 1.5: Summary

- [ ] T063 [US1] Create docs/module-01-ros2/05-summary.md with frontmatter
- [ ] T064 [P] [US1] Write key takeaways section summarizing Module 1 concepts
- [ ] T065 [P] [US1] Write limitations and tradeoffs section: ROS 2 complexity, learning curve, overhead
- [ ] T066 [P] [US1] Write next steps section pointing to Module 2 (simulation)
- [ ] T067 [US1] Validate Module 1 against contracts/content-checklist.yaml
- [ ] T068 [US1] Test: Run `npm run build` and verify Module 1 renders correctly without errors

**Parallel Execution**:
- T029-T030 can run in parallel (setup)
- T032-T036 can run in parallel after T031 (Chapter 1.1 sections)
- T038-T043 can run in parallel after T037 (Chapter 1.2 sections)
- T045-T056 can run in parallel after T044 (Chapter 1.3 sections)
- T058-T062 can run in parallel after T057 (Chapter 1.4 sections)
- T064-T066 can run in parallel after T063 (Chapter 1.5 sections)

---

## Phase 4: User Story 2 - Module 2: Digital Twin (Simulation) [P2]

**Goal**: Create Module 2 content teaching simulation with Gazebo and Unity
**Independent Test**: Student can create simulated environment, verify sensor data matches expected behaviors
**Dependencies**: Requires US1 (ROS 2 concepts like URDF, topics for sensor data)

### Tasks

#### Module 2 Setup

- [ ] T069 [P] [US2] Create module category config: docs/module-02-digital-twin/_category_.json
- [ ] T070 [P] [US2] Create image directory: static/img/module-02/
- [ ] T071 [P] [US2] Create code examples directory: static/code-examples/simulation-examples/

#### Chapter 2.1: Overview

- [ ] T072 [US2] Create docs/module-02-digital-twin/01-overview.md with frontmatter
- [ ] T073 [P] [US2] Write concept introduction: what is digital twin, why simulation matters for robotics
- [ ] T074 [P] [US2] Write section on benefits: rapid prototyping, safe testing, no hardware needed
- [ ] T075 [P] [US2] Write section on Gazebo vs Unity: when to use each, strengths/weaknesses
- [ ] T076 [P] [US2] Create comparison diagram: Gazebo vs Unity features
- [ ] T077 [P] [US2] Add references to Gazebo Fortress and Unity Robotics Hub documentation

#### Chapter 2.2: Architecture

- [ ] T078 [US2] Create docs/module-02-digital-twin/02-architecture.md with frontmatter
- [ ] T079 [P] [US2] Write system architecture section: simulation engine, physics engine, sensor models, ROS 2 bridge
- [ ] T080 [P] [US2] Write data flow section: URDF → simulator → sensor data → ROS 2 topics
- [ ] T081 [P] [US2] Create architecture diagram: static/img/module-02/simulation-architecture.svg
- [ ] T082 [P] [US2] Create Mermaid diagram showing ROS 2 integration with Gazebo (ros_gz bridge)
- [ ] T083 [P] [US2] Add references to Gazebo architecture and Unity integration documentation

#### Chapter 2.3: Tooling

- [ ] T084 [US2] Create docs/module-02-digital-twin/03-tooling.md with frontmatter
- [ ] T085 [P] [US2] Write section on Gazebo: installation, worlds, models, plugins
- [ ] T086 [P] [US2] Write section on Unity: installation, Unity Robotics Hub, URDF Importer
- [ ] T087 [P] [US2] Write section on physics engines: ODE (Gazebo), PhysX (Unity), realistic collisions
- [ ] T088 [P] [US2] Write section on sensor modeling: cameras, IMU, LIDAR, depth sensors
- [ ] T089 [P] [US2] Write section on ROS 2 integration: ros_gz packages, Unity ROS TCP Connector
- [ ] T090 [P] [US2] Create code example: Gazebo world file with humanoid robot (static/code-examples/simulation-examples/humanoid_world.sdf)
- [ ] T091 [P] [US2] Create code example: Launch file to start Gazebo with ROS 2 bridge (static/code-examples/simulation-examples/gazebo_launch.py)
- [ ] T092 [P] [US2] Create code example: Unity scene setup with ROS integration (README with steps)
- [ ] T093 [P] [US2] Write best practices: simulation fidelity vs performance, sensor noise, time scaling
- [ ] T094 [P] [US2] Add references to Gazebo plugins and Unity Robotics documentation

#### Chapter 2.4: Integration

- [ ] T095 [US2] Create docs/module-02-digital-twin/04-integration.md with frontmatter
- [ ] T096 [P] [US2] Write prerequisites section linking to Module 1 ROS 2 concepts (URDF, topics)
- [ ] T097 [P] [US2] Write integration architecture section: ROS 2 URDF → Gazebo/Unity → sensor topics
- [ ] T098 [P] [US2] Write concrete example: load Module 1 simple robot URDF into Gazebo
- [ ] T099 [P] [US2] Create integration diagram showing data flow: ROS 2 ↔ Gazebo ↔ Sensor topics
- [ ] T100 [P] [US2] Write troubleshooting section: URDF parsing errors, bridge connection issues

#### Chapter 2.5: Summary

- [ ] T101 [US2] Create docs/module-02-digital-twin/05-summary.md with frontmatter
- [ ] T102 [P] [US2] Write key takeaways section
- [ ] T103 [P] [US2] Write limitations section: sim-to-real gap, physics approximations, computational cost
- [ ] T104 [P] [US2] Write next steps section pointing to Module 3 (AI navigation in simulation)
- [ ] T105 [US2] Validate Module 2 against content checklist
- [ ] T106 [US2] Test: Build and verify Module 2 renders correctly

**Parallel Execution**: Similar structure to Module 1, sections within chapters can run in parallel

---

## Phase 5: User Story 3 - Module 3: AI Brain (Isaac) [P2]

**Goal**: Create Module 3 content teaching AI-driven autonomy with NVIDIA Isaac
**Independent Test**: Student can explain VSLAM pipeline, configure Isaac for navigation
**Dependencies**: Requires US1 (ROS 2), soft dependency on US2 (simulation for testing)

### Tasks

#### Module 3 Setup

- [ ] T107 [P] [US3] Create module category config: docs/module-03-ai-brain/_category_.json
- [ ] T108 [P] [US3] Create image directory: static/img/module-03/
- [ ] T109 [P] [US3] Create code examples directory: static/code-examples/isaac-examples/

#### Chapter 3.1: Overview

- [ ] T110 [US3] Create docs/module-03-ai-brain/01-overview.md with frontmatter
- [ ] T111 [P] [US3] Write concept introduction: AI for robotics, perception, autonomy
- [ ] T112 [P] [US3] Write section on NVIDIA Isaac: Isaac Sim, Isaac SDK, Isaac ROS
- [ ] T113 [P] [US3] Write section on why Isaac: GPU acceleration, perception gems, industry adoption
- [ ] T114 [P] [US3] Create diagram showing Isaac ecosystem components
- [ ] T115 [P] [US3] Add references to Isaac Sim and Isaac ROS documentation

#### Chapter 3.2: Architecture

- [ ] T116 [US3] Create docs/module-03-ai-brain/02-architecture.md with frontmatter
- [ ] T117 [P] [US3] Write system architecture section: perception → SLAM → navigation → planning
- [ ] T118 [P] [US3] Write data flow section: camera → VSLAM → map → path planner → action
- [ ] T119 [P] [US3] Create architecture diagram: static/img/module-03/isaac-perception-pipeline.svg
- [ ] T120 [P] [US3] Create Mermaid diagram showing VSLAM data flow
- [ ] T121 [P] [US3] Add references to Isaac perception and navigation documentation

#### Chapter 3.3: Tooling

- [ ] T122 [US3] Create docs/module-03-ai-brain/03-tooling.md with frontmatter
- [ ] T123 [P] [US3] Write section on perception: camera processing, object detection, segmentation
- [ ] T124 [P] [US3] Write section on Visual SLAM: feature tracking, map building, localization
- [ ] T125 [P] [US3] Write section on navigation: costmaps, path planning algorithms (A*, RRT)
- [ ] T126 [P] [US3] Write section on Isaac ROS packages: isaac_ros_visual_slam, isaac_ros_nvblox
- [ ] T127 [P] [US3] Create code example: Isaac VSLAM configuration (static/code-examples/isaac-examples/vslam_config.yaml)
- [ ] T128 [P] [US3] Create code example: Navigation launch file with Isaac (static/code-examples/isaac-examples/navigation_launch.py)
- [ ] T129 [P] [US3] Write best practices: sensor calibration, map quality, computational resources
- [ ] T130 [P] [US3] Add references to Isaac SLAM and navigation tutorials

#### Chapter 3.4: Integration

- [ ] T131 [US3] Create docs/module-03-ai-brain/04-integration.md with frontmatter
- [ ] T132 [P] [US3] Write prerequisites section linking to Module 1 (ROS 2) and Module 2 (simulation)
- [ ] T133 [P] [US3] Write integration architecture: Gazebo camera → Isaac VSLAM → ROS nav stack
- [ ] T134 [P] [US3] Write concrete example: run Isaac navigation in Gazebo simulation
- [ ] T135 [P] [US3] Create integration diagram: Modules 1+2+3 data flow
- [ ] T136 [P] [US3] Write troubleshooting: GPU requirements, ROS message compatibility

#### Chapter 3.5: Summary

- [ ] T137 [US3] Create docs/module-03-ai-brain/05-summary.md with frontmatter
- [ ] T138 [P] [US3] Write key takeaways section
- [ ] T139 [P] [US3] Write limitations section: GPU dependency, computational cost, calibration needs
- [ ] T140 [P] [US3] Write next steps section pointing to Module 4 (VLA integration)
- [ ] T141 [US3] Validate Module 3 against content checklist
- [ ] T142 [US3] Test: Build and verify Module 3 renders correctly

**Parallel Execution**: Similar structure to Modules 1-2, sections within chapters can run in parallel

---

## Phase 6: User Story 4 - Module 4: VLA (Capstone) [P3]

**Goal**: Create Module 4 content teaching vision-language-action systems with capstone integration
**Independent Test**: Student can build voice-to-action demo with LLM planning
**Dependencies**: Requires US1 (ROS 2), US2 (simulation), US3 (Isaac navigation)

### Tasks

#### Module 4 Setup

- [ ] T143 [P] [US4] Create module category config: docs/module-04-vla/_category_.json
- [ ] T144 [P] [US4] Create image directory: static/img/module-04/
- [ ] T145 [P] [US4] Create code examples directory: static/code-examples/vla-examples/

#### Chapter 4.1: Overview

- [ ] T146 [US4] Create docs/module-04-vla/01-overview.md with frontmatter
- [ ] T147 [P] [US4] Write concept introduction: vision-language-action models, embodied AI
- [ ] T148 [P] [US4] Write section on VLA systems: voice input, LLM planning, robot execution
- [ ] T149 [P] [US4] Write section on state-of-the-art: RT-1, RT-2, SayCan (with citations)
- [ ] T150 [P] [US4] Create diagram showing VLA pipeline: voice → LLM → actions → robot
- [ ] T151 [P] [US4] Add references to RT-1, RT-2 papers and Whisper documentation

#### Chapter 4.2: Architecture

- [ ] T152 [US4] Create docs/module-04-vla/02-architecture.md with frontmatter
- [ ] T153 [P] [US4] Write system architecture section: voice layer, planning layer, execution layer
- [ ] T154 [P] [US4] Write data flow section: audio → text → task plan → ROS actions → execution
- [ ] T155 [P] [US4] Create architecture diagram: static/img/module-04/vla-full-stack.svg
- [ ] T156 [P] [US4] Create Mermaid sequence diagram: user speaks → Whisper → LLM → ROS → Isaac → robot
- [ ] T157 [P] [US4] Add references to LLM planning and embodied AI research

#### Chapter 4.3: Tooling

- [ ] T158 [US4] Create docs/module-04-vla/03-tooling.md with frontmatter
- [ ] T159 [P] [US4] Write section on voice input: Whisper installation, speech-to-text usage
- [ ] T160 [P] [US4] Write section on LLM planning: OpenAI API / open models, prompt engineering for tasks
- [ ] T161 [P] [US4] Write section on action execution: LLM output → ROS 2 action servers
- [ ] T162 [P] [US4] Create code example: Whisper integration (static/code-examples/vla-examples/whisper_stt.py)
- [ ] T163 [P] [US4] Create code example: LLM task planner (static/code-examples/vla-examples/llm_planner.py)
- [ ] T164 [P] [US4] Create code example: ROS action client for navigation (static/code-examples/vla-examples/navigation_client.py)
- [ ] T165 [P] [US4] Write best practices: prompt design, error handling, task decomposition
- [ ] T166 [P] [US4] Add references to Whisper and LLM API documentation

#### Chapter 4.4: Integration

- [ ] T167 [US4] Create docs/module-04-vla/04-integration.md with frontmatter
- [ ] T168 [P] [US4] Write prerequisites section linking to Modules 1, 2, 3
- [ ] T169 [P] [US4] Write full stack integration: voice → plan → navigate (Isaac) → execute (Gazebo)
- [ ] T170 [P] [US4] Write concrete example: "Go to kitchen" command end-to-end flow
- [ ] T171 [P] [US4] Create integration diagram showing all 4 modules working together
- [ ] T172 [P] [US4] Write troubleshooting: LLM hallucinations, ROS action failures, timing issues

#### Chapter 4.5: Summary

- [ ] T173 [US4] Create docs/module-04-vla/05-summary.md with frontmatter
- [ ] T174 [P] [US4] Write key takeaways section
- [ ] T175 [P] [US4] Write limitations section: LLM reliability, task complexity, real-world gaps
- [ ] T176 [P] [US4] Write research directions: VLA model training, multimodal grounding, sim-to-real transfer

#### Chapter 4.6: Capstone Project

- [ ] T177 [US4] Create docs/module-04-vla/06-capstone.md with frontmatter
- [ ] T178 [P] [US4] Write project overview: voice-commanded household robot scenario
- [ ] T179 [P] [US4] Write architecture section showing all modules integrated
- [ ] T180 [P] [US4] Write implementation Part 1: voice input system setup
- [ ] T181 [P] [US4] Write implementation Part 2: LLM planning system
- [ ] T182 [P] [US4] Write implementation Part 3: navigation execution (Isaac + Gazebo)
- [ ] T183 [P] [US4] Write running instructions: step-by-step to run complete system
- [ ] T184 [P] [US4] Write expected output section: demo video link, success criteria
- [ ] T185 [P] [US4] Write limitations and future directions section
- [ ] T186 [P] [US4] Create complete capstone code package: static/code-examples/vla-examples/capstone/
- [ ] T187 [US4] Validate Module 4 against content checklist (including capstone validation)
- [ ] T188 [US4] Test: Build and verify Module 4 renders correctly, capstone flow is traceable

**Parallel Execution**: Similar structure to Modules 1-3, with additional capstone tasks that can run in parallel

---

## Phase 7: Polish & Cross-Cutting Concerns

**Goal**: Finalize supporting content, validation, and deployment configuration
**Independent Test**: Full site builds, all links valid, content passes quality checks

### Tasks

#### Appendix Completion

- [ ] T189 [P] Populate appendix/glossary.md with all technical terms from Modules 1-4 (ROS 2, Gazebo, Isaac, VLA terminology)
- [ ] T190 [P] Populate appendix/resources.md with official documentation links, tutorials, community resources
- [ ] T191 [P] Populate appendix/troubleshooting.md with common issues from all modules and solutions

#### Intro & Landing Page

- [ ] T192 Update docs/intro.md with complete book overview, module summaries, learning path guidance

#### Styling & UX

- [ ] T193 [P] Customize src/css/custom.css with project-specific colors, typography, spacing
- [ ] T194 [P] Add project logo and favicon to static/img/
- [ ] T195 [P] Configure theme settings in docusaurus.config.js: navbar, footer, color mode

#### Validation & Testing

- [ ] T196 Run full content validation against contracts/content-checklist.yaml for all 4 modules
- [ ] T197 [P] Test: Run `npm run build` and ensure zero errors
- [ ] T198 [P] Test: Run link checker and fix any broken internal or external links
- [ ] T199 [P] Test: Validate all chapter frontmatter against contracts/module-schema.yaml
- [ ] T200 [P] Test: Verify all code examples are syntactically correct (Python, YAML, XML linting)
- [ ] T201 Test: Mobile rendering check on at least 2 device sizes

#### Documentation & Deployment

- [ ] T202 [P] Create README.md for repository root: project description, setup instructions, build commands
- [ ] T203 [P] Create CONTRIBUTING.md with writer's guide (link to quickstart.md)
- [ ] T204 [P] Configure GitHub Pages deployment in docusaurus.config.js: organizationName, projectName, deploymentBranch
- [ ] T205 Test: Deploy to GitHub Pages and verify site is accessible
- [ ] T206 [P] Create LICENSE file (MIT or CC BY-SA 4.0 as specified in constitution)

**Parallel Execution**: T189-T191 (appendix), T193-T195 (styling), T197-T200 (validation tests), T202-T204, T206 can run in parallel

---

## Parallel Execution Examples

### Phase 1 (Setup) Parallelization:
```bash
# After T001 completes, run these in parallel:
T002 & T003 & T004 & T005 & T006 & T007 & T008 & T010 & T012 & T013 & T014
# Wait for all to complete, then run T009, T011, T015 sequentially
```

### Phase 3 (Module 1) Chapter 1.3 Parallelization:
```bash
# After T044 completes (create chapter file), run content sections in parallel:
T045 & T046 & T047 & T048 & T049 & T050 & T051 & T052 & T053 & T054 & T055 & T056
```

### Phase 7 (Polish) Parallelization:
```bash
# Run validation tests in parallel:
T197 & T198 & T199 & T200
# Run appendix completion in parallel:
T189 & T190 & T191
# Run styling and docs in parallel:
T193 & T194 & T195 & T202 & T203 & T204 & T206
```

---

## Dependency Graph

```
Phase 1 (Setup)
    ↓
Phase 2 (Foundational)
    ↓
    ├──→ Phase 3 (US1 - Module 1: ROS 2) [MVP]
    │       ↓
    │       ├──→ Phase 4 (US2 - Module 2: Digital Twin)
    │       │       ↓
    │       │       └──→ Phase 5 (US3 - Module 3: AI Brain)
    │       │               ↓
    │       │               └──→ Phase 6 (US4 - Module 4: VLA)
    │       │
    │       └──→ Phase 5 (US3 can start after US1 without US2)
    │               ↓
    │               └──→ Phase 6 (US4 - Module 4: VLA)
    │
    └──→ All User Stories complete
            ↓
        Phase 7 (Polish)
```

**Story Completion Order**:
1. **Required First**: Setup (Phase 1) + Foundational (Phase 2)
2. **MVP**: Module 1 (Phase 3) - independently deliverable and testable
3. **Next Priority**: Module 2 (Phase 4) - builds on Module 1
4. **Parallel Option**: Module 3 (Phase 5) can start after Module 1, doesn't strictly need Module 2
5. **Final Integration**: Module 4 (Phase 6) - requires all previous modules
6. **Completion**: Polish (Phase 7) - after all modules done

---

## Validation Checklist

Before marking each phase complete:

**Per-Module Validation** (Phases 3-6):
- [ ] All chapter files created with valid frontmatter
- [ ] All sections have 300-800 words (recommended range)
- [ ] All code examples are syntactically valid and tested
- [ ] All API references link to official documentation
- [ ] All diagrams have alt text and captions
- [ ] Module builds without errors (`npm run build`)
- [ ] Module passes content-checklist.yaml validation
- [ ] Independent test criteria from spec.md can be verified

**Full Site Validation** (Phase 7):
- [ ] All 4 modules complete and validated
- [ ] Appendix fully populated
- [ ] No broken internal or external links
- [ ] All frontmatter validates against schema
- [ ] Site deploys successfully to GitHub Pages
- [ ] Mobile rendering verified
- [ ] All constitutional principles satisfied (technical accuracy, citations, no plagiarism)

---

## Notes

**Tests**: Tests are validation-focused (build success, link checking, syntax validation) rather than unit tests, as appropriate for educational content.

**Research-Concurrent**: Each task involving content writing assumes research will be done concurrently (verifying against official documentation before writing).

**Independent Testing**: Each user story (module) has clear acceptance criteria from spec.md and can be tested independently.

**MVP Strategy**: Phase 3 (Module 1: ROS 2) alone constitutes a deliverable MVP that provides foundational value to students.

**Incremental Delivery**: Modules can be released sequentially as they complete, providing value earlier while maintaining dependency integrity.
