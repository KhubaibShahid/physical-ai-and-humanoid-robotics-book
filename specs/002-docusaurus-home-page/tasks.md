---
description: "Task list for Docusaurus Home Page implementation"
---

# Tasks: Docusaurus Home Page with Book and Chatbot Buttons

**Input**: Design documents from `/specs/002-docusaurus-home-page/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: No test tasks included - not explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: Docusaurus static site structure
- Main file: `src/pages/index.mdx`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and verification of Docusaurus structure

- [X] T001 Verify Docusaurus v3.x is installed and configured in package.json
- [X] T002 Verify src/pages directory exists or create it
- [X] T003 [P] Verify docusaurus.config.js has correct homepage routing configuration
- [X] T004 [P] Check that Node.js 18+ is available in the environment

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core setup that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Verify Docusaurus builds successfully with `npm run build`
- [X] T006 Verify Docusaurus dev server starts with `npm start`
- [X] T007 Create backup of any existing src/pages/index.mdx if present

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Access Book Content (Priority: P1) 🎯 MVP

**Goal**: Create homepage that introduces the book with hero section, book description, and "Read the Book" button linking to Module 1

**Independent Test**: Navigate to homepage, verify hero title "Physical AI & Humanoid Robotics" is displayed, verify book description paragraph is visible, click "Read the Book" button and verify it navigates to Module 1

### Implementation for User Story 1

- [X] T008 [US1] Create src/pages/index.mdx file with basic MDX frontmatter and structure
- [X] T009 [US1] Add hero title "Physical AI & Humanoid Robotics" to src/pages/index.mdx
- [X] T010 [US1] Add hero description (brief explanation of the book) to src/pages/index.mdx
- [X] T011 [US1] Add book description paragraph explaining Physical AI and embodied intelligence to src/pages/index.mdx
- [X] T012 [US1] Add "Read the Book" button component with link to Module 1 (/docs/module-1/ or appropriate path) in src/pages/index.mdx
- [X] T013 [US1] Verify the homepage renders correctly in development server
- [X] T014 [US1] Verify the "Read the Book" button links correctly to Module 1

**Checkpoint**: At this point, User Story 1 should be fully functional - homepage displays hero content and button navigates to Module 1

---

## Phase 4: User Story 2 - Access Chatbot Feature (Priority: P2)

**Goal**: Add "Ask the Chatbot" button that links to a placeholder page for the future RAG chatbot

**Independent Test**: Navigate to homepage, verify "Ask the Chatbot" button is displayed, click button and verify it navigates to a placeholder page indicating the chatbot feature is coming soon

### Implementation for User Story 2

- [X] T015 [US2] Create placeholder page for chatbot at src/pages/chatbot.mdx or appropriate location
- [X] T016 [US2] Add "Coming Soon" message and description to the chatbot placeholder page
- [X] T017 [US2] Add "Ask the Chatbot" button component to src/pages/index.mdx
- [X] T018 [US2] Configure "Ask the Chatbot" button to link to the placeholder page in src/pages/index.mdx
- [X] T019 [US2] Verify both buttons ("Read the Book" and "Ask the Chatbot") display correctly together
- [X] T020 [US2] Verify the "Ask the Chatbot" button links correctly to the placeholder page

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - homepage has both buttons functioning

---

## Phase 5: User Story 3 - Understand Book Structure (Priority: P3)

**Goal**: Display overview of the 4-module structure with one-line descriptions for each module

**Independent Test**: Navigate to homepage, verify a section showing all 4 modules is visible, verify each module has a title and one-line description

### Implementation for User Story 3

- [X] T021 [US3] Add section heading for module overview (e.g., "Book Modules" or "What You'll Learn") to src/pages/index.mdx
- [X] T022 [P] [US3] Add Module 1 title and one-line description to the modules section in src/pages/index.mdx
- [X] T023 [P] [US3] Add Module 2 title and one-line description to the modules section in src/pages/index.mdx
- [X] T024 [P] [US3] Add Module 3 title and one-line description to the modules section in src/pages/index.mdx
- [X] T025 [P] [US3] Add Module 4 title and one-line description to the modules section in src/pages/index.mdx
- [X] T026 [US3] Format the modules section with appropriate styling (list or grid layout) in src/pages/index.mdx
- [X] T027 [US3] Verify all 4 modules display correctly with their descriptions

**Checkpoint**: All user stories should now be independently functional - complete homepage with hero, buttons, and module overview

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final improvements and validation across all user stories

- [X] T028 [P] Verify responsive design works on mobile, tablet, and desktop screen sizes
- [X] T029 [P] Add minimal custom CSS styling if needed to maintain clean, non-marketing tone
- [X] T030 Verify all internal links work correctly (Module 1, chatbot placeholder)
- [X] T031 Test page load time meets the 3-second requirement from success criteria
- [X] T032 Run `npm run build` to verify the site builds successfully without errors
- [X] T033 [P] Review content for clean, minimal, non-marketing tone per specification
- [X] T034 Verify MDX syntax follows Docusaurus conventions and constitution requirements
- [X] T035 Test the homepage on different browsers (Chrome, Firefox, Safari if available)
- [X] T036 [P] Update quickstart.md if any setup steps have changed during implementation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Independent of US1 (both buttons can be added separately)
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Independent of US1/US2

### Within Each User Story

- All tasks modify the same file (src/pages/index.mdx) so must be done sequentially within each story
- However, US2 and US3 could be worked on in parallel by different developers as they modify different sections

### Parallel Opportunities

- Phase 1 tasks T003 and T004 can run in parallel
- Phase 5 tasks T022-T025 (adding individual module descriptions) can run in parallel if different developers work on different sections
- Phase 6 tasks T028, T029, T033, T036 can run in parallel

---

## Parallel Example: User Story 3

```bash
# Launch all module additions together (if team has capacity):
Task: "Add Module 1 title and one-line description to src/pages/index.mdx"
Task: "Add Module 2 title and one-line description to src/pages/index.mdx"
Task: "Add Module 3 title and one-line description to src/pages/index.mdx"
Task: "Add Module 4 title and one-line description to src/pages/index.mdx"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - verifies Docusaurus is ready)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test homepage with hero and "Read the Book" button
5. Deploy/demo if ready - this alone provides value

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP - basic homepage!)
3. Add User Story 2 → Test independently → Deploy/Demo (adds chatbot button)
4. Add User Story 3 → Test independently → Deploy/Demo (adds module overview)
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (hero and "Read the Book" button)
   - Developer B: User Story 2 (chatbot button and placeholder)
   - Developer C: User Story 3 (module overview)
3. Stories complete and integrate independently into src/pages/index.mdx
4. Note: Since all modify the same file, coordination needed for merging

---

## Notes

- [P] tasks = different sections/files, minimal dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- All tasks modify src/pages/index.mdx - coordinate merges carefully
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Follow Docusaurus MDX conventions and constitution requirements
- Maintain minimal, clean, non-marketing tone throughout