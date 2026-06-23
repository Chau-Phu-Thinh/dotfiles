# NoteGraph Project Task Plan

## Project Focus

NoteGraph is an Android smart note-taking application inspired by Notion and Obsidian. The application focuses on fast note creation, structured note management, backlinks, graph-based knowledge exploration, reminders, search, and reliable offline-first usage with later backend synchronization.

The project deliverables include a complete requirements document, software design document, unit test plan, final integrated report, and presentation. All tasks should support the main product goal: a stable mobile application that helps users create, organize, search, connect, and review personal notes efficiently.

## Week 4 - Project Initiation and Planning

- Confirm the project topic: NoteGraph Android Smart Note-Taking Application.
- Define the product vision, target users, main problems, and expected outcomes.
- Assign team roles and responsibilities for requirements, design, implementation planning, testing, and documentation.
- Agree on collaboration tools for documentation, diagrams, version control, and test case management.
- Select the initial technology stack for the mobile application, backend, local database, server database, and supporting tools.
- Prepare a detailed 10-week project schedule with milestones, deliverables, and review points.

## Week 5 - Requirements Analysis

- Gather and analyze functional requirements based on the core features:
  - Note management
  - To-do list management
  - Graph view
  - Search
  - Reminder and notification
  - Synchronization
  - User account management
- Identify user classes and characteristics, including general users and advanced users.
- Define actors, use cases, and system boundaries.
- Create use case diagrams for the main application workflows.
- Draft the Software Requirements Specification, including purpose, scope, product overview, user characteristics, assumptions, and constraints.

## Week 6 - SRS Refinement and Module Decomposition

- Complete detailed use case descriptions for important workflows such as creating notes, linking notes, searching notes, setting reminders, and viewing the graph.
- Define functional requirements and non-functional requirements, including usability, performance, reliability, security, maintainability, adaptability, and portability.
- Establish business rules for notes, tags, folders, tasks, reminders, backlinks, and synchronization.
- Decompose the system into major modules, such as:
  - Authentication and user profile module
  - Note and folder management module
  - To-do and reminder module
  - Search and tag module
  - Graph visualization module
  - Synchronization and backup module
- Review the SRS against the project goals and update it based on instructor feedback if available.

## Week 7 - Software Architecture and Data Design

- Define the overall software architecture for the mobile application and backend services.
- Design the offline-first data flow between the React Native application, local SQLite/Room storage, and backend synchronization layer.
- Create conceptual, logical, and physical data models.
- Prepare ERD diagrams for users, notes, folders, tags, tasks, reminders, note links, and synchronization records.
- Define key data structures used for note relationships and graph visualization.
- Document API-first backend boundaries for authentication, note synchronization, backup, and profile management.

## Week 8 - UI/UX Design and Application Flow

- Design wireframes for the main screens:
  - Login and registration
  - Notes list
  - Note editor
  - Folder and tag management
  - To-do list
  - Reminder setup
  - Search results
  - Graph view
  - Profile and settings
- Define navigation flows for common user actions.
- Create flowcharts for critical processes such as note creation, note linking, search, reminder notification, and data synchronization.
- Review UI/UX decisions against the goal of a clean, intuitive, and fast mobile note-taking experience.

## Week 9 - Software Design Finalization

- Finalize the software design document based on instructor feedback.
- Complete architecture diagrams, class diagrams, module descriptions, database design, and UI flow documentation.
- Verify consistency between the SRS, module design, database schema, and user interface flows.
- Ensure every major requirement has a corresponding design component.

## Week 10 - Unit Test Planning

- Identify all important functions, methods, modules, and workflows that require testing.
- Prepare unit test cases for:
  - Creating, editing, deleting, tagging, and categorizing notes
  - Creating and completing to-do tasks
  - Searching by title, content, and tag
  - Creating and triggering reminders
  - Creating note links and displaying graph relationships
  - Registering, logging in, logging out, and managing user profiles
  - Synchronizing, backing up, and recovering data
- Define expected inputs, outputs, preconditions, test steps, expected results, and pass/fail criteria.

## Week 11 - Test Review and Document Integration

- Review test cases for completeness, clarity, and traceability to the SRS.
- Check that every functional requirement has related design details and test cases.
- Integrate the SRS, software design, and unit test plan into a consistent technical documentation set.
- Conduct internal team review and revise unclear, duplicated, or inconsistent sections.
- Update documents based on instructor feedback if available.

## Week 12 - Final Review and Risk Buffer

- Use this week as a buffer for delayed tasks, additional feedback, document polishing, and consistency checks.
- Resolve remaining gaps in diagrams, requirements, design descriptions, and test cases.
- Confirm that the documentation clearly explains the product scope, technical solution, user workflows, and testing strategy.
- Prepare the final submission package.

## Week 13 - Submission and Presentation

- Submit the final project documents to the instructor.
- Prepare and deliver the project presentation.
- Present the application concept, core features, architecture, database design, UI flow, and test strategy.
- Summarize team contributions, lessons learned, and possible future improvements such as real-time collaboration or expanded synchronization features.

## Supporting Tools

- Version control and document history: GitHub, Git, shared Google Drive folders.
- Diagramming: Draw.io and PlantUML.
- Collaborative documentation: Google Docs and Markdown files.
- Test case management: Excel or Google Sheets.
- Project tracking: Slack, Github.

