# NoteGraph - Android smart note-taking app

## Overview

NoteGraph is planned as an offline lightweight  Android note-taking application built with React Native and TypeScript. The stack should support fast mobile interaction, structured note editing, local persistence, note relationship visualization, reminders, authentication, and later synchronization with a backend service.

## Core idea

- Each note is a knowledge unit
- Links between notes create relationship
- Graph view help users discover clusters, relation topics and navigation paths

## Primary goal

- Provide a fast and clean mobile 

## Mobile Application

- Framework: React Native
- Language: TypeScript
- Target platform: Android first
- Navigation: React Navigation
- State management: lightweight React state for local UI state; a structured store such as Zustand or Redux Toolkit if shared application state becomes complex
- Forms and validation: React Hook Form with schema validation if needed
- Local notifications: React Native notification library for reminders and scheduled alerts

## Local Data Layer

- Local database: SQLite
- Storage strategy: offline-first local persistence for notes, tasks, tags, folders, reminders, links, and synchronization metadata
- Main purpose: allow users to create, edit, delete, search, and browse notes without requiring an active internet connection

## Backend

- Framework: Spring Boot 3.x
- Language: Kotlin
- API style: REST API first
- Data access: Spring Data JPA
- Security: Spring Security
- Authentication: JWT-based authentication
- Real-time features: WebSocket only in a later phase if real-time collaboration becomes part of the scope

## Server Database

- Primary project option: PostgreSQL
- Main entities: users, notes, folders, tags, tasks, reminders, note links, synchronization records, and backup records

## Core Technical Responsibilities

- Note management: create, edit, delete, tag, categorize, and persist notes.
- To-do management: create task lists, mark tasks as completed, set priorities, and manage deadlines.
- Search: search notes by title, content, and tag.
- Graph view: represent note relationships through backlinks and display connected knowledge nodes.
- Reminder and notification: schedule reminders and notify users at the correct time.
- Synchronization: back up and synchronize local data with the server in later project phases.
- User account management: register, login, logout, and manage basic profile information.

## Development and Documentation Tools

- Version control: GitHub, Git
- Diagrams: Draw.io and PlantUML
- Documentation: Markdown and Google Docs
- Test case management: 
- Project tracking: Slack

## Stack Rationale

React Native with TypeScript is suitable for building a maintainable mobile application with strong typing and reusable UI components. SQLite supports the offline-first requirement and gives users fast access to notes, tasks, and graph data. Spring Boot with Kotlin provides a structured backend foundation for authentication, synchronization, backup, and future expansion. A REST-first backend keeps the system simple for the current academic scope, while WebSocket support can be reserved for future real-time collaboration.
