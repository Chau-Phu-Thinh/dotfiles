---
# Save note locally

| Use case name: Save note locally | ID: UC-06                  |
| -------------------------------- | -------------------------- |
| **Primary actor:** System        | **Priority:** Mandatory    |
| **Secondary actor:** None        | **Classification:** Simple |
|                                  |                            |

**Brief description:** System automatically save current note after 10s, and notify save status on interface
**Associated use cases:** Create note, Edit note, Move note to file and link note to another
**Preconditions:** The user has an existing note open in the editor and has made at least one change to the note content.
**Postconditions:** The modified note content is persisted to the local database (SQLite/Room). The save status indicator on the interface reflects the result of the save operation.
**Basic flow of events:**
1. The user opens an existing note and begins editing its content.
2. The system detects a change in the note content and starts a 10-second inactivity timer.
3. The 10-second timer expires with no further input from the user.
4. The system saves the current note content to the local database (SQLite/Room) using an offline-first mechanism.
5. The system updates the save status indicator on the interface to "Saved".
**Alternative and exception flows:**
- **AF-1 (User continues typing):** At step 3, if the user resumes typing before the timer expires, the system resets the 10-second inactivity timer and returns to step 2.
- **AF-2 (Manual save):** At any point during editing, if the user triggers a manual save (e.g., pressing the save button or a keyboard shortcut), the system immediately performs step 4 and step 5 without waiting for the timer.
- **AF-3 (User closes the note):** If the user navigates away from or closes the note before the timer expires, the system immediately saves the note content to the local database before closing, then updates the save status indicator.
- **EF-1 (Storage write failure):** At step 4, if the system cannot write to the local database due to insufficient device storage, the system aborts the save operation, displays an "Out of memory" error message on the interface, and retains the unsaved content in the editor to prevent data loss.
- **EF-2 (Database error):** At step 4, if a database error occurs (e.g., corruption or lock conflict), the system displays a "Save failed. Please try again." message on the interface and keeps the unsaved content in the editor.

**Functional Requirements:**

- **FR-01:** The system must automatically trigger a save operation after 10 seconds of user inactivity following any content change in the currently open note.
- **FR-02:** The system must reset the inactivity timer each time the user resumes editing before the timer expires.
- **FR-03:** The system must persist the note content to a local database (SQLite/Room) using an offline-first architecture, requiring no network connection to complete a save.
- **FR-04:** The system must display a real-time save status indicator (e.g., "Saving…" and "Saved") on the interface to inform the user of the current save state.
- **FR-05:** The system must immediately save the note content when the user navigates away from or closes the note, regardless of whether the inactivity timer has expired.
- **FR-06:** The system must display a descriptive error message on the interface and preserve the unsaved content in the editor if a save operation fails due to storage or database errors.

