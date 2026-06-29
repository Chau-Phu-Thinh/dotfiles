# Link Note to another note

| Use case name: Link Note to Another Note | ID: UC-05                  |
| ---------------------------------------- | -------------------------- |
| **Primary actor:** User                  | **Priority:** Mandatory    |
| **Secondary actor:** System              | **Classification:** Simple |

**Brief description:** The user links a note to another note using a wiki-link (`[[note name]]`), navigates to the linked note, and can return to the previously opened note using the back-navigation control.
**Associated use cases:** (Save note locally)
**Preconditions:** The user is on an existing note that is open in the editor.
**Postconditions:** The user is currently located on the newly navigated note, and the origin note has been automatically saved to the local database.
**Basic flow of events:**
1. The user types `[[note name]]` in the current note body.
2. The system renders the text as a clickable link (blue for an existing note, red for a non-existing note).
3. The user clicks the rendered link.
4. The system navigates to the linked note. If the note does not exist, the system creates a blank note with that name.
5. The system automatically saves the origin note to the local database (SQLite/Room) using an offline-first mechanism.
6. The system updates the navigation history stack and enables the back (`←`) navigation control.
**Alternative and exception flows:**
- **AF-1 (Navigate back):** At step 6, the user clicks the `←` button on the interface. The system navigates back to the last visited note and automatically saves the current note to the local database.
- **AF-2 (Navigate forward):** After navigating back, if the user clicks the `→` button on the interface, the system navigates forward to the next note in the navigation history stack.
- **AF-3 (Link to a non-existing note):** At step 1, if the user types `[[newnote]]` and the referenced note does not exist, the system highlights the link in red. At step 3, when the user clicks it, the system creates a new blank note with that name and navigates to it.
- **EF-1 (Memory full error):** If the system cannot write the note data due to full device memory, the system displays the message "Out of memory" on the interface and keeps the user in the editor to prevent data loss.

**Functional Requirements:**

- **FR-01:** The system must support inline wiki-link syntax (`[[note name]]`) to create links between notes during the editing process.
- **FR-02:** The system must visually distinguish links to existing notes (blue) from links to non-existing notes (red) in real time.
- **FR-03:** The system must automatically create a blank note when the user navigates to a non-existing linked note.
- **FR-04:** The system must implement an automatic local storage mechanism (SQLite/Room) using an offline-first architecture to save the origin note upon every navigation event.
- **FR-05:** The system must maintain a navigation history stack and provide back (`←`) and forward (`→`) controls to allow the user to traverse visited notes.

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

---
