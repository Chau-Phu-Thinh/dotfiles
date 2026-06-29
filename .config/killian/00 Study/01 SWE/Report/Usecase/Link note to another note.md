---
# Link Note to Another Note

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
