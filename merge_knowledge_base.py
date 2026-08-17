"""
merge_knowledge_base.py
------------------------
Safely merges disease_database_additions.json (33 new entries) into
your existing knowledge_base/disease_database.json (5 entries),
without deleting or overwriting anything that's already there.

Usage (from project root, D:\\AgriVision-AI):
    python merge_knowledge_base.py
"""

import json
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MAIN_DB_PATH = os.path.join(BASE_DIR, "knowledge_base", "disease_database.json")
ADDITIONS_PATH = os.path.join(BASE_DIR, "disease_database_additions.json")
BACKUP_PATH = os.path.join(BASE_DIR, "knowledge_base", "disease_database.backup.json")

if not os.path.exists(MAIN_DB_PATH):
    print(f"ERROR: Could not find {MAIN_DB_PATH}")
    exit(1)

if not os.path.exists(ADDITIONS_PATH):
    print(f"ERROR: Could not find {ADDITIONS_PATH}")
    print("Make sure disease_database_additions.json is in the project root.")
    exit(1)

# 1. Back up the existing database first, just in case
shutil.copy(MAIN_DB_PATH, BACKUP_PATH)
print(f"Backup saved -> {BACKUP_PATH}")

# 2. Load both files
with open(MAIN_DB_PATH, "r") as f:
    main_db = json.load(f)

with open(ADDITIONS_PATH, "r") as f:
    additions = json.load(f)

# 3. Merge: additions are only added if the key doesn't already exist,
#    so your original 5 entries are never overwritten
added_count = 0
skipped_count = 0
for key, value in additions.items():
    if key not in main_db:
        main_db[key] = value
        added_count += 1
    else:
        skipped_count += 1

# 4. Save the merged result back
with open(MAIN_DB_PATH, "w") as f:
    json.dump(main_db, f, indent=4)

print(f"\nMerge complete.")
print(f"Added     : {added_count} new entries")
print(f"Skipped   : {skipped_count} entries (already existed)")
print(f"Total now : {len(main_db)} entries in disease_database.json")