# # fix_training.py - Auto-fix training.py for Flax 0.9+
# import re
# from pathlib import Path

# training_file = Path(r'D:\Acer\Code\Dat_pr\hypernerf\hypernerf\training.py')

# print("Fixing training.py for Flax 0.9+ compatibility...")

# # Read file
# with open(training_file, 'r', encoding='utf-8') as f:
#     content = f.read()

# # Backup
# backup_file = training_file.with_suffix('.py.bak')
# with open(backup_file, 'w', encoding='utf-8') as f:
#     f.write(content)
# print(f"✓ Backup created: {backup_file}")

# # Fix 1: Remove flax.optim type annotations
# content = re.sub(
#     r'state:\s*flax\.optim\.OptimizerState',
#     'state',
#     content
# )

# # Fix 2: Replace flax.optim references with generic types
# content = re.sub(
#     r'flax\.optim\.',
#     '',
#     content
# )

# # Write fixed version
# with open(training_file, 'w', encoding='utf-8') as f:
#     f.write(content)

# print(f"✓ Fixed: {training_file}")
# print("\nChanges:")
# print("  - Removed flax.optim type annotations")
# print("  - Made compatible with Flax 0.9+")

# fix_schedules.py
from pathlib import Path
import re

schedules_file = Path(r'D:\Acer\Code\Dat_pr\hypernerf\hypernerf\schedules.py')

print("Fixing schedules.py for Python 3.12...")

# Read
with open(schedules_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Backup
backup = schedules_file.with_suffix('.py.bak')
with open(backup, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"✓ Backup: {backup}")

# Fix 1: Add import at top (after other imports)
if 'from collections.abc import' not in content:
    # Find first import line
    import_match = re.search(r'^import collections', content, re.MULTILINE)
    if import_match:
        # Add after import collections
        insert_pos = import_match.end()
        new_import = '\ntry:\n    from collections.abc import Mapping\nexcept ImportError:\n    from collections import Mapping'
        content = content[:insert_pos] + new_import + content[insert_pos:]
        print("✓ Added collections.abc import")

# Fix 2: Replace collections.Mapping with Mapping
content = re.sub(
    r'\bcollections\.Mapping\b',
    'Mapping',
    content
)
print("✓ Replaced collections.Mapping with Mapping")

# Write
with open(schedules_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✓ Fixed: {schedules_file}")
