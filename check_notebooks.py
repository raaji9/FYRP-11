import json, ast, glob, sys, traceback

notebooks = glob.glob('**/*.ipynb', recursive=True)
print(f"Found {len(notebooks)} notebooks\n")

errors = []
for nb_path in sorted(notebooks):
    # Skip obvious binary/cache files
    if 'ipynb_checkpoints' in nb_path:
        continue
    
    print(f"Checking: {nb_path}")
    
    # 1) Validate JSON
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        msg = f"  JSON ERROR: {e}"
        print(msg)
        errors.append((nb_path, msg))
        continue
    except Exception as e:
        msg = f"  FILE READ ERROR: {e}"
        print(msg)
        errors.append((nb_path, msg))
        continue
    
    # 2) Check notebook structure
    if 'cells' not in nb:
        msg = "  STRUCTURE ERROR: no 'cells' key"
        print(msg)
        errors.append((nb_path, msg))
        continue
    
    # 3) Check each code cell for syntax errors
    cell_errors = 0
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', [])
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = source or ''
        
        if not code.strip():
            continue
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            # Show first line of the error
            lines = code.split('\n')
            line_no = e.lineno if e.lineno else 1
            context = lines[min(line_no-1, len(lines)-1)].strip() if lines else ''
            msg = f"  SYNTAX ERROR in cell {i} (line {line_no}): {e.msg} | context: ...{context[:80]}..."
            print(msg)
            errors.append((nb_path, msg))
            cell_errors += 1
    
    if cell_errors == 0:
        print(f"  ✅ OK ({len(nb['cells'])} cells)")

print("\n" + "=" * 60)
print(f"Total: {len(notebooks)} notebooks checked")
print(f"Errors found: {len(errors)}")
if errors:
    print("\nError Summary:")
    for nb_path, msg in errors:
        print(f"  {nb_path}: {msg}")
        sys.exit(1)  # Exit with error code if any issues found
else:
    print("\nAll notebooks passed syntax check! ✅")