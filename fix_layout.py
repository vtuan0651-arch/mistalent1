import os

with open("g:/OneDrive - Đại học Thương mại/Máy tính/MIS/app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
in_input_col = False
in_workflow = False

i = 0
while i < len(lines):
    line = lines[i]
    
    if line.startswith('col_input, col_result = st.columns([1.0, 2.2], gap="large")'):
        new_lines.append('result = st.session_state.get("opc_result")\n\n')
        new_lines.append('tab_ops, tab_dashboard = st.tabs(["⚙️ Operations (Input & Workflow)", "🏆 Full-Page Decision Dashboard"])\n\n')
        new_lines.append('with tab_ops:\n')
        new_lines.append('    col_input, col_workflow = st.columns([1.0, 2.2], gap="large")\n')
        i += 1
        continue
        
    if line.startswith('with col_input:'):
        new_lines.append('    with col_input:\n')
        in_input_col = True
        i += 1
        continue
        
    if line.startswith('result = st.session_state.get("opc_result")'):
        # Skip this as we moved it up
        i += 1
        continue
        
    if line.startswith('with col_result:'):
        # We replace this and the following line (which is the old tabs definition)
        i += 2 # skip "with col_result:" and the tabs def
        continue
        
    if line.startswith('with tab_workflow:'):
        new_lines.append('    with col_workflow:\n')
        in_workflow = True
        i += 1
        continue
        
    if line.startswith('with tab_decision:'):
        in_input_col = False
        in_workflow = False
        new_lines.append('with tab_dashboard:\n')
        i += 1
        continue
        
    if in_input_col or in_workflow:
        if line.strip() == "":
            new_lines.append("\n")
        else:
            new_lines.append("    " + line)
    else:
        new_lines.append(line)
        
    i += 1

with open("g:/OneDrive - Đại học Thương mại/Máy tính/MIS/app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
    
print("Layout fixed!")
