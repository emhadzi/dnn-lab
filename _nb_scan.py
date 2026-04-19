import json
nb = json.load(open(r'c:/Users/ManosChatzigeorgiou/Documents/ntua/dnn-lab/DL_LabProject_25_26.ipynb', encoding='utf-8'))
print('nbformat:', nb.get('nbformat'), nb.get('nbformat_minor'))
for i in [34, 36, 37, 40, 41, 42]:
    cell = nb['cells'][i]
    print(f'Cell {i} keys: {list(cell.keys())}')
    print(f'  id: {cell.get("id")}')
