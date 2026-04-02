# Istruzioni per la Consegna

## 1. Rinominare tutti i file

Sostituisci `XXXXXX` con il tuo numero di matricola (es. `123456`) in TUTTI questi punti:

| File/Cartella da rinominare | → Nuovo nome |
|---|---|
| `solver_XXXXXX/` (cartella) | `solver_123456/` |
| `solver_XXXXXX/solver_XXXXXX.py` | `solver_123456/solver_123456.py` |
| `solver_XXXXXX/__init__.py` | (stesso nome, ma modifica il contenuto) |

## 2. Modificare il contenuto dei file

### `solver_123456.py`
- Cerca e sostituisci TUTTE le occorrenze di `XXXXXX` → `123456`
  - `class solver_XXXXXX` → `class solver_123456`
  - `self.name = 'solver_XXXXXX'` → `self.name = 'solver_123456'`
  - Importazioni in cima al file

### `__init__.py`
```python
from .abstract_solver import AbstractSolver
from .solver_123456 import solver_123456

__all__ = ['AbstractSolver', 'solver_123456']
```

## 3. Aggiornare `main.py`
```python
from solver_123456 import solver_123456
...
solver = solver_123456(inst)
```

## 4. Aggiornare `results_checker.py`
```python
solver_name = 'solver_123456'
```

## 5. Struttura della cartella da comprimere

```
assignment_123456.zip
└── solver_123456/
    ├── solver_123456.py      ← SOLVER PRINCIPALE
    ├── __init__.py
    ├── abstract_solver.py
    ├── additional_script.py
    └── requirements.txt
```

## 6. Test prima della consegna

```bash
python main.py                 # deve terminare senza errori
python results_checker.py      # deve stampare ✅ FEASIBLE solution
```

## Dipendenze (requirements.txt)
```
pandas==2.2.2
scipy>=1.7.0
numpy>=1.21.0
```
Tutte open-source, installabili via pip, nessuna licenza proprietaria.
