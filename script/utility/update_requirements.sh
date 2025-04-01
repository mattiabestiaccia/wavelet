#!/bin/bash

# Attiva l'ambiente virtuale
source wavelet_venv/bin/activate

# Aggiorna base.txt con le dipendenze attuali
pip freeze > requirements/base.txt

# Rimuovi le dipendenze di sviluppo dal base.txt
grep -v -E "black|isort|pytest|ipython|jupyter" requirements/base.txt > requirements/base.txt.tmp
mv requirements/base.txt.tmp requirements/base.txt

# Aggiorna dev.txt
echo "-r base.txt" > requirements/dev.txt
pip freeze | grep -E "black|isort|pytest|ipython|jupyter" >> requirements/dev.txt

echo "Requirements files updated successfully!"