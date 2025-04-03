# Protocollo per adattare la libreria wavelet a immagini fino a 10 bande

  Ecco il piano di adattamento strutturato in fasi:

  1. Modifiche alla configurazione di base

  - Aggiornare Config in base.py per supportare dinamicamente
   fino a 10 bande
  - Adattare la formula di calcolo dei coefficienti di
  scattering

  2. Modifiche ai processori di immagini

  - Aggiornare ImageProcessor in processors.py per:
    - Rimuovere la conversione forzata a RGB
    - Supportare normalizzazione parametrica per numero
  variabile di canali
    - Adattare il metodo process_image per gestire immagini
  multibanda

  3. Modifiche ai dataset

  - Aggiornare datasets.py per:
    - Modificare get_default_transform per supportare
  normalizzazione multibanda
    - Rendere compute_dataset_statistics dinamico in base al
  numero di canali
    - Rimuovere la conversione RGB da __getitem__

  4. Modifiche ai modelli

  - Aggiornare models.py per:
    - Rendere il reshape in forward() flessibile per
  qualsiasi numero di canali
    - Adattare i parametri di inizializzazione per gestire
  input multibanda
    - Modificare create_model per supportare configurazioni
  con diversi canali

  5. Aggiornamento degli strumenti di visualizzazione

  - Estendere gli strumenti di visualizzazione per supportare
   fino a 10 bande
  - Implementare nuove modalità di visualizzazione specifiche
   per dati multibanda

  6. Documentazione

  - Creare documentazione su come utilizzare la libreria con
  immagini multibanda
  - Fornire esempi di utilizzo con immagini a vari numeri di
  bande

# PROMPTS DI IMPLEMENTAZIONE

  Per implementare queste modifiche, potrai chiedermi di
  eseguire i seguenti prompt:

  1. "Modifica il file wavelet_lib/base.py per supportare
  immagini multibanda fino a 10 canali, incluso l'adattamento
   della formula dei coefficienti di scattering"
  2. "Aggiorna wavelet_lib/processors.py per eliminare la
  conversione forzata a RGB e supportare la normalizzazione
  dinamica in base al numero di canali"
  3. "Modifica wavelet_lib/datasets.py per supportare il
  caricamento e l'elaborazione di immagini con numero di
  canali variabile"
  4. "Adatta wavelet_lib/models.py per gestire il reshape
  dinamico in base al numero di canali di input"
  5. "Crea un file MULTIBAND_USAGE.md con documentazione
  dettagliata sull'utilizzo della libreria con immagini
  multibanda e esempi di codice"
  6. "Implementa un nuovo script di test
  script/tests/test_multiband.py per verificare il corretto
  funzionamento con immagini multibanda"

# SPECIFICHE TECNICHE

  Per supportare 10 bande, consiglio queste specifiche
  precise:

  1. In Config:
    - Modificare il default di num_channels=3 a supportare
  fino a 10
    - La formula per scattering_coeffs deve essere adattata
  per scale proporzionalmente
  2. Normalizzazione:
    - Sostituire valori fissi [0.5, 0.5, 0.5] con arrays
  dinamici in base a num_channels
    - Implementare calcolo automatico delle statistiche per
  ogni canale
  3. Caricamento immagini:
    - Utilizzare rasterio anziché PIL per supportare
  nativamente immagini multibanda
    - Mantenere retrocompatibilità con immagini RGB
  4. Visualizzazione:
    - Supportare la visualizzazione selettiva di canali o
  combinazioni
    - Implementare mappature colore appropriate per dati
  multispettrali