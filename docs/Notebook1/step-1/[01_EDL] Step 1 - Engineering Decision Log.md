# \[01\_EDL\] Step 1 — Engineering Decision Log

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Fase:** Step 1 — Setup dell'Ambiente, Configurazione Centralizzata, Determinismo e Ispezione Dati Grezzi (COCO 2017)  
**Scopo del Documento:** Tracciamento formale delle decisioni progettuali adottate nella Cella 1, delle opzioni scartate e delle implicazioni architetturali derivate dall'ispezione dei dati grezzi.

---

## EDL-01 — Adozione di COCO 2017 (*Thing*, $C = 81$) con Integrazione di `pycocotools`

* **Stato:** Approvata e Implementata
* **Contesto e Problema:**  
  Le specifiche progettuali (`H3.pdf`) menzionano COCO 2017 (*Stuff*) e lo scheletro del Kaggle Notebook presuppone maschere semantiche `.png` pre-renderizzate in cartelle parallele (`train_labels`). L'ispezione della directory montata su Kaggle (`/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017`) ha invece accertato la presenza di **COCO 2017 (*Thing*)** privo di maschere `.png` pre-renderizzate (`Cartelle maschere semantiche pre-renderizzate: NESSUNA`), con annotazioni poligonali/RLE nei file `instances_train2017.json` e `instances_val2017.json`.
* **Alternative Valutate:**
  * *Opzione A:* Aggiungere un secondo dataset esterno (`stuffthingmaps_trainval2017`) per ottenere maschere `.png` *Stuff*.
  * *Opzione B (Selezionata):* Operare direttamente sul dataset **COCO 2017 (*Thing*)** già montato, formulando il task su **$C = 81$ classi** ($0 = \text{background}$ e $1 \dots 80 = \text{categorie Thing}$) e integrando esplicitamente la libreria `pycocotools` per decodificare e rasterizzare le annotazioni JSON.
* **Decisione Adottata:**  
  Adottare **COCO 2017 (*Thing*)** con $C = 81$ e integrare `pycocotools` (previa approvazione esplicita in conformità ai vincoli metodologici).
* **Razionale Tecnico:**  
  Evita dipendenze da mirror esterni non verificati, mantiene l'intero progetto auto-contenuto sul dataset ufficiale COCO 2017 montato su Kaggle e consente di gestire in modo deterministico sia l'ordinamento per area delle istanze sovrapposte sia il campionamento bilanciato per classe tramite gli indici JSON.

---

## EDL-02 — Centralizzazione della Configurazione, Persistenza Artefatti e Flag Modulari (`Cella 1`)

* **Stato:** Approvata e Implementata (Refactoring Cella 1)
* **Contesto e Problema:**  
  In un notebook complesso destinato a eseguire esplorazione statistica, *babysitting*, mini-ablation diagnostica e confronti multi-loss, disperdere percorsi e parametri nelle singole celle compromette la manutenibilità e costringe a rieseguire blocchi diagnostici lenti a ogni riavvio di sessione.
* **Decisione Adottata:**  
  Concentrare nella **Cella 1** tutti gli import, le costanti globali (`SEED`, `CROP_SIZE`, `NUM_CLASSES`, `IGNORE_INDEX`, `AUX_LOSS_WEIGHT`, `IMAGENET_MEAN`, `IMAGENET_STD`, `NUM_WORKERS`, `TRAIN_MAX_SAMPLES`, `CACHE_IN_RAM`), la creazione automatica delle directory `/kaggle/working/figures` e `/kaggle/working/checkpoints`, l'helper `save_and_show_fig` (esportazione a $300\text{ DPI}$ PNG + opzionale SVG con `plt.close(fig)`) e i flag booleani `RUN_EDA_PLOTS` e `RUN_BABYSITTING`.
* **Razionale Tecnico:**  
  1. Disaccoppia la configurazione sperimentale dall'implementazione delle classi e dei loop.
  2. Garantisce la persistenza su disco di tutti i grafici e dei checkpoint (`shared_init_fcn_resnet50.pt`, pesi migliori) per la documentazione e per i notebook successivi.
  3. Permette di saltare instantaneamente i blocchi diagnostici già validati (`RUN_EDA_PLOTS=False`, `RUN_BABYSITTING=False`) prima di avviare addestramenti lunghi.

---

## EDL-03 — Determinismo Totale (`SEED = 42` e Backend `cuDNN`)

* **Stato:** Approvata e Implementata
* **Contesto e Problema:**  
  L'addestramento *from scratch* di `FCN-ResNet50`, lo shuffling dei mini-batch e i ritagli casuali introducono sorgenti multiple di stocasticità che potrebbero mascherare le differenze reali tra diverse funzioni di perdita nello Step 4.
* **Decisione Adottata:**  
  Implementare `set_global_seed(SEED=42)` fissando il seme su `PYTHONHASHSEED`, `random`, `numpy`, `torch`, `torch.cuda.manual_seed_all` e imponendo `torch.backends.cudnn.deterministic = True` e `torch.backends.cudnn.benchmark = False`.
* **Razionale Tecnico:**  
  Assicura l'esatta riproducibilità delle inizializzazioni casuali dei pesi, del campionamento dei sottoinsiemi e della sequenza dei batch, soddisfacendo il requisito metodologico *ceteris paribus*.

---

## EDL-04 — Strategia Ibrida Memoria/Disco al Posto del Precaricamento Integrale in RAM (`preload()`)

* **Stato:** Approvata e Implementata
* **Contesto e Problema:**  
  Lo scheletro del corso carica l'intero dataset in RAM prima del training (`preload()`). Su `train2017` ($118\,287$ immagini), precaricare tutti i tensori saturerebbe immediatamente i $13\text{--}30\text{ GB}$ di RAM delle istanze Kaggle provocando un crash *Out-Of-Memory* (OOM).
* **Decisione Adottata:**  
  Adottare una strategia ibrida parametrica controllata da `TRAIN_MAX_SAMPLES` e `CACHE_IN_RAM`:
  * **Caricamento *on-the-fly* (`CACHE_IN_RAM = False`)** quando si utilizza l'intero dataset (`TRAIN_MAX_SAMPLES = None`, $118\,287$ immagini).
  * **Precaricamento selettivo in RAM (`CACHE_IN_RAM = True`)** per sottoinsiemi contenuti (`TRAIN_MAX_SAMPLES <= 15000`, es. $16$ immagini nel *babysitting* e $12\,000$ immagini nello studio comparativo multi-loss).
* **Razionale Tecnico:**  
  Elimina ogni rischio di OOM sul dataset completo massimizzando al contempo il throughput I/O durante gli esperimenti comparativi ripetitivi.

---

## EDL-05 — Vincoli di Preprocessing Derivati dall'Ispezione Grezza (`RGB` e Dimensioni $< 256\text{ px}$)

* **Stato:** Approvata e Trasferita allo Step 2
* **Contesto e Problema:**  
  La profilazione su $500$ immagini di `val2017` nella Sezione 5 ha rivelato due anomalie strutturali:
  1. Presenza di immagini in scala di grigi a singolo canale (`PIL mode == 'L'`: $2$ immagini su $500$, pari allo $0.40\%$).
  2. Presenza di immagini con altezza o larghezza inferiore a `CROP_SIZE = 256` ($H_{\min} = 230\text{ px}$, $W_{\min} = 246\text{ px}$).
* **Decisione Adottata:**  
  1. Imporre `.convert("RGB")` su ogni immagine letta da disco prima della conversione in tensore.
  2. Scartare il `RandomCrop(256, 256)` diretto sull'immagine nativa (che solleverebbe `ValueError` quando $H < 256$ o $W < 256$) a favore del **ridimensionamento proporzionale del lato corto a $256\text{ px}$** seguito dal ritaglio $256 \times 256$ (Opzione C nello Step 2).
* **Razionale Tecnico:**  
  Previene crash silenziosi di *shape mismatch* (`1` vs `3` canali) nel `DataLoader` e garantisce che ogni campione abbia lato minimo $\ge 256\text{ px}$ senza introdurre bordi artificiali (`IGNORE_INDEX = 255`) né distorcere l'*aspect ratio* originale.