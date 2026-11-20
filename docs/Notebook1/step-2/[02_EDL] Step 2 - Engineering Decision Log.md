# \[02\_EDL\] Step 2 — Engineering Decision Log

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Fase:** Step 2 — Parsing Annotazioni JSON (`pycocotools`), Pipeline `COCOSemanticDataset`, Campionamento Bilanciato e `DataLoader`  
**Scopo del Documento:** Tracciamento formale delle decisioni progettuali e algoritmiche adottate nelle Celle 2, 3 e 4, delle alternative scartate e dei relativi razionali tecnici.

---

## EDL-06 — Rasterizzazione con Ordinamento per Area Decrescente (*Painter's Algorithm*)

* **Stato:** Approvata e Implementata
* **Contesto e Problema:**  
  In COCO 2017 (*Thing*), le annotazioni di una stessa immagine sono memorizzate come istanze poligonali/RLE indipendenti che possono sovrapporsi spazialmente (ad esempio una `person` seduta su una `chair` davanti a un `dining table`, oppure un `dog` davanti a una `person`). Una rasterizzazione nell'ordine arbitrario del JSON permetterebbe a un'istanza grande caricata successivamente di sovrascrivere e cancellare un oggetto più piccolo in primo piano.
* **Decisione Adottata:**  
  Ordinare le annotazioni di ciascuna immagine per area decrescente (`sorted(anns, key=lambda a: a["area"], reverse=True)`) prima di scrivere gli indici di classe sulla maschera `uint8`.
* **Razionale Tecnico:**  
  Disegnando prima le superfici più estese e successivamente gli oggetti più piccoli, questi ultimi restano visibili in primo piano nella maschera semantica. L'efficacia della scelta è direttamente verificabile nella Riga 3 di `01_step2_dataloader_sanity_overlay.png`, dove `dog` (giallo ocra) resta nitidamente sovrapposto a `person` (viola).

---

## EDL-07 — Strategia Geometrica **Opzione C** (Resize Proporzionale Lato Corto + Crop Sincronizzato)

* **Stato:** Approvata e Implementata
* **Contesto e Problema:**  
  COCO 2017 contiene immagini a risoluzioni variabili con $822$ campioni in `train2017` ($0.69\%$) aventi lato corto $< 256\text{ px}$. Occorreva uniformare tutti i campioni a $256 \times 256\text{ px}$ senza sollevare errori di dimensione, senza distorcere la geometria degli oggetti e senza introdurre zone artificiali di `IGNORE_INDEX = 255`.
* **Alternative Valutate:**
  * *Opzione A (`RandomCrop(256)` diretto senza resize):* Fallisce con `ValueError` su $822$ immagini di training e $42$ di validation e soffre di "effetto buco della serratura" sugli oggetti estesi.
  * *Opzione B (`Resize((256, 256))` anisotropo):* Schiaccia l'*aspect ratio* ($640 \times 426 \rightarrow 256 \times 256$), alterando la forma naturale di persone, veicoli e animali.
  * *Opzione C (Selezionata):* `TF.resize(..., size=256)` sul lato corto (preservando l'*aspect ratio* e portando entrambi i lati $\ge 256\text{ px}$) seguito da `RandomCrop(256, 256)` + `TF.hflip` sincronizzati in training (`is_train=True`) e `TF.center_crop(256, 256)` deterministico in validation (`is_train=False`).
* **Decisione Adottata:**  
  Implementare l'**Opzione C** utilizzando `InterpolationMode.BILINEAR` (`antialias=True`) per le immagini RGB e `InterpolationMode.NEAREST` per le maschere categoriche.
* **Razionale Tecnico:**  
  Preserva le proporzioni reali degli oggetti, evita qualunque bordo fittizio e impedisce all'interpolazione di generare indici di classe intermedi fittizi sulle maschere discrete. Inoltre, l'uso di `torch.rand(1).item() < 0.5` per il flip mantiene l'intera stocasticità governata dal `torch.Generator` del `DataLoader`.

---

## EDL-08 — Granularità del Campionamento Bilanciato (*Image-Level Presence* vs *Instance/Pixel Count*)

* **Stato:** Approvata, Implementata e Documentata in `[02_QA]`
* **Contesto e Problema:**  
  Per accelerare lo studio comparativo multi-loss dello Step 4 tramite un sottoinsieme di $N = 12\,000$ immagini senza perdere le classi rare di COCO (`hair drier`: $189$ img, `toaster`: $217$ img; squilibrio $339.2\times$), occorreva stabilire su quale unità di conteggio uniformare il campionamento.
* **Alternative Valutate:**
  * *Campionamento casuale uniforme:* Lascerebbe a `hair drier` appena $\approx 19$ immagini su $12\,000$.
  * *Bilanciamento sul numero totale di istanze ($N_{\text{inst}}$):* Concentrerebbe il subset su pochissime scene affollate (es. $10$ foto di un negozio con $15$ tostapane ciascuna), inducendo *overfitting* sullo sfondo.
  * *Bilanciamento sulla presenza per immagine distinta ($N_{\text{img}}$, Selezionata):* Algoritmo *greedy* ordinato dalle classi più rare alle più frequenti con quota minima $\lfloor N / 80 \rfloor = 150$ immagini per classe.
* **Decisione Adottata:**  
  Adottare il campionamento *greedy* basato sulla **presenza a livello di immagine** (`select_class_balanced_subset`), comprimendo il coefficiente di Gini da $0.442$ a $0.365$ e garantendo almeno $152$ immagini per ciascuna delle $80$ classi.
* **Razionale Tecnico:**  
  Massimizza la diversità di scene, sfondi e illuminazioni per le categorie rare (poiché l'unità caricata dal `DataLoader` è l'intera immagine) e delega allo **Step 4** il ribilanciamento vero e proprio a livello di **pixel** tramite le funzioni di perdita focali e regionali (`Focal`, `Dice`, `Jaccard`), come formalizzato in `Step02_QA_Granularita_Campionamento.md`.

---

## EDL-09 — Preservazione Integrale del Validation Set (`val2017`, $5\,000$ Immagini)

* **Stato:** Approvata e Implementata
* **Contesto e Problema:**  
  È stato valutato se applicare il medesimo algoritmo di bilanciamento anche a `val2017` durante la validazione.
* **Decisione Adottata:**  
  Mantenere **`val2017` rigorosamente intatto** (`5000` immagini, `Gini = 0.446`, `CenterCrop(256, 256)` deterministico).
* **Razionale Tecnico:**  
  Altera le frequenze delle classi nel validation set introdurrebbe *Prior Probability Shift*, falsando la `mIoU` e la `Pixel Accuracy` rispetto alla distribuzione naturale dei dati.

---

## EDL-10 — Visualizzazione Analitica: Scala Logaritmica, Curva di Lorenz e *Cleveland Dumbbell Chart*

* **Stato:** Approvata e Implementata (`Cella 3`)
* **Contesto e Problema:**  
  Un istogramma lineare standard a barre affiancate su $80$ categorie veniva schiacciato dall'outlier `person` ($54.2\%$), comprimendo $76$ classi su $80$ in una fascia stretta tra $0\%$ e $10\%$.
* **Decisione Adottata:**  
  Sostituire i grafici lineari con due visualizzazioni complementari esportate a $300\text{ DPI}$ PNG e SVG usando esclusivamente `matplotlib` e `numpy`:
  1. `02_step2_coco_uniformity_lorenz_gini`: distribuzione ordinata in scala logaritmica + Curva di Lorenz con indice di Gini.
  2. `03_step2_coco_class_distribution_dumbbell`: *Cleveland Dumbbell Chart* in scala logaritmica affiancato dal pannello del moltiplicatore `Subset % / Full %`.
* **Razionale Tecnico:**  
  Rende leggibili simultaneamente sia le *head classes* sia le *tail classes* e quantifica in modo sintetico l'effetto del campionamento *greedy* tramite un unico scalare economico-statistico ($G = 0.442 \rightarrow 0.365$).

---

## EDL-11 — Istanziazione dei `DataLoader` Post-Calibrazione GPU (`Cella 4`) e Controllo Parametro `TRAIN_MAX_SAMPLES`

* **Stato:** Approvata e Implementata
* **Contesto e Problema:**  
  Nel refactoring del notebook, l'istanziazione di `train_loader` e `val_loader` è stata collocata nella **Cella 4** subito dopo `calibrate_batch_size_and_device` affinché entrambi i loader utilizzino direttamente il `BATCH_SIZE = 16` calibrato sull'hardware attivo (`num_workers=2`, `pin_memory=True`, `seed_worker`, `drop_last=True` in train). Durante il test rapido della Cella 4 è stato impostato `TRAIN_MAX_SAMPLES = 16` (`CACHE_IN_RAM = True`).
* **Decisione Adottata:**  
  Confermare l'architettura post-calibrazione della Cella 4 e imporre come prerequisito operativo esplicito per lo **Step 4** il ripristino di `TRAIN_MAX_SAMPLES = 12000` (o `None` per il training esteso) nella Cella 1 prima di avviare il loop comparativo delle funzioni di perdita.