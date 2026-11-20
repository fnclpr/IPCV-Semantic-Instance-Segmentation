# \[03\_EDL\] Step 3 — Engineering Decision Log

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Fase:** Step 3 — Architettura `FCN-ResNet50` *from scratch*, Calibrazione Hardware, Persistenza Pesi Iniziali e Diagnostica Avanzata (Step 3-BIS)  
**Scopo del Documento:** Tracciamento formale delle decisioni architetturali e metodologiche adottate nelle Celle 4 e 5, delle alternative scartate e delle evidenze sperimentali raccolte.

---

## EDL-12 — Selezione del `BATCH_SIZE` Operativo ($16$ vs $8$ e $32$) su `2 x Tesla T4` (`Cella 4`)

* **Stato:** Approvata e Implementata
* **Contesto e Problema:**  
  In ambiente multi-GPU (`2 x NVIDIA Tesla T4`, $14.56\text{ GB}$ VRAM ciascuna) gestito tramite `nn.DataParallel`, la dimensione del mini-batch determina contemporaneamente la stabilità delle statistiche locali di `BatchNorm2d` (che riceve `BATCH_SIZE / 2` campioni per GPU) e la frequenza di aggiornamento dei pesi per epoca.
* **Evidenze Empiriche (`calibrate_batch_size_and_device` — Cella 4):**
  * `BATCH_SIZE = 8`: $3.25\text{ GB}$ VRAM picco, $682.0\text{ ms}$/step ($4$ immagini/GPU).
  * `BATCH_SIZE = 16`: $5.17\text{ GB}$ VRAM picco, $1152.3\text{ ms}$/step ($8$ immagini/GPU).
  * `BATCH_SIZE = 32`: $9.00\text{ GB}$ VRAM picco, $2226.6\text{ ms}$/step ($16$ immagini/GPU).
* **Decisione Adottata:**  
  Fissare **`BATCH_SIZE = 16`** come standard operativo per i `DataLoader` (Cella 4) e per gli addestramenti dello Step 4.
* **Razionale Tecnico:**  
  Occupa meno del $36\%$ della VRAM di una singola T4 (lasciando ampio margine per i tensori *one-hot* `(16, 81, 256, 256)` delle *loss* regionali nello Step 4), garantisce $8$ campioni per replica `BatchNorm2d` e raddoppia il numero di aggiornamenti dei pesi per epoca rispetto a `BATCH_SIZE = 32` a parità di throughput ($\approx 72\text{ ms}$/immagine).

---

## EDL-13 — Persistenza Centralizzata dei Pesi Iniziali (`shared_init_fcn_resnet50.pt`) e Scelta Backbone

* **Stato:** Approvata e Implementata (`Cella 4`)
* **Contesto e Problema:**  
  La Sezione 12 dello scheletro del corso impone che ogni esperimento comparativo sulle funzioni di perdita inizi dagli **identici pesi iniziali** (*Ceteris Paribus*). Mantenere i pesi solo in memoria volatile espone al rischio di perdita di sincronizzazione tra riavvii di sessione o notebook separati.
* **Decisione Adottata:**  
  1. Istanziazione di `fcn_resnet50(weights=None, weights_backbone=None, num_classes=81, aux_loss=True)` ($35\,368\,418$ parametri addestrabili) al posto di `fcn_resnet101` ($54.3\text{M}$ parametri, più lenta a convergere *from scratch*).
  2. Salvataggio deterministico su disco dello stato iniziale comune in `/kaggle/working/checkpoints/shared_init_fcn_resnet50.pt` subito dopo la calibrazione GPU nella Cella 4.
* **Razionale Tecnico:**  
  Garantisce che sia la mini-ablation dello Step 3-BIS sia tutti i modelli dello Step 4 possano ricaricare esattamente lo stesso punto di partenza nello spazio dei parametri, rendendo scientificamente confrontabili le traiettorie di apprendimento.

---

## EDL-14 — Incapsulamento Modulare della Diagnostica (`RUN_BABYSITTING`) e Distinzione tra Metrica *Batch-Wise* e Accumulatore Globale (`Cella 5`)

* **Stato:** Approvata e Implementata
* **Contesto e Problema:**  
  Durante il *babysitting* su $16$ immagini ($2$ mini-batch da $8$, ossia $4$ immagini per GPU sotto `DataParallel`), le statistiche mobili (`running_mean`, `running_var`) di `BatchNorm2d` subiscono un ritardo rispetto ai pesi aggiornati rapidamente *from scratch*, mentre la valutazione finale richiede una Confusion Matrix globale deterministica in `model.eval()`.
* **Decisione Adottata:**  
  1. Incapsulare l'intera suite diagnostica della Cella 5 (`run_step3_babysitting` e `run_step3bis_mini_ablation`) dietro il flag booleano **`RUN_BABYSITTING`**, così da poter saltare istantaneamente i $\approx 14\text{ minuti}$ di test dopo la prima validazione.
  2. Mantenere `compute_batch_miou_and_acc` confinata esclusivamente come **termometro relativo di convergenza** in `model.train()` durante le $80$ epoche dello Step 3 (`0.53%` $\rightarrow$ `79.13%`), adottando invece l'accumulatore globale deterministico in `model.eval()` (`_evaluate_ablation_subset`) per lo Step 3-BIS e per tutti gli step successivi.
* **Razionale Tecnico:**  
  Evita il costo di $80$ cicli separati di `eval()` durante il *babysitting* preservando un segnale di convergenza pulito (poiché `shuffle=False` mantiene i $2$ batch identici in ogni epoca), mentre impone il calcolo esatto per classe dell'IoU globale in ogni valutazione comparativa ufficiale.

---

## EDL-15 — Analisi Critica dei Risultati Step 3-BIS (`T1` vs `T2` vs `T3`) e Conferma del Setup per lo Step 4

* **Stato:** Approvata e Validata Sperimentalmente (`Cella 5`)
* **Contesto e Problema:**  
  Nella mini-ablation su $16$ immagini ($29$ classi oggetto presenti), `T1` (`256px`, `80 ep`, LR fisso) ha ottenuto `mIoU Obj = 42.38%` (`PixAcc = 91.94%`), mentre `T2` (`256px`, `150 ep`, `CosineAnnealingLR`) ha ottenuto una *training loss* inferiore (`0.0597` vs `0.0664`) e più classi oggetto rilevate (`19/29` vs `18/29`), ma una `mIoU Obj` in `eval()` pari a `38.47%`. `T3` (`384px`, `150 ep`, `CosineLR`) è sceso ulteriormente a `36.01%` impiegando `455.8s`.
* **Diagnostica dei Fenomeni Osservati:**
  1. **Deriva delle *running statistics* di `BatchNorm2d` a $150$ epoche su micro-subset:** Prolungando il training a $150$ epoche senza regolarizzazione (`weight_decay=0.0`) su sole $16$ immagini ($4$ immagini/GPU), nelle epoche finali in cui `CosineAnnealingLR` riduce quasi a zero l'aggiornamento dei pesi, i layer `BatchNorm2d` continuano ad aggiornare `running_mean` e `running_var` con `momentum=0.1` su micro-batch rumorosi, accentuando il divario *train-eval*.
  2. **Validità di `CosineAnnealingLR` nel regime generalizzato (Step 4):** Sul dataset bilanciato da $12\,000$ immagini (`750` batch/epoca, $8$ immagini/GPU) con `weight_decay = 1e-4`, le statistiche di `BatchNorm2d` convergono stabilmente e il decadimento continuo del coseno ($\eta_{\max} \rightarrow \eta_{\min} = 10^{-5}$) previene le oscillazioni tardive senza richiedere soglie arbitrarie (`StepLR`) né alterare il budget di iterazioni (`ReduceLROnPlateau`).
  3. **Scarto definitivo della risoluzione $384 \times 384$ (`T3`):** Pur recuperando parzialmente le mucche distanti nella Riga 1 di `06_step3bis_mini_ablation_visual_comparison.png`, `T3` quadruplica il tempo di calcolo ($455.8\text{ s}$, $4.30\times$ rispetto a `T1`) e introduce buchi interni negli oggetti estesi (fusoliera dell'aereo nella Riga 4) perché il campo recettivo fisso della ResNet-50 copre una frazione minore dell'oggetto ingrandito.
* **Decisione Adottata per lo Step 4:**  
  Confermare **`CROP_SIZE = 256`**, **`BATCH_SIZE = 16`**, **`AdamW`** (`weight_decay = 1e-4`) + **`CosineAnnealingLR`** a partire da `shared_init_fcn_resnet50.pt`, demandando il recupero degli oggetti piccoli e rari alle **funzioni di perdita regionali e focali** (`Focal`, `Dice`, `Jaccard`, `0.5 * CE + 0.5 * Dice`) nello Step 4 a costo zero di memoria VRAM.