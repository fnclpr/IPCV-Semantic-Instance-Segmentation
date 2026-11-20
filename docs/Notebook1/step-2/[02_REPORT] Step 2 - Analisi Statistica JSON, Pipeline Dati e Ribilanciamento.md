# \[02\_REPORT\] Step 2 — Analisi Statistica JSON, Pipeline Dati e Ribilanciamento

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Riferimento Pianificazione:** `[00_PLANNING] Scaletta Progetto CV` — Step 2  
**Allineamento Scheletro Kaggle Notebook:** Celle 2, 3 e 4 — Sezioni 6, 7 e 8 (*COCO JSON Parsing, Statistical Analysis, Class-Balanced Subset, COCOSemanticDataset & DataLoader Verification*)  

---

## 1. Inquadramento Teorico e Obiettivi Progettuali

Dall'ispezione condotta nello Step 1 sono emerse tre caratteristiche fisiche di **COCO 2017 (*Thing*)** che governano la progettazione della pipeline dati: l'assenza di maschere `.png` pre-renderizzate, la presenza di immagini in scala di grigi (`PIL mode == 'L'`, $0.4\%$) e la presenza di immagini con lato minore inferiore a `CROP_SIZE = 256` ($H_{\min} = 230\text{ px}$).

Lo **Step 2** traduce tali vincoli in una pipeline PyTorch deterministica e ad alte prestazioni articolata su tre blocchi modulari:

1. **Indicizzazione JSON e Mappatura Contigua (`Cella 2`):** Caricamento di `instances_train2017.json` ($11.97\text{ s}$) e `instances_val2017.json` ($0.51\text{ s}$) tramite `pycocotools` (tempo totale $23.52\text{ s}$) e costruzione della mappatura biunivoca dagli ID non contigui di COCO ($1 \dots 90$) agli indici contigui $1 \dots 80$, riservando $0$ al `background` ($C = 81$).
2. **Analisi Statistica Integrale e Campionamento Bilanciato (`Cella 3`):** Profilazione del $100\%$ delle annotazioni JSON e selezione di un sottoinsieme bilanciato da $12\,000$ immagini basato sulla presenza a livello di immagine (*image-level presence*), corredata dall'esportazione ad alta risoluzione delle figure `02` e `03`.
3. **Implementazione di `COCOSemanticDataset` e `DataLoader` Operativi (`Celle 3 e 4`):** Rasterizzazione delle maschere con ordinamento per area decrescente (*Painter's Algorithm*), trasformazioni geometriche sincronizzate (**Opzione C**), supporto ibrido RAM/disco e verifica tensoriale e visiva sul primo batch (`01_step2_dataloader_sanity_overlay.png`).

---

## 2. Evidenze Empiriche dell'Analisi Statistica Integrale (`Cella 3`)

| Proprietà Misurata (100% Record JSON) | `TRAIN 2017` ($118\,287$ img) | `VAL 2017` ($5\,000$ img) |
| :--- | :---: | :---: |
| **Immagini senza oggetti (solo `background`)** | $1\,021$ ($0.86\%$) | $48$ ($0.96\%$) |
| **Immagini con lato corto $< 256\text{ px}$** | $822$ ($0.69\%$) | $42$ ($0.84\%$) |
| **Classi *Thing* coperte** | $80 / 80$ | $80 / 80$ |
| **Top 5 classi più frequenti (n. immagini)** | `person` ($64\,115$), `chair` ($12\,774$), `car` ($12\,251$), `dining table` ($11\,837$), `cup` ($9\,189$) | `person` ($2\,693$), `chair` ($580$), `car` ($535$), `dining table` ($501$), `cup` ($390$) |
| **Top 5 classi più rare (n. immagini)** | `hair drier` ($189$), `toaster` ($217$), `parking meter` ($705$), `scissors` ($947$), `bear` ($960$) | `toaster` ($8$), `hair drier` ($9$), `scissors` ($28$), `toothbrush` ($34$), `parking meter` ($37$) |
| **Rapporto di squilibrio (`max / min` immagini)** | **$339.2\times$** | **$336.6\times$** |
| **Area relativa media singolo oggetto** | $4.33\%$ dell'immagine | $4.35\%$ dell'immagine |
| **Istanze con lato equiv. $< 8\text{ px}$ post-resize** | $130\,797 / 860\,001$ (**$15.21\%$**) | $5\,526 / 36\,781$ (**$15.02\%$**) |

---

## 3. Campionamento Bilanciato *Greedy* e Guida ai Grafici Distributivi

Per condurre gli esperimenti comparativi dello Step 4 in tempi rapidi senza tagliare fuori le categorie rare, `select_class_balanced_subset` seleziona $12\,000$ immagini garantendo a ciascuna classe una quota minima $\lfloor 12000 / 80 \rfloor = 150$ immagini ($1.25\%$ del subset), portando il minimo effettivo a **$152$ immagini** (`80/80` classi coperte, massimo $6\,398$, squilibrio ridotto da $339.2\times$ a **$42.1\times$**).

1. **Distribuzione Logaritmica e Curva di Lorenz (`02_step2_coco_uniformity_lorenz_gini.png`):**
   * Il **pannello sinistro** evidenzia in scala logaritmica la distribuzione *long-tail* di `train2017` rispetto alla media uniforme ideale ($4\,287$ img/classe).
   * Il **pannello destro** mostra la sovrapposizione quasi perfetta tra `train2017` (`Gini = 0.442`) e `val2017` (`Gini = 0.446`), confermando la rappresentatività statistica del validation set (lasciato intatto per evitare *Prior Probability Shift*), mentre il **Subset Bilanciato ($N = 12\,000$)** abbassa il coefficiente di Gini a **$0.365$** ($-17.4\%$ relativo), senza annullarlo a causa della co-occorrenza multi-oggetto.
2. **Cleveland Dumbbell Chart e Moltiplicatore (`03_step2_coco_class_distribution_dumbbell.png`):**
   * Il **pannello sinistro** documenta in scala logaritmica il sollevamento di tutte le classi della coda inferiore sopra la soglia minima garantita dell'$1.25\%$.
   * Il **pannello destro** quantifica il moltiplicatore `Subset % / Full %`: le classi più rare ricevono fino a **$7.93\times$** (`Hair Drier`) e **$7.04\times$** (`Toaster`), mentre le classi dominanti che compaiono spesso isolate vengono lievemente attenuate (`0.78x` `Cell Phone`, `0.80x` `Clock`, `0.98x` `Person`).
3. **Granularità Image-Level vs Pixel-Level (`[02_QA]`):**
   * Come approfondito in `Step02_QA_Granularita_Campionamento.md`, il campionamento per immagine distinta massimizza la varietà ambientale evitando l'*overfitting* contestuale tipico del conteggio per istanze, mentre lo sbilanciamento residuo a livello di area in pixel verrà compensato tramite le funzioni di perdita focali e regionali nello **Step 4**.

---

## 4. Pipeline `COCOSemanticDataset` e Verifica `DataLoader` (`Cella 4`)

* **Rasterizzazione e *Painter's Algorithm*:** Le istanze vengono ordinate per area decrescente affinché gli oggetti più piccoli in primo piano non vengano coperti dalle superfici più estese.
* **Trasformazioni Sincronizzate (**Opzione C**):** Ridimensionamento proporzionale del lato corto a `256 px` (`BILINEAR` + `antialias=True` per RGB, `NEAREST` per maschere), seguito da `RandomCrop(256, 256)` + `TF.hflip` sincronizzati in training e `CenterCrop(256, 256)` deterministico in validation.
* **Diagnostica del Primo Batch (`Cella 4`, `BATCH_SIZE = 16`, `0.706s`):**
  * `images`: `(16, 3, 256, 256)`, `torch.float32`, range `[-2.12, 2.64]`
  * `masks`: `(16, 256, 256)`, `torch.int64`, range `[0, 80]`
  * **Griglia Visiva (`01_step2_dataloader_sanity_overlay.png`):** Conferma l'allineamento sub-pixel su tutte e quattro le scene campionate (`mouse, keyboard, book`, `sheep`, `person, fire hydrant, dog` e `car, hot dog, cake`), inclusa la corretta sovrapposizione in primo piano di `dog` su `person` nella Riga 3.
  * *(Nota operativa: nel test rapido della Cella 4 era attivo `TRAIN_MAX_SAMPLES = 16` con cache RAM in $0.1\text{ s}$; prima dello Step 4 occorre impostare `TRAIN_MAX_SAMPLES = 12000` nella Cella 1).*

---

## 5. Implicazioni Dirette per gli Step Successivi

1. **Sbilanciamento Pixel-Wise ($339.2\times$ + Background Dominante):** Giustifica il confronto sistematico nello **Step 4** tra *Cross-Entropy*, *Focal Loss*, *Dice Loss*, *Jaccard Loss* e combinazioni ibride (`0.5 * CE + 0.5 * Dice`).
2. **Oggetti Piccoli Post-Resize ($15.21\%$ con lato equivalente $< 8\text{ px}$):** Spiega il limite fisico strutturale dell'*output stride* $s = 8$ della FCN standard (griglia latente $32 \times 32$ su ingressi $256 \times 256$), verificato sperimentalmente nello Step 3.