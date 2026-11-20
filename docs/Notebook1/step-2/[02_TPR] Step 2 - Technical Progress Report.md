# \[02\_TPR\] Step 2 — Technical Progress Report

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Fase:** Step 2 — Parsing Annotazioni JSON (`pycocotools`), Analisi Statistica Completa, Campionamento Bilanciato, `COCOSemanticDataset` e Verifica `DataLoader`  
**Riferimento Strutturale:** Kaggle Notebook di Riferimento — Celle 2, 3 e 4 (Sezioni 6, 7 e 8)  

---

## 1. Obiettivo e Perimetro dello Step 2

Lo **Step 2** trasforma le annotazioni vettoriali grezze di **COCO 2017 (*Thing*)** in una pipeline PyTorch deterministica e riproducibile per la *Semantic Segmentation* multi-classe ($C = 81$). In conformità con le Sezioni 6–8 dello scheletro del corso e con l'organizzazione modulare delle **Celle 2, 3 e 4**, sono stati conseguiti cinque obiettivi verificabili:

1. **Indicizzazione JSON e Mappatura Contigua delle Classi (`Cella 2 — Sezione 6 Parte 1`):** Caricamento di `instances_train2017.json` e `instances_val2017.json` tramite `pycocotools` e costruzione della corrispondenza biunivoca dagli ID originali non contigui di COCO ($1 \dots 90$) agli indici contigui $1 \dots 80$, riservando $0$ al `background`.
2. **Analisi Statistica Integrale degli Split (`Cella 3 — Sezione 6 Parte 2`):** Profilazione sul $100\%$ dei record JSON di `train2017` ($118\,287$ immagini) e `val2017` ($5\,000$ immagini) per misurare immagini prive di oggetti, immagini con lato corto $< 256\text{ px}$, squilibrio tra classi, area relativa media e frazione di istanze che scendono sotto la soglia critica di $8\text{ px}$ equivalenti dopo il ridimensionamento.
3. **Campionamento Bilanciato *Greedy* ed Esportazione Diagnostica (`Cella 3 — Sezioni 6 & 7`):** Implementazione di `select_class_balanced_subset` basato sulla presenza a livello di immagine (*image-level presence*) con quota minima garantita ($\lfloor N / 80 \rfloor$) e salvataggio ad alta risoluzione ($300\text{ DPI}$ PNG + SVG) delle figure `02_step2_coco_uniformity_lorenz_gini` e `03_step2_coco_class_distribution_dumbbell`.
4. **Implementazione di `COCOSemanticDataset` (`Cella 3 — Sezione 7`):** Rasterizzazione delle maschere semantiche con ordinamento per area decrescente (*Painter's Algorithm*), conversione forzata `.convert("RGB")`, trasformazioni geometriche sincronizzate (**Opzione C**) e supporto ibrido *on-the-fly* / precaricamento in RAM (`CACHE_IN_RAM`).
5. **Istanziazione dei `DataLoader` Operativi e *Sanity Check* Visivo (`Cella 4 — Sezione 8`):** Creazione di `train_loader` e `val_loader` con `BATCH_SIZE = 16` calibrato sulla GPU, verifica dimensionale dei tensori sul primo batch e salvataggio della griglia di allineamento `01_step2_dataloader_sanity_overlay.png`.

---

## 2. Parsing JSON e Profilazione Statistica Integrale (`Celle 2 e 3`)

### 2.1 Tempi di Caricamento e Mappatura Categorie (`Cella 2`)

* **Tempo caricamento `instances_train2017.json`:** $11.97\text{ s}$ (inclusa creazione indice)
* **Tempo caricamento `instances_val2017.json`:** $0.51\text{ s}$ (inclusa creazione indice)
* **Tempo complessivo Cella 2:** $23.52\text{ s}$
* **Schema di codifica maschere:** `0 = background`, `1..80 = classi COCO Thing` (`NUM_CLASSES = 81`, `IGNORE_INDEX = 255`).

### 2.2 Confronto Statistico Completo: `TRAIN 2017` vs `VAL 2017` (`Cella 3`)

| Metrica Statistica (100% Annotazioni JSON) | `TRAIN 2017` ($118\,287$ img) | `VAL 2017` ($5\,000$ img) | Implicazione Progettuale |
| :--- | :---: | :---: | :--- |
| **Immagini senza oggetti (solo `background`)** | $1\,021$ ($0.86\%$) | $48$ ($0.96\%$) | Escluse dal subset bilanciato in fase di riempimento |
| **Immagini con lato corto $< 256\text{ px}$** | $822$ ($0.69\%$) | $42$ ($0.84\%$) | Conferma la necessità di resize preventivo (**Opzione C**) |
| **Classi *Thing* coperte** | $80 / 80$ | $80 / 80$ | Copertura completa in entrambi gli split |
| **Top 5 classi più frequenti (n. immagini)** | `person` ($64\,115$), `chair` ($12\,774$), `car` ($12\,251$), `dining table` ($11\,837$), `cup` ($9\,189$) | `person` ($2\,693$), `chair` ($580$), `car` ($535$), `dining table` ($501$), `cup` ($390$) | Identica gerarchia tra training e validation (*split i.i.d.*) |
| **Top 5 classi più rare (n. immagini)** | `hair drier` ($189$), `toaster` ($217$), `parking meter` ($705$), `scissors` ($947$), `bear` ($960$) | `toaster` ($8$), `hair drier` ($9$), `scissors` ($28$), `toothbrush` ($34$), `parking meter` ($37$) | Forte coda lunga (*long-tail*) da proteggere nei subset ridotti |
| **Rapporto di squilibrio (`max / min` img)** | **$339.2\times$** | **$336.6\times$** | Giustifica il campionamento bilanciato per gli esperimenti rapidi |
| **Area relativa media singolo oggetto** | $4.33\%$ dell'immagine | $4.35\%$ dell'immagine | Lo sfondo (`0`) occupa mediamente l'$80\text{--}90\%$ dei pixel |
| **Istanze con lato equiv. $< 8\text{ px}$ post-resize** | $130\,797 / 860\,001$ (**$15.21\%$**) | $5\,526 / 36\,781$ (**$15.02\%$**) | Limite fisico imposto dall'*output stride* $s = 8$ della FCN |

---

## 3. Efficacia del Campionamento Bilanciato *Greedy* ($N = 12\,000$)

L'applicazione di `select_class_balanced_subset(coco_train, max_samples=12000, seed=42)` ha prodotto i seguenti risultati quantitativi (approfonditi teoricamente nella scheda `[02_QA]`):

| Indicatore Distributivo | `TRAIN 2017` Intero ($118\,287$) | `VAL 2017` Intero ($5\,000$) | Subset Bilanciato ($12\,000$) | Variazione Ottenuta |
| :--- | :---: | :---: | :---: | :--- |
| **Classi rappresentate** | $80 / 80$ | $80 / 80$ | **$80 / 80$** | Copertura totale preservata |
| **Min / Max immagini per classe** | $189$ / $64\,115$ | $8$ / $2\,693$ | **$152$ / $6\,398$** | Pavimento minimo garantito $\ge 150$ img ($1.25\%$) |
| **Rapporto squilibrio (`max / min`)** | $339.2\times$ | $336.6\times$ | **$42.1\times$** | Compressione dello squilibrio di **$8.06\times$** |
| **Coefficiente di Gini (Curva di Lorenz)** | $0.442$ | $0.446$ | **$0.365$** | Riduzione della disuguaglianza del **$-17.4\%$** relativo |

---

## 4. Verifica dei `DataLoader` Operativi (`Cella 4`)

Nel codice rifattorizzato, la **Cella 4** esegue prima la calibrazione hardware (`BATCH_SIZE = 16` selezionato su $2 \times \text{Tesla T4}$), salva il checkpoint condiviso dei pesi iniziali (`shared_init_fcn_resnet50.pt`) e istanzia direttamente `train_loader` e `val_loader` con `BATCH_SIZE = 16`, `NUM_WORKERS = 2`, `pin_memory = True` e `seed_worker`:

* **Configurazione `val_loader` (deterministica):** `5000` immagini | `313` batch (`batch_size=16`, `shuffle=False`, `drop_last=False`).
* **Smoke test di refactoring su `train_loader` (`TRAIN_MAX_SAMPLES = 16`, `CACHE_IN_RAM = True`):**
  * Precaricamento in RAM di $16$ coppie completato in **$0.1\text{ s}$** (`1` batch operativo da `16` immagini).
  * *(Nota operativa: prima di avviare lo Step 4 occorre reimpostare `TRAIN_MAX_SAMPLES = 12000` nella Cella 1 affinché `train_loader` esponga $750$ batch/epoca invece di $1$ solo batch).*
* **Sanity check tensoriale del primo batch (`0.706s`):**
  * `images`: shape `(16, 3, 256, 256)`, `dtype=torch.float32`, intervallo `[-2.12, 2.64]` (coerente con normalizzazione ImageNet).
  * `masks`: shape `(16, 256, 256)`, `dtype=torch.int64`, intervallo `[0, 80]` (nessun indice fuori dominio).

---

## 5. Guida alla Lettura delle Tre Figure Diagnostiche Salva-Artefatti

### 5.1 Figura 1 — Griglia di Allineamento Immagine-Maschera (`01_step2_dataloader_sanity_overlay.png`)

* **Come si legge:** Griglia $4 \times 3$ estratta dal primo mini-batch di `train_loader` dopo l'applicazione delle trasformazioni sincronizzate (**Opzione C**): Colonna $1$ `Immagine (256x256)` denormalizzata, Colonna $2$ `Maschera` con etichette presenti nel titolo, Colonna $3$ `Overlay Allineamento` ($\alpha = 0.45$).
* **Cosa dimostra:**
  * **Riga 1 (`mouse, keyboard, book`):** Perfetta coincidenza dei bordi poligonali su oggetti adiacenti su una scrivania, confermando la sincronizzazione tra `TF.resize`, `RandomCrop` e `TF.hflip`.
  * **Riga 2 (`sheep`):** Segmentazione nitida di un intero gregge distribuito su più piani di profondità, a conferma che l'interpolazione `NEAREST` sulle maschere non introduce artefatti né classi spurie tra pixel adiacenti.
  * **Riga 3 (`person, fire hydrant, dog`):** Verifica visiva diretta del **Painter's Algorithm** (`area` decrescente): l'istanza `dog` (giallo ocra) sovrapposta davanti a `person` (viola) viene rasterizzata per ultima e rimane perfettamente visibile anziché essere sovrascritta dalla maschera della persona; anche `fire hydrant` (a destra) combacia al pixel.
  * **Riga 4 (`car, hot dog, cake`):** Conferma la coerenza della mappatura contigua $1 \dots 80$ anche in presenza di piccole istanze sullo sfondo (`car` azzurra a sinistra).

### 5.2 Figura 2 — Distribuzione Logaritmica e Curva di Lorenz (`02_step2_coco_uniformity_lorenz_gini.png`)

* **Pannello Sinistro (*Distribuzione delle 80 Classi Thing in COCO train2017 — Scala Log*):**
  * **Come si legge:** Le $80$ categorie sono ordinate sull'asse $X$ dalla più frequente (`person`, $64\,115$ img) alla più rara (`hair drier`, $189$ img); l'asse $Y$ è in scala logaritmica e la linea rossa tratteggiata indica la media uniforme ideale ($4\,287$ img/classe).
  * **Cosa dimostra:** Evidenzia visivamente la struttura *long-tail* di COCO, con una testa ripida di poche classi sopra la linea media e una lunga coda decrescente di oltre $65$ categorie sotto-rappresentate.
* **Pannello Destro (*Curva di Lorenz — Misura di Disuguaglianza tra Classi*):**
  * **Come si legge:** L'area tra la diagonale rossa (`Gini = 0.000`, uniformità perfetta) e ciascuna curva quantifica la concentrazione delle occorrenze.
  * **Cosa dimostra:** Le curve di `COCO train2017 intero` (`Gini = 0.442`) e `COCO val2017 intero` (`Gini = 0.446`) sono pressoché sovrapposte, dimostrando che `val2017` rispecchia fedelmente la distribuzione reale; la curva verde (`Subset Bilanciato N=12000`, `Gini = 0.365`) si avvicina sensibilmente alla diagonale riducendo lo squilibrio, senza raggiungere `0.000` a causa della co-occorrenza multi-oggetto.

### 5.3 Figura 3 — Cleveland Dumbbell Chart e Fattore di Ribilanciamento (`03_step2_coco_class_distribution_dumbbell.png`)

* **Pannello Sinistro (*Confronto Prevalenza delle 80 Classi — Scala Logaritmica*):**
  * **Come si legge:** Per ciascuna delle $80$ classi (`01. Person` $\rightarrow$ `80. Hair Drier`), confronta la prevalenza percentuale nel dataset completo (pallino blu) con quella nel subset bilanciato da $12\,000$ immagini (pallino verde) rispetto alla soglia minima garantita ($1.25\%$, linea rossa tratteggiata).
  * **Cosa dimostra:** Tutte le classi nella coda inferiore vengono sollevate sopra la soglia dell'$1.25\%$ (es. `80. Hair Drier` passa da $0.16\%$ a $1.27\%$, `79. Toaster` da $0.18\%$ a $1.29\%$, `78. Parking Meter` da $0.60\%$ a $1.70\%$).
* **Pannello Destro (*Moltiplicatore `Subset % / Full %`*):**
  * **Come si legge:** Le barre verdi ($\ge 1.0\times$) mostrano il fattore di amplificazione delle classi rare (`7.93x` per `Hair Drier`, `7.04x` per `Toaster`, `2.84x` per `Parking Meter`), mentre le barre arancioni ($< 1.0\times$) mostrano la lieve attenuazione delle classi sovra-rappresentate che compaiono spesso isolate (`0.78x` `Cell Phone`, `0.80x` `Clock`, `0.82x` `Dog`, `0.98x` `Person`).

---

## 6. Checklist di Validazione dello Step 2

| Controllo Diagnostico | Stato | Esito Verificato |
| :--- | :---: | :--- |
| Parsing JSON e mappatura biunivoca $1 \dots 80$ | **SUPERATO** | Completato in $23.52\text{ s}$; $0 = \text{background}$, $1 \dots 80 = \text{Thing}$ |
| Profilazione statistica integrale `train2017` / `val2017` | **SUPERATO** | Squilibrio $339.2\times$, $15.21\%$ istanze $< 8\text{ px}$ post-resize |
| Campionamento bilanciato *greedy* ($N = 12\,000$) | **SUPERATO** | $80/80$ classi coperte, $\min = 152$ img/classe, Gini $0.442 \rightarrow 0.365$ |
| Allineamento immagine-maschera (**Opzione C** + *Painter*) | **SUPERATO** | Verificato visivamente in `01_step2_dataloader_sanity_overlay.png` |
| Esportazione figure diagnostiche $300\text{ DPI}$ / SVG | **SUPERATO** | Salvate figure `01`, `02` e `03` in `/kaggle/working/figures` |