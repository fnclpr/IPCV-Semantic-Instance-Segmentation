# \[02\_QA\] Step 2 — Uniformità del Dataset, Granularità del Campionamento e Sbilanciamento nella *Semantic Segmentation*

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Riferimento Codice:** `Notebook1_SetupDatasetCOCO2017_Babysitting_(Step1_Step3-BIS)` — Celle 1, 2, 3, 4 e 5  
**Collegamento Progettuale:** Step 2 (*Analisi Statistica JSON e Campionamento Bilanciato*) $\rightarrow$ Step 4 (*Confronto Loss Function*)  
**Tipo di Documento:** Scheda di Approfondimento Teorico-Metodologico (Q&A)

---

## 1. Quesiti Metodologici Formalizzati

Questo documento risponde in modo organico a quattro quesiti chiave emersi durante lo sviluppo e il refactoring della pipeline dati su **COCO 2017 (*Thing*)**:

1. **Diagnostica dell'Uniformità:** *Quanto è realmente disuniforme COCO 2017 (*Thing*) e come misuriamo rigorosamente lo squilibrio prima e dopo il campionamento senza farci ingannare dagli outlier (come la classe `person`)?*
2. **Granularità del Campionamento:** *Nella funzione `select_class_balanced_subset` (Cella 3), il criterio di uniformazione opera sul numero di immagini distinte per classe, sul numero di istanze oppure sul numero di pixel? Qual è la scelta metodologicamente corretta per la Semantic Segmentation?*
3. **Limiti Fisici dell'Uniformità:** *Perché nemmeno l'algoritmo greedy porta l'indice di Gini a $0.000$, e perché il validation set (`val2017`) non deve essere ribilanciato?*
4. **Integrazione nell'Architettura Rifattorizzata:** *Come viene governato il campionamento bilanciato tra la Cella 1 (configurazione), la Cella 3 (definizione e grafici EDA), la Cella 4 (`DataLoader` operativi) e lo Step 4 (funzioni di perdita)?*

---

## 2. Evidenze Empiriche di Non-Uniformità in COCO 2017 (`Celle 2 e 3`)

L'analisi integrale eseguita dalla funzione `analyze_coco_split` (Cella 3) sul $100\%$ dei record JSON caricati tramite `pycocotools` (Cella 2) ha quantificato con precisione la struttura *long-tail* di entrambi gli split:

| Indicatore Distributivo (100% Annotazioni JSON) | `TRAIN 2017` ($118\,287$ img) | `VAL 2017` ($5\,000$ img) | Implicazione Teorica |
| :--- | :---: | :---: | :--- |
| **Immagini senza oggetti (solo `background`)** | $1\,021$ ($0.86\%$) | $48$ ($0.96\%$) | Escluse nella Fase 2 del campionatore *greedy* |
| **Immagini con lato corto $< 256\text{ px}$** | $822$ ($0.69\%$) | $42$ ($0.84\%$) | Gestite tramite resize proporzionale (**Opzione C**) |
| **Classi *Thing* coperte** | $80 / 80$ | $80 / 80$ | Copertura completa della tassonomia |
| **Classe più frequente (`head class`)** | `person`: $64\,115$ img ($54.20\%$) | `person`: $2\,693$ img ($53.86\%$) | Domina i gradienti nella *Cross-Entropy* standard |
| **Classe più rara (`tail class`)** | `hair drier`: $189$ img ($0.16\%$) | `toaster`: $8$ img ($0.16\%$) | Rischio di *underfitting* selettivo su subset casuali |
| **Rapporto di squilibrio (`max / min` img)** | **$339.2\times$** | **$336.6\times$** | Forte asimmetria tra *head* e *tail classes* |
| **Area relativa media singolo oggetto** | $4.33\%$ dell'immagine | $4.35\%$ dell'immagine | Circa l'$80\text{--}90\%$ dei pixel appartiene a `0 = background` |
| **Istanze con lato equiv. $< 8\text{ px}$ post-resize** | $130\,797 / 860\,001$ ($15.21\%$) | $5\,526 / 36\,781$ ($15.02\%$) | Sotto la risoluzione della griglia latente FCN ($s = 8$) |
| **Coefficiente di Gini (Curva di Lorenz)** | **$0.442$** | **$0.446$** | Identica struttura statistica tra train e val |

---

## 3. Come Misuriamo Visivamente e Quantitativamente l'Uniformità (`Cella 3`)

Un istogramma lineare standard su $80$ classi risulta poco leggibile perché l'outlier `person` ($54.2\%$) schiaccia verso lo zero oltre $75$ categorie. Per superare questo limite, la **Cella 3** (quando `RUN_EDA_PLOTS = True`) genera ed esporta in `/kaggle/working/figures/` ($300\text{ DPI}$ PNG + SVG) due strumenti diagnostici complementari:

1. **Distribuzione Logaritmica e Curva di Lorenz (`02_step2_coco_uniformity_lorenz_gini.png` / `.svg`):**
   * **Pannello Sinistro (Scala Logaritmica):** Ordina le $80$ classi dalla più frequente (`person`, $64\,115$) alla più rara (`hair drier`, $189$) confrontandole con la media uniforme ideale ($4\,287$ immagini/classe).
   * **Pannello Destro (Curva di Lorenz e Indice di Gini $G \in [0, 1]$):** Misura l'area tra la bisettrice di uniformità perfetta ($G = 0.000$) e la curva cumulata delle occorrenze. Dimostra che `train2017` ($G = 0.442$) e `val2017` ($G = 0.446$) hanno la stessa distribuzione naturale, mentre il **Subset Bilanciato ($N = 12\,000$)** abbassa il coefficiente di Gini a **$0.365$** (riduzione relativa del **$-17.4\%$**).
2. **Cleveland Dumbbell Chart e Moltiplicatore di Ribilanciamento (`03_step2_coco_class_distribution_dumbbell.png` / `.svg`):**
   * **Pannello Sinistro:** Mostra in scala logaritmica lo spostamento di ciascuna delle $80$ classi dalla prevalenza originale (pallino blu) a quella nel subset bilanciato (pallino verde) rispetto alla soglia minima garantita $\lfloor 12\,000 / 80 \rfloor / 12\,000 = 1.25\%$ (linea rossa tratteggiata).
   * **Pannello Destro:** Quantifica il moltiplicatore $\frac{p_{\text{subset}}(c)}{p_{\text{full}}(c)}$: le classi di coda ricevono un'amplificazione fino a **$7.93\times$** (`Hair Drier`, da $0.16\%$ a $1.27\%$) e **$7.04\times$** (`Toaster`, da $0.18\%$ a $1.29\%$), mentre le classi dominanti che compaiono spesso isolate vengono lievemente attenuate (`0.78x` `Cell Phone`, `0.80x` `Clock`, `0.82x` `Dog`, `0.98x` `Person`).

---

## 4. Analisi Critica dei Tre Livelli di Granularità

In un dataset multi-oggetto e multi-classe come **COCO 2017**, la frequenza di una categoria $c \in \{1, \dots, 80\}$ può essere definita a tre livelli fisicamente distinti:

| Livello di Granularità | Definizione Formale | Comportamento nel Codice Attuale | Impatto sull'Apprendimento |
| :--- | :--- | :--- | :--- |
| **1. Frequenza per Immagine** *(adottata in `select_class_balanced_subset`)* | $N_{\text{img}}(c) = \lvert \{ i : \exists\,\text{istanza di } c \text{ in } i \} \rvert$ | Ogni immagine contenente almeno un'istanza di $c$ incrementa `selected_class_counts[c]` di $+1$. | Massimizza la **diversità di scena, sfondo e illuminazione** per ogni categoria. |
| **2. Frequenza per Istanza** *(conteggio annotazioni JSON)* | $N_{\text{inst}}(c) = \sum_{i=1}^{N} n_i(c)$ | Utilizzata solo a fini diagnostici (`class_inst_counts` in `analyze_coco_split`). | Premia scene affollate di duplicati, inducendo *overfitting* sullo sfondo. |
| **3. Frequenza per Pixel** *(area effettiva nelle maschere)* | $N_{\text{px}}(c) = \sum_{i=1}^{N} \sum_{y,x} \mathbb{I}(M_i[y,x] = c)$ | Non forzata nel campionamento; governata dalle **loss function** nello Step 4. | Determina direttamente il **gradiente pixel-wise** $\nabla_W \mathcal{L}$. |

### 4.1 Perché NON bilanciare sul numero di istanze ($N_{\text{inst}}$)

Sostituire il conteggio per immagine distinta con il conteggio per istanze nella selezione del subset sarebbe metodologicamente errato per due motivi:

1. **Unità atomica del `DataLoader`:** `COCOSemanticDataset.__getitem__` (Cella 3) restituisce l'**intera immagine** $(3, 256, 256)$ con la relativa maschera $(256, 256)$, non ritagli di singole istanze.
2. **Rischio di *Overfitting* Contestuale:** Se una classe rara (es. `toaster` o `parking meter`) comparisse $15$ volte in ciascuna di $10$ immagini del medesimo ambiente, raggiungerebbe la quota di $150$ istanze con **sole $10$ fotografie**. La rete memorizzerebbe quei $10$ sfondi specifici anziché generalizzare la forma dell'oggetto. Contare le **immagini distinte** garantisce invece che la rete osservi almeno $152$ scene differenti per ogni categoria.

### 4.2 Perché l'uniformità perfetta ($G = 0.000$) è fisicamente irraggiungibile

Anche applicando l'algoritmo *greedy* con quota minima $\lfloor N / 80 \rfloor = 150$, il coefficiente di Gini scende da $0.442$ a $0.365$ senza annullarsi per tre ragioni strutturali:

1. **Co-occorrenza multi-oggetto:** Ogni immagine selezionata per includere una classe rara (`hair drier`, `toaster`, `toothbrush`) contiene spesso anche `person`, `chair` o `dining table`. Nella Fase 1 di `select_class_balanced_subset`, `selected_class_counts[present_c] += 1` aggiorna tutte le classi co-presenti nell'immagine, mantenendo `person` a $6\,398$ immagini ($53.3\%$) e comprimendo comunque il rapporto di squilibrio da **$339.2\times$** a **$42.1\times$**.
2. **Disparità intrinseca di area in pixel:** $152$ immagini contenenti `hair drier` o `fork` producono molti meno pixel annotati rispetto a $152$ immagini contenenti `bus`, `bed` o `bear`, oltre alla dominanza strutturale della classe `0 = background` ($80\text{--}90\%$ dei pixel).
3. **Effetto del cropping spaziale (**Opzione C**):** Durante il training (`is_train=True`), `RandomCrop(256, 256)` può escludere parzialmente o totalmente piccole istanze periferichericadenti fuori dalla finestra di ritaglio.

---

## 5. Perché il Validation Set (`val2017`) Non Viene Ribilanciato (*Prior Probability Shift*)

Nella **Cella 4**, il validation set viene costruito mantenendo intatte tutte le $5\,000$ immagini originali:

```python
val_ids = sorted(coco_val.getImgIds())  # Validation set sempre completo (5000 immagini)
```

Applicare il campionamento bilanciato anche a `val2017` introdurrebbe **Prior Probability Shift** (alterazione delle probabilità a priori $P(y = c)$): sovra-rappresentando artificialmente le classi rare nel set di valutazione, le metriche aggregate (`mIoU`, `Pixel Accuracy`) non misurerebbero più le prestazioni reali del modello sulla distribuzione naturale del mondo reale ($G = 0.446$).

---

## 6. Architettura Solutiva nel Codice Rifattorizzato (`Celle 1–5` $\rightarrow$ Step 4)

Nel notebook rifattorizzato (`Notebook1_SetupDatasetCOCO2017_Babysitting_(Step1_Step3-BIS)`), il problema dello sbilanciamento è risolto separando nettamente i ruoli tra **livello dati** e **livello funzione di perdita**:

1. **Configurazione Centralizzata (`Cella 1`):**
   * Il parametro `TRAIN_MAX_SAMPLES` governa in modo trasparente la dimensione del training set: impostato a `12000` (con `CACHE_IN_RAM = True`) produce un subset bilanciato da $750$ batch/epoca (`BATCH_SIZE = 16`) ideale per lo studio comparativo delle *loss* nello Step 4; impostato a `None` (con `CACHE_IN_RAM = False`) utilizza l'intero `train2017` ($118\,287$ immagini).
   * *(Nota operativa verificata nella Cella 4: durante lo smoke test rapido della Cella 4 era attivo `TRAIN_MAX_SAMPLES = 16`; prima di eseguire lo Step 4 è obbligatorio reimpostare `TRAIN_MAX_SAMPLES = 12000` nella Cella 1).*
2. **Riutilizzo Modulare di `select_class_balanced_subset` (`Celle 3, 4 e 5`):**
   * Nella **Cella 4**, `select_class_balanced_subset(coco_train, max_samples=TRAIN_MAX_SAMPLES, seed=SEED)` alimenta il `train_loader` operativo (`BATCH_SIZE = 16`).
   * Nella **Cella 5**, `select_class_balanced_subset(coco_train, max_samples=100, seed=SEED)[:16]` seleziona $16$ immagini ricche di categorie diverse ($29$ classi *Thing* distinte) per rendere il *babysitting* e la mini-ablation (`Step 3-BIS`) molto più informativi rispetto a $16$ immagini casuali dominate solo da `person`.
3. **Compensazione *Pixel-Level* nello Step 4 (`shared_init_fcn_resnet50.pt`):**
   * Poiché il bilanciamento *image-level* dello Step 2 non può (e non deve) eliminare lo squilibrio in pixel tra classi grandi, classi piccole e `background`, lo **Step 4** confronta *ceteris paribus* (partendo dagli identici pesi salvati in `/kaggle/working/checkpoints/shared_init_fcn_resnet50.pt`) cinque formulazioni della funzione di perdita:
     * **Cross-Entropy Loss (`CE`):** ogni pixel ha peso uniforme $\frac{1}{H \cdot W}$ (sensibile allo squilibrio di area).
     * **Focal Loss (`FL`):** attenua il contributo dei pixel facili dominati dal `background` tramite il fattore modulante $(1 - p_t)^\gamma$.
     * **Multi-Class Dice Loss & Jaccard (IoU) Loss:** normalizzano l'intersezione rispetto all'area della classe ($|P_c| + |G_c|$ oppure $|P_c \cup G_c|$), attribuendo a un oggetto piccolo/raro lo stesso peso relativo di un oggetto esteso.
     * **Combinazione Ibrida (`0.5 * CE + 0.5 * Dice`):** unisce la regolarità del gradiente locale della `CE` con la sensibilità alle classi piccole della `Dice`.

---

## 7. Sintesi Pronta per la Discussione Orale

* **Diagnosi:** *Su COCO 2017 (*Thing*), le $80$ categorie seguono una distribuzione long-tail con squilibrio $339.2\times$ (`person` $54.2\%$ vs `hair drier` $0.16\%$) e indice di Gini $G = 0.442$ (`train2017`) / $0.446$ (`val2017`).*
* **Scelta di Granularità:** *Il campionatore greedy (`select_class_balanced_subset`) bilancia sulla presenza per **immagine distinta** ($N_{\text{img}}(c)$), garantendo almeno $152$ scene diverse per ciascuna delle $80$ classi su $N = 12\,000$ ($G = 0.365$) ed evitando l'overfitting contestuale che deriverebbe dal conteggio per istanze.*
* **Validazione e Funzioni di Perdita:** *Il validation set (`val2017`, $5\,000$ immagini) resta intatto per evitare Prior Probability Shift, mentre lo sbilanciamento residuo a livello di pixel viene compensato nello Step 4 tramite loss focali e regionali (`Focal`, `Dice`, `Jaccard`, `CE + Dice`).*