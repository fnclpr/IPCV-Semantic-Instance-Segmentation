# \[01\_REPORT\] Step 1 — Setup, Configurazione e Ispezione Dati (COCO 2017)

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Riferimento Pianificazione:** `[00_PLANNING] Scaletta Progetto CV` — Step 1  
**Allineamento Scheletro Kaggle Notebook:** Cella 1 — Sezioni 1–5 (*Imports, Centralized Configuration, Reproducibility, Directory & Split Inspection, Raw Sample Profiling*)  

---

## 1. Inquadramento Teorico e Obiettivi Progettuali

Prima di implementare una pipeline di *Deep Learning* per la **Semantic Segmentation**, è necessario stabilire un ambiente sperimentale deterministico e misurare le proprietà fisiche dei dati grezzi su disco. Come evidenziato nei moduli teorici del corso (*CS231n — Babysitting the Learning Process*) e nell'impostazione modulare del Kaggle Notebook di riferimento, lo **Step 1** assolve a quattro funzioni propedeutiche:

1. **Centralizzazione della Configurazione e Persistenza (`Sezioni 1–2`):** Raccogliere in un'unica cella iniziale tutti gli import, i percorsi (`DATA_ROOT`, `FIGURES_DIR`, `CHECKPOINTS_DIR`), le costanti strutturali (`SEED = 42`, `CROP_SIZE = 256`, `NUM_CLASSES = 81`, `IGNORE_INDEX = 255`, `AUX_LOSS_WEIGHT = 0.5`) e i flag di controllo (`RUN_EDA_PLOTS`, `RUN_BABYSITTING`, `TRAIN_MAX_SAMPLES`, `CACHE_IN_RAM`), eliminando *magic numbers* e garantendo l'archiviazione ordinata di figure e checkpoint.
2. **Riproducibilità Deterministica (`Sezione 3`):** Bloccare tutte le sorgenti di stocasticità (`PYTHONHASHSEED`, `random`, `numpy`, `torch`, `torch.cuda` e backend `cuDNN`) affinché l'inizializzazione *from scratch* dei pesi e il campionamento dei batch risultino identici tra esperimenti diversi.
3. **Verifica della Struttura Directory ed Estensioni Reali (`Sezione 4`):** Accertare empiricamente quali file sono presenti nel dataset COCO 2017 montato su Kaggle (`train2017`, `val2017`, `test2017` e `annotations/`), verificando l'estensione reale delle immagini e l'assenza di maschere `.png` pre-renderizzate.
4. **Ispezione Statistica dei Campioni Grezzi (`Sezione 5`):** Analizzare modalità colore (`PIL mode`) e distribuzioni di altezza e larghezza $(H, W)$ su un campione deterministico di $500$ immagini per prevenire errori di dimensione o di canale nella pipeline dello Step 2.

---

## 2. Evidenze Empiriche Misurate sull'Ambiente Kaggle

### 2.1 Ambiente Hardware e Software

* **Versioni Runtime:** `PyTorch 2.11.0+cu130` | `Torchvision 0.26.0+cu130`
* **Acceleratore Rilevato:** `cuda` con **$2 \times \text{NVIDIA Tesla T4}$** ($14.56\text{ GB}$ VRAM ciascuna, $29.12\text{ GB}$ totali)
* **Percorso Dataset (`DATA_ROOT`):** `/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017`
* **Directory di Output Create:** `/kaggle/working/figures` e `/kaggle/working/checkpoints`
* **Seed Deterministico Fissato:** `42`

### 2.2 Struttura Directory, Split ed Estensioni (`inspect_raw_coco_directory`)

| Elemento Ispezionato | Risultato Empirico |
| :--- | :--- |
| **Contenuto `DATA_ROOT`** | `['annotations', 'test2017', 'train2017', 'val2017']` |
| **Maschere `.png` pre-renderizzate** | `NESSUNA` (task definito su COCO 2017 *Thing*, $C = 81$ classi via JSON) |
| **Split `train2017`** | $118\,287$ file — Estensioni: `{'.jpg': 118287}` |
| **Split `val2017`** | $5\,000$ file — Estensioni: `{'.jpg': 5000}` |
| **Split `test2017`** | $40\,670$ file — Estensioni: `{'.jpg': 40670}` |
| **Annotazioni JSON (`annotations/`)** | `instances_train2017.json` ($448.02\text{ MB}$), `instances_val2017.json` ($19.06\text{ MB}$), più file `captions_*.json` e `person_keypoints_*.json` |

### 2.3 Profilazione su $500$ Immagini Grezze di `val2017`

| Proprietà Analizzata | Valore Misurato |
| :--- | :--- |
| **Modalità colore (`PIL mode`)** | `{'RGB': 498, 'L': 2}` ($99.6\%$ RGB a $3$ canali, $0.4\%$ scala di grigi a $1$ canale) |
| **Risoluzioni distinte** | $133$ risoluzioni diverse su $500$ immagini campionate |
| **Altezza $H$ (`min / med / max`)** | $230\text{ px}$ / $480\text{ px}$ / $640\text{ px}$ |
| **Larghezza $W$ (`min / med / max`)** | $246\text{ px}$ / $640\text{ px}$ / $640\text{ px}$ |

---

## 3. Analisi Critica e Decisioni Progettuali Trasferite allo Step 2

Dall'analisi quantitativa dello Step 1 derivano quattro vincoli ingegneristici che governano direttamente la progettazione del `COCOSemanticDataset` nello Step 2:

1. **Rasterizzazione via `pycocotools` (`EDL-01`):** L'assenza di maschere `.png` pre-renderizzate nel dataset COCO 2017 (*Thing*) richiede l'integrazione approvata di `pycocotools` per decodificare poligoni e RLE dai file `instances_*.json`, mappando le $80$ categorie *Thing* negli indici contigui $1 \dots 80$ e riservando $0$ al `background` ($C = 81$).
2. **Normalizzazione dei Canali RGB (`EDL-05`):** La presenza dello $0.4\%$ di immagini in scala di grigi (`PIL mode == 'L'`, $2/500$) impone la chiamata esplicita `.convert("RGB")` su ogni immagine caricata; in caso contrario, il `DataLoader` fallirebbe durante l'impilamento di tensori $(1, H, W)$ con tensori $(3, H, W)$.
3. **Gestione delle Immagini con Lato $< 256\text{ px}$ (`EDL-05`):** Avendo rilevato immagini con $H_{\min} = 230\text{ px}$ e $W_{\min} = 246\text{ px}$, un `RandomCrop(256, 256)` applicato direttamente alle dimensioni native solleverebbe un `ValueError`. Si adotta pertanto la strategia **Opzione C**: ridimensionamento proporzionale del lato corto a `CROP_SIZE = 256` (preservando l'*aspect ratio* e portando ogni lato $\ge 256\text{ px}$) seguito da `RandomCrop(256, 256)` + `RandomHorizontalFlip` sincronizzati in training e `CenterCrop(256, 256)` deterministico in validazione.
4. **Strategia Ibrida Memoria/Disco (`EDL-04`):** A differenza dello scheletro del corso, precaricare $118\,287$ immagini in RAM causerebbe un crash *Out-Of-Memory*. Il dataset supporterà quindi il caricamento *on-the-fly* (`CACHE_IN_RAM = False`) sull'intero split e il precaricamento selettivo (`CACHE_IN_RAM = True`) sui sottoinsiemi sperimentali ridotti.