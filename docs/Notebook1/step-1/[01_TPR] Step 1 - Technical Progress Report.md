# \[01\_TPR\] Step 1 — Technical Progress Report

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Fase:** Step 1 — Setup dell'Ambiente, Configurazione Centralizzata, Determinismo e Ispezione Dati Grezzi (COCO 2017)  
**Riferimento Strutturale:** Kaggle Notebook di Riferimento — Cella 1 (Sezioni 1–5)  

---

## 1. Obiettivo e Perimetro dello Step 1

Lo **Step 1** stabilisce l'infrastruttura sperimentale deterministica e verifica le caratteristiche fisiche del dataset **COCO 2017** montato nell'ambiente Kaggle prima di costruire la pipeline di caricamento dati (`Dataset` / `DataLoader`) e l'architettura neurale. In conformità con le Sezioni 1–5 dello scheletro del corso e con il refactoring modulare della **Cella 1**, sono stati conseguiti quattro obiettivi verificabili:

1. **Centralizzazione della Configurazione e Persistenza (`Sezione 2`):** Definizione in un unico blocco iniziale dei percorsi di input (`DATA_ROOT`), delle directory persistenti di output (`FIGURES_DIR`, `CHECKPOINTS_DIR`), delle costanti strutturali (`SEED`, `CROP_SIZE`, `NUM_CLASSES`, `IGNORE_INDEX`, `AUX_LOSS_WEIGHT`, `IMAGENET_MEAN`, `IMAGENET_STD`) e dei flag modulari di controllo dell'esecuzione (`RUN_EDA_PLOTS`, `RUN_BABYSITTING`, `TRAIN_MAX_SAMPLES`, `CACHE_IN_RAM`).
2. **Riproducibilità Deterministica Completa (`Sezione 3`):** Blocco dei generatori pseudo-casuali su Python (`os.environ["PYTHONHASHSEED"]`, `random`), `numpy`, `torch` CPU e `torch.cuda`, unitamente alla configurazione deterministica dei backend `cuDNN`.
3. **Ispezione della Struttura Directory ed Estensioni Reali (`Sezione 4`):** Verifica empirica su disco degli split (`train2017`, `val2017`, `test2017`), dei file di annotazione JSON in `annotations/` e dell'assenza di maschere semantiche `.png` pre-renderizzate.
4. **Profilazione Statistica dei Campioni Grezzi (`Sezione 5`):** Campionamento deterministico di $500$ immagini da `val2017` per misurare modalità colore (`PIL mode`), varietà delle risoluzioni e intervalli spaziali $(H, W)$.

---

## 2. Configurazione Hardware, Software e Parametri Globali

L'esecuzione della **Cella 1** sull'ambiente Kaggle ha restituito la seguente configurazione verificata:

| Parametro / Risorsa | Valore Rilevato / Fissato | Ruolo nel Progetto |
| :--- | :--- | :--- |
| **Framework e Runtime** | `PyTorch 2.11.0+cu130` \| `Torchvision 0.26.0+cu130` | Supporto nativo per `fcn_resnet50`, trasformazioni funzionali e `DataParallel` |
| **Acceleratore Hardware** | `cuda` — $2 \times \text{NVIDIA Tesla T4}$ ($14.56\text{ GB}$ VRAM ciascuna) | Esecuzione multi-GPU ($29.12\text{ GB}$ VRAM complessivi) |
| **Percorso Dataset (`DATA_ROOT`)** | `/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017` | Directory radice verificata automaticamente |
| **Persistenza Artefatti** | `/kaggle/working/figures` e `/kaggle/working/checkpoints` | Archiviazione centralizzata di figure ($300\text{ DPI}$ PNG/SVG) e pesi `.pt` |
| **Riproducibilità (`SEED`)** | $42$ (`cudnn.deterministic=True`, `cudnn.benchmark=False`) | Garanzia di confronto *ceteris paribus* tra esperimenti |
| **Risoluzione e Classi** | `CROP_SIZE = 256` ($256 \times 256\text{ px}$), `NUM_CLASSES = 81`, `IGNORE_INDEX = 255` | $0 = \text{background}$, $1 \dots 80 = \text{classi Thing}$ |
| **Peso Ramo Ausiliario** | `AUX_LOSS_WEIGHT = 0.5` | Supervisione intermedia su `layer3` della FCN |

---

## 3. Risultati dell'Ispezione Strutturale di COCO 2017 (`Sezione 4`)

La funzione `inspect_raw_coco_directory(DATA_ROOT)` ha accertato l'esatta organizzazione del dataset su disco:

* **Contenuto della directory radice:** `['annotations', 'test2017', 'train2017', 'val2017']`
* **Cartelle maschere semantiche pre-renderizzate:** `NESSUNA`
* **Cardinalità ed estensioni reali degli split immagine:**

| Split | Numero di File | Estensioni Rilevate | Utilizzo Progettuale |
| :--- | :---: | :---: | :--- |
| **`train2017`** | $118\,287$ | `{'.jpg': 118287}` ($100\%$ JPEG) | Addestramento principale ed estrazione subset bilanciato |
| **`val2017`** | $5\,000$ | `{'.jpg': 5000}` ($100\%$ JPEG) | Validazione deterministica integrale (*zero data leakage*) |
| **`test2017`** | $40\,670$ | `{'.jpg': 40670}` ($100\%$ JPEG) | Test cieco / inferenza qualitativa |

* **File di annotazione rilevati in `annotations/` ($6$ file JSON):**
  * `instances_train2017.json` ($448.02\text{ MB}$) e `instances_val2017.json` ($19.06\text{ MB}$) — impiegati per la rasterizzazione delle maschere semantiche.
  * `captions_train2017.json` ($87.61\text{ MB}$), `captions_val2017.json` ($3.69\text{ MB}$), `person_keypoints_train2017.json` ($227.82\text{ MB}$), `person_keypoints_val2017.json` ($9.56\text{ MB}$).

---

## 4. Profilazione Statistica su Campione Grezzo (`Sezione 5`)

L'analisi su $500$ immagini campionate deterministicamente da `val2017` (`random.Random(42)`) ha evidenziato tre caratteristiche strutturali critiche per la progettazione dello Step 2:

| Indicatore Misurato ($n = 500$ immagini `val2017`) | Valore Empirico | Implicazione Diretta per lo Step 2 |
| :--- | :--- | :--- |
| **Modalità colore (`PIL mode`)** | `{'RGB': 498, 'L': 2}` ($0.40\%$ scala di grigi) | Conversione obbligatoria `.convert("RGB")` per evitare tensori a $1$ canale |
| **Risoluzioni distinte** | $133$ risoluzioni diverse su $500$ immagini | Impossibilità di impilare tensori in batch senza cropping/resize sincronizzato |
| **Altezza $H$ (`min / med / max`)** | $230\text{ px}$ / $480\text{ px}$ / $640\text{ px}$ | Esistono immagini con $H < 256\text{ px}$ ($H_{\min} = 230\text{ px}$) |
| **Larghezza $W$ (`min / med / max`)** | $246\text{ px}$ / $640\text{ px}$ / $640\text{ px}$ | Esistono immagini con $W < 256\text{ px}$ ($W_{\min} = 246\text{ px}$) |

---

## 5. Checklist di Validazione dello Step 1

| Controllo Diagnostico | Stato | Esito Verificato |
| :--- | :---: | :--- |
| Rilevamento acceleratore CUDA multi-GPU | **SUPERATO** | $2 \times \text{Tesla T4}$ attive (`PyTorch 2.11.0+cu130`) |
| Creazione directory persistenti e helper grafici | **SUPERATO** | `/kaggle/working/figures` e `/kaggle/working/checkpoints` pronte |
| Blocco seed deterministico globale | **SUPERATO** | `SEED = 42` propagato a Python, NumPy, PyTorch e cuDNN |
| Verifica struttura directory, estensioni e JSON | **SUPERATO** | $118\,287$ `.jpg` in train, $5\,000$ `.jpg` in val, maschere via JSON |
| Profilazione risoluzioni e modalità colore grezze | **SUPERATO** | Rilevate immagini `'L'` ($2/500$) e lati minimi $< 256\text{ px}$ ($230 \times 246$) |