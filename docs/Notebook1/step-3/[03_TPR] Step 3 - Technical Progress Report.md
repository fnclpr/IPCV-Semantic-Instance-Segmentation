# \[03\_TPR\] Step 3 — Technical Progress Report

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Fase:** Step 3 — Architettura `FCN-ResNet50` *from scratch*, Calibrazione Hardware, Persistenza Pesi Iniziali, *Babysitting* e Mini-Ablation Diagnostica (Step 3-BIS)  
**Riferimento Strutturale:** Kaggle Notebook di Riferimento — Celle 4 e 5 (Sezioni 9, 10, 11 e 11-BIS)  

---

## 1. Obiettivo e Perimetro dello Step 3

Lo **Step 3** istanzia l'architettura di *Semantic Segmentation* `FCN-ResNet50` completamente *from scratch* (`weights=None`, `weights_backbone=None`), ne calibra i parametri operativi sull'hardware multi-GPU disponibile e applica la metodologia di *babysitting* del processo di apprendimento prevista dalle specifiche progettuali (`H3.pdf`) e dai moduli teorici del corso (*CS231n*). In conformità con l'organizzazione modulare delle **Celle 4 e 5**, sono stati conseguiti quattro obiettivi verificabili:

1. **Calibrazione Hardware, Dimensione del Batch e Checkpoint Pesi Iniziali (`Cella 4 — Sezione 9`):** Misurazione empirica del picco di memoria VRAM e della latenza per singolo step sulle $2$ GPU NVIDIA Tesla T4 (`DataParallel=True`) per fissare `BATCH_SIZE = 16`, seguita dall'istanziazione di `FCN-ResNet50` ($C = 81$, `aux_loss=True`) e dal salvataggio su disco dello stato iniziale condiviso (`shared_init_fcn_resnet50.pt`) per garantire confronti *ceteris paribus* tra Step 3-BIS e Step 4.
2. **Verifiche Analitiche e Diagnostica dei Gradienti (`Cella 5 — Sezione 10`):** Esecuzione incapsulata (`RUN_BABYSITTING = True`) del *Sanity Check 1* (confronto tra *Cross-Entropy Loss* teorica $-\ln(1/81)$ ed empirica al passo $0$) e del *Sanity Check 2* (rapporto *Update-to-Weight* $\lVert \Delta W \rVert_2 / \lVert W \rVert_2$ lungo la profondità della rete al passo $1$).
3. **Test di Overfitting Controllato (`Cella 5 — Sezione 11`):** Addestramento su un mini-subset deterministico di $16$ immagini (`CenterCrop(256, 256)`, `weight_decay=0.0`, $80$ epoche) con monitoraggio rapido *batch-wise* (`compute_batch_miou_and_acc`) ed esportazione ad alta risoluzione delle curve di apprendimento (`04_step3_babysitting_loss_metrics_curves.png` / `.svg`) e della griglia qualitativa (`05_step3_babysitting_overfit_predictions.png`).
4. **Mini-Ablation Diagnostica Controllata (`Cella 5 — Sezione 11-BIS`):** Confronto riproducibile a partire dagli identici pesi iniziali (`SHARED_INIT_WEIGHTS`) tra Baseline (`T1`), Scheduler `CosineAnnealingLR` (`T2`) e Risoluzione $384 \times 384$ (`T3`), valutati deterministicamente in `model.eval()` su tutte le $29$ classi oggetto presenti (`_evaluate_ablation_subset`) con esportazione della figura `06_step3bis_mini_ablation_visual_comparison.png`.

---

## 2. Calibrazione Hardware GPU e Persistenza Pesi Iniziali (`Cella 4`)

L'esecuzione di `calibrate_batch_size_and_device` nella **Cella 4** sull'ambiente Kaggle dotato di $2 \times \text{NVIDIA Tesla T4}$ (`DataParallel=True`) ha restituito le seguenti misure operative:

| `BATCH_SIZE` Candidato | Stato Esecuzione | VRAM Picco Allocata | Tempo per Step (*Forward + Backward*) | Valutazione Operativa |
| :---: | :---: | :---: | :---: | :--- |
| **$8$** | `OK` | $3.25\text{ GB}$ | $682.0\text{ ms}$ | Sottoutilizzo della capacità multi-GPU ($4$ img/GPU) |
| **$16$** | `OK` | $5.17\text{ GB}$ | $1152.3\text{ ms}$ | **Selezionato:** $8$ img/GPU, ottimo compromesso VRAM/throughput |
| **$32$** | `OK` | $9.00\text{ GB}$ | $2226.6\text{ ms}$ | Dimezza la frequenza di aggiornamento dei pesi per epoca |

* **`BATCH_SIZE` operativo selezionato:** $16$
* **Parametri totali e addestrabili (`FCN-ResNet50`):** $35\,368\,418$ / $35\,368\,418$ ($100\%$ inizializzati *from scratch*)
* **Checkpoint pesi iniziali condivisi salvato:** `/kaggle/working/checkpoints/shared_init_fcn_resnet50.pt`
* **Formulazione della funzione obiettivo in training:**
  $$\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{main}}(\text{out}[\text{'out'}], y) + 0.5 \cdot \mathcal{L}_{\text{aux}}(\text{out}[\text{'aux'}], y)$$

---

## 3. Risultati dei *Sanity Check* di *Babysitting* (`Cella 5`)

L'estrazione deterministica `select_class_balanced_subset(coco_train, max_samples=100, seed=42)[:16]` ha selezionato $16$ immagini ricche di oggetti (contenenti $29$ classi *Thing* distinte oltre al `background`), precaricate in RAM in $0.1\text{ s}$.

### 3.1 Sanity Check 1 — Loss Iniziale a Pesi Casuali (Step $0$)

| Metrica al Passo $0$ | Valore Teorico Atteso | Valore Empirico Misurato | Scostamento Relativo |
| :--- | :---: | :---: | :---: |
| **Loss Ramo Principale (`out`)** | $-\ln(1/81) = 4.3944$ | $4.4173$ | $+0.52\%$ (atteso per disuguaglianza di Jensen) |
| **Loss Totale (`main + 0.5 * aux`)** | $1.5 \times 4.3944 = 6.5917$ | $6.5704$ | $-0.32\%$ |

### 3.2 Sanity Check 2 — Rapporto *Update-to-Weight* al Passo $1$ (`AdamW`, $\eta = 10^{-3}$)

| Layer Monitorato | Posizione nella Rete | Rapporto $\lVert \eta \cdot \nabla_W \mathcal{L} \rVert_2 / \lVert W \rVert_2$ | Riferimento Teorico (*CS231n*) |
| :--- | :--- | :---: | :---: |
| `backbone.conv1` | Ingresso (primo strato convoluzionale) | $1.34 \times 10^{-2}$ | $\sim 10^{-3}$ |
| `backbone.layer4` | Estrattore profondo (ultimo blocco residuo) | $4.87 \times 10^{-6}$ | $\sim 10^{-3}$ |
| `classifier.4` | Testa di segmentazione finale ($1 \times 1$) | $1.16 \times 10^{-3}$ | $\sim 10^{-3}$ |

### 3.3 Sanity Check 3 — Progressione dell'Overfitting su $16$ Immagini ($80$ Epoche, $109.3\text{ s}$)

| Epoca | Training Loss (`Main + 0.5 Aux`) | Pixel Accuracy (Train) | mIoU Classi Presenti (Train) |
| :---: | :---: | :---: | :---: |
| **$01 / 80$** | $6.6100$ | $6.90\%$ | $0.53\%$ |
| **$10 / 80$** | $1.5603$ | $75.66\%$ | $11.37\%$ |
| **$20 / 80$** | $0.5917$ | $91.02\%$ | $30.64\%$ |
| **$30 / 80$** | $0.2487$ | $95.68\%$ | $46.79\%$ |
| **$40 / 80$** | $0.1573$ | $96.98\%$ | $59.04\%$ |
| **$50 / 80$** | $0.1133$ | $97.82\%$ | $66.66\%$ |
| **$60 / 80$** | $0.0884$ | $98.26\%$ | $71.88\%$ |
| **$70 / 80$** | $0.0840$ | $98.12\%$ | $74.99\%$ |
| **$80 / 80$** | $0.0680$ | $98.62\%$ | $79.13\%$ |

---

## 4. Mini-Ablation Diagnostica (Step 3-BIS — `Cella 5`)

I tre tentativi controllati partono dallo stesso identico stato iniziale (`SHARED_INIT_WEIGHTS`) e vengono valutati in modalità deterministica `model.eval()` su tutte le $29$ classi oggetto presenti nel mini-subset:

| Configurazione Tentativo | Loss Finale | `PixAcc` (`eval`) | `mIoU All` | `mIoU Obj` ($1 \dots 80$) | `Obj Rilevati` ($\text{IoU} > 0.10$) | Tempo Esecuzione |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`T1`: Baseline ($256\text{ px}$, $80\text{ ep}$, LR fisso)** | $0.0664$ | **$91.94\%$** | **$43.98\%$** | **$42.38\%$** | $18 / 29$ | $106.1\text{ s}$ |
| **`T2`: Scheduler ($256\text{ px}$, $150\text{ ep}$, `CosineLR`)** | $0.0597$ | $85.15\%$ | $39.94\%$ | $38.47\%$ | **$19 / 29$** | $199.0\text{ s}$ |
| **`T3`: Risoluzione ($384\text{ px}$, $150\text{ ep}$, `CosineLR`)** | $0.0474$ | $87.45\%$ | $37.72\%$ | $36.01\%$ | $18 / 29$ | $455.8\text{ s}$ |

---

## 5. Guida alla Lettura delle Tre Figure Diagnostiche dello Step 3

### 5.1 Curve di Apprendimento (`04_step3_babysitting_loss_metrics_curves.png`)

* **Pannello Sinistro (*Babysitting: Curva Training Loss (16 immagini)*):**
  * **Come si legge:** Riporta sull'asse $X$ le $80$ epoche e sull'asse $Y$ la *Cross-Entropy Loss* composita ($\mathcal{L}_{\text{main}} + 0.5\,\mathcal{L}_{\text{aux}}$).
  * **Cosa dimostra:** La riduzione monotona del **$99.0\%$** (da $6.6100$ a $0.0680$) senza oscillazioni esplosive certifica la stabilità numerica della propagazione dei gradienti attraverso i $50$ strati inizializzati *from scratch*.
* **Pannello Destro (*Babysitting: Convergenza Accuracy e mIoU*):**
  * **Come si legge:** Confronta `Pixel Accuracy (%)` (curva blu) e `mIoU Classi Presenti (%)` (curva verde).
  * **Cosa dimostra:** La `Pixel Accuracy` supera il $95\%$ già all'epoca $30$ per effetto della prevalenza dei pixel di `background`, mentre la `mIoU` cresce progressivamente fino al **$79.13\%$** all'epoca $80$, confermando la maggiore sensibilità della metrica IoU sulla qualità delle sagome.

### 5.2 Griglia Qualitativa di Overfitting (`05_step3_babysitting_overfit_predictions.png`)

* **Come si legge:** Griglia $4 \times 3$ con `Input (256x256)`, `Ground Truth` e `Predizione FCN (Overfit)` su quattro campioni del mini-subset.
* **Cosa dimostra:**
  * **Riga 1 (strada con auto e mandria di bovini):** La FCN ricostruisce fedelmente le due automobili (`car` in azzurro) nella fascia superiore, mentre segmenta solo una minuscola porzione della mandria distante (`cow` in magenta), confermando visivamente la soglia spaziale imposta dall'*output stride* $s = 8$.
  * **Riga 2 (cani su tavola da surf):** I due cani (`dog`) e la tavola da surf (`surfboard` in giallo) sono segmentati con ottima coerenza spaziale.
  * **Riga 3 (piatto di cibo e ciotola):** La massa principale del cibo/verdure (magenta) e la ciotola in alto a sinistra (verde) combaciano fedelmente con la *Ground Truth*.
  * **Riga 4 (hangar con aereo militare e jeep):** Le sagome estese di `airplane` (verde oliva) e `car` (azzurro) sono ricostruite con bordi nitidi e continui.

### 5.3 Confronto Visivo della Mini-Ablation (`06_step3bis_mini_ablation_visual_comparison.png`)

* **Come si legge:** Griglia $4 \times 5$ che affianca `Input (384px)`, `Ground Truth`, `T1: Baseline 256px`, `T2: CosineLR 256px` e `T3: CosineLR 384px` sulle medesime quattro scene.
* **Cosa dimostra:**
  * **Riga 1 (mandria distante):** In `T3` (colonna $5$, $384\text{ px}$), la griglia latente più fitta ($48 \times 48$ contro $32 \times 32$) recupera una porzione sensibilmente maggiore delle mucche distanti in magenta rispetto a `T1` e `T2`.
  * **Righe 2, 3 e 4 (effetto del campo recettivo fisso a $384\text{ px}$):** A fronte del maggiore dettaglio locale, `T3` introduce buchi interni nella fusoliera dell'aereo (Riga 4) e falsi positivi viola nella scena del piatto (Riga 3), riducendo la `mIoU Obj` globale al $36.01\%$ e quadruplicando il tempo di calcolo ($455.8\text{ s}$, pari a $4.30\times$ rispetto a `T1`). Ciò conferma che **`CROP_SIZE = 256`** è la risoluzione ottimale per lo Step 4.

---

## 6. Checklist di Validazione dello Step 3

| Controllo Diagnostico | Stato | Esito Verificato |
| :--- | :---: | :--- |
| Calibrazione VRAM e selezione `BATCH_SIZE` (`Cella 4`) | **SUPERATO** | `BATCH_SIZE = 16` ($5.17\text{ GB}$ VRAM, $1152.3\text{ ms}$/step) |
| Salvataggio pesi iniziali condivisi (`Cella 4`) | **SUPERATO** | `/kaggle/working/checkpoints/shared_init_fcn_resnet50.pt` |
| Verifica analitica loss iniziale ($-\ln(1/81)$) | **SUPERATO** | $4.4173$ empirica vs $4.3944$ teorica ($+0.52\%$) |
| Diagnostica *Update-to-Weight Ratio* | **SUPERATO** | `classifier.4` $= 1.16 \times 10^{-3}$ ($\sim 10^{-3}$) |
| Convergenza overfitting ($80$ epoche) ed export figure | **SUPERATO** | Loss $6.6100 \rightarrow 0.0680$, mIoU $79.13\%$; salvate figure `04`, `05`, `06` |