# \[03\_REPORT\] Step 3 — Architettura `FCN-ResNet50` *from scratch*, *Babysitting* e Diagnostica Avanzata

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Riferimento Pianificazione:** `[00_PLANNING] Scaletta Progetto CV` — Step 3 e Step 3-BIS  
**Allineamento Scheletro Kaggle Notebook:** Celle 4 e 5 — Sezioni 9, 10, 11 e 11-BIS (*GPU/Batch Calibration, FCN From Scratch & Shared Init Checkpoint, Analytical Sanity Checks, Controlled Overfitting & Mini-Ablation*)  

---

## 1. Inquadramento Teorico e Obiettivi Progettuali

In conformità con le specifiche progettuali (`H3.pdf`) e con i principi metodologici del corso (*CS231n — Babysitting the Learning Process*), prima di avviare confronti computazionalmente onerosi su migliaia di immagini è necessario verificare l'integrità matematica dell'architettura, la stabilità del flusso dei gradienti e il dimensionamento hardware.

Lo **Step 3**, implementato nelle **Celle 4 e 5** del notebook rifattorizzato, persegue quattro obiettivi strutturali:

1. **Calibrazione Hardware, `BATCH_SIZE` Operativo e Persistenza Pesi Iniziali (`Cella 4 — Sezione 9`):** Misurazione di VRAM picco e tempo per step sulle $2$ GPU NVIDIA Tesla T4 (`DataParallel=True`), selezione di `BATCH_SIZE = 16`, istanziazione di `fcn_resnet50` *from scratch* (`weights=None`, `weights_backbone=None`, `num_classes=81`, `aux_loss=True`) e salvataggio su disco dello stato iniziale comune (`shared_init_fcn_resnet50.pt`).
2. **Verifiche Analitiche e Diagnostica dei Gradienti (`Cella 5 — Sezione 10`):**
   * *Sanity Check 1 (Step $0$):* Verifica che a pesi casuali la *Cross-Entropy Loss* iniziale sul ramo principale rispetti il valore teorico $-\ln(1/81) = 4.3944$ ($6.5917$ includendo il ramo ausiliario pesato a $0.5$).
   * *Sanity Check 2 (Step $1$):* Misura del rapporto *Update-to-Weight* $\lVert \Delta W \rVert_2 / \lVert W \rVert_2$ su `backbone.conv1`, `backbone.layer4` e `classifier.4` rispetto all'ordine di grandezza teorico $\sim 10^{-3}$.
3. **Test di Overfitting Controllato (`Cella 5 — Sezione 11`):** Addestramento su $16$ immagini deterministiche (`CenterCrop(256, 256)`, `weight_decay=0.0`, $80$ epoche) con monitoraggio *batch-wise* ed esportazione delle curve (`04_step3_babysitting_loss_metrics_curves.png` / `.svg`) e della griglia qualitativa (`05_step3_babysitting_overfit_predictions.png`).
4. **Mini-Ablation Diagnostica (`Cella 5 — Sezione 11-BIS`):** Confronto controllato a partire da `SHARED_INIT_WEIGHTS` tra Baseline (`T1`), `CosineAnnealingLR` (`T2`) e risoluzione $384 \times 384$ (`T3`) valutati deterministicamente in `model.eval()` su tutte le $29$ classi oggetto presenti (`06_step3bis_mini_ablation_visual_comparison.png`).

---

## 2. Risultati della Calibrazione Hardware e dei *Sanity Check* (`Celle 4 e 5`)

### 2.1 Calibrazione GPU e `BATCH_SIZE` (`2 x Tesla T4`, `DataParallel=True`)

| `BATCH_SIZE` | Esito | VRAM Picco Allocata | Tempo per Step | Scelta Progettuale |
| :---: | :---: | :---: | :---: | :--- |
| $8$ | `OK` | $3.25\text{ GB}$ | $682.0\text{ ms}$ | Sottoutilizzo multi-GPU ($4$ immagini per GPU) |
| $16$ | `OK` | $5.17\text{ GB}$ | $1152.3\text{ ms}$ | **Selezionato come `BATCH_SIZE` operativo** ($8$ img/GPU) |
| $32$ | `OK` | $9.00\text{ GB}$ | $2226.6\text{ ms}$ | Dimezza il numero di aggiornamenti per epoca |

### 2.2 Parametri del Modello, Checkpoint Iniziale e Verifiche al Passo $0$ e Passo $1$

* **Parametri totali e addestrabili (`FCN-ResNet50`):** $35\,368\,418$ ($100\%$ addestrabili)
* **Checkpoint pesi iniziali salvato:** `/kaggle/working/checkpoints/shared_init_fcn_resnet50.pt`
* **Loss Iniziale (Step $0$):**
  * **Main Loss:** $4.4173$ misurata vs $4.3944$ teorica ($-\ln(1/81)$, scostamento $+0.52\%$ dovuto alla disuguaglianza di Jensen su *logit* casuali a varianza non nulla)
  * **Total Loss (`main + 0.5 * aux`):** $6.5704$ misurata vs $6.5917$ teorica ($-0.32\%$)
* **Rapporto *Update-to-Weight* al Passo $1$ (`AdamW`, $\eta = 10^{-3}$):**
  * `backbone.conv1 (input)`: $1.34 \times 10^{-2}$
  * `backbone.layer4 (deep)`: $4.87 \times 10^{-6}$
  * `classifier.4 (head)`: $1.16 \times 10^{-3}$ (allineato al target teorico $\sim 10^{-3}$)

### 2.3 Convergenza del Test di Overfitting su $16$ Immagini ($80$ Epoche, $109.3\text{ s}$)

| Epoca | Loss (`Main + 0.5 Aux`) | Pixel Accuracy (Train) | mIoU Classi Presenti (Train) |
| :---: | :---: | :---: | :---: |
| $01 / 80$ | $6.6100$ | $6.90\%$ | $0.53\%$ |
| $10 / 80$ | $1.5603$ | $75.66\%$ | $11.37\%$ |
| $20 / 80$ | $0.5917$ | $91.02\%$ | $30.64\%$ |
| $30 / 80$ | $0.2487$ | $95.68\%$ | $46.79\%$ |
| $40 / 80$ | $0.1573$ | $96.98\%$ | $59.04\%$ |
| $50 / 80$ | $0.1133$ | $97.82\%$ | $66.66\%$ |
| $60 / 80$ | $0.0884$ | $98.26\%$ | $71.88\%$ |
| $70 / 80$ | $0.0840$ | $98.12\%$ | $74.99\%$ |
| $80 / 80$ | $0.0680$ | $98.62\%$ | $79.13\%$ |

---

## 3. Guida alla Lettura dei Grafici e delle Predizioni Visive

1. **Curva Loss vs Accuracy/mIoU (`04_step3_babysitting_loss_metrics_curves.png`):**
   * Il **pannello sinistro** mostra una discesa regolare e monotona della *Cross-Entropy Loss* da $6.6100$ a $0.0680$ ($-99.0\%$), confermando l'assenza di esplosione o scomparsa dei gradienti.
   * Il **pannello destro** evidenzia la differenza strutturale tra `Pixel Accuracy` (che supera il $95\%$ già all'epoca $30$ per effetto del `background` dominante) e `mIoU Classi Presenti` (che sale progressivamente fino al **$79.13\%$** all'epoca $80$), dimostrando perché la `mIoU` è la metrica primaria di valutazione.
2. **Griglia Qualitativa $4 \times 3$ (`05_step3_babysitting_overfit_predictions.png`):**
   * Le macro-sagome (`car`, `dog`, `surfboard`, `bowl`, `broccoli`, `airplane`) sono ricostruite con notevole fedeltà ai contorni.
   * La mandria distante nella Riga $1$ (`cow` in magenta) viene segmentata solo in minima parte: ciò fornisce la prova visiva diretta del limite fisico dell'*output stride* $s = 8$ della FCN (mappa latente $32 \times 32$, in cui ogni cella corrisponde a $8 \times 8\text{ px}$).

---

## 4. Risultati e Analisi Critica della Mini-Ablation Diagnostica (Step 3-BIS)

Valutazione deterministica in `model.eval()` su tutte le $29$ classi oggetto presenti nel mini-subset a partire da `SHARED_INIT_WEIGHTS`:

| Configurazione Tentativo | Loss Finale | `PixAcc` (`eval`) | `mIoU All` | `mIoU Obj` | `Obj Rilevati` | Tempo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`T1`: Baseline ($256\text{ px}$, $80\text{ ep}$, LR fisso)** | $0.0664$ | **$91.94\%$** | **$43.98\%$** | **$42.38\%$** | $18 / 29$ | $106.1\text{ s}$ |
| **`T2`: Scheduler ($256\text{ px}$, $150\text{ ep}$, `CosineLR`)** | $0.0597$ | $85.15\%$ | $39.94\%$ | $38.47\%$ | **$19 / 29$** | $199.0\text{ s}$ |
| **`T3`: Risoluzione ($384\text{ px}$, $150\text{ ep}$, `CosineLR`)** | $0.0474$ | $87.45\%$ | $37.72\%$ | $36.01\%$ | $18 / 29$ | $455.8\text{ s}$ |

### Tre Lezioni Tecniche Estratte dallo Step 3-BIS e da `06_step3bis_mini_ablation_visual_comparison.png`

1. **Ruolo della metrica *batch-wise* vs `model.eval()` con `BatchNorm2d` su `DataParallel`:**  
   La metrica *batch-wise* (`compute_batch_miou_and_acc`) è un ottimo termometro relativo di convergenza durante il *babysitting* (`79.13%`), ma nello Step 3-BIS e nello Step 4 è indispensabile l'accumulatore globale in `model.eval()`. Prolungare il training a $150$ epoche senza regolarizzazione su sole $16$ immagini (`T2`) abbassa ulteriormente la *training loss* (`0.0597`) ma introduce una lieve deriva nelle *running statistics* di `BatchNorm2d` (aggiornate su micro-batch da $4$ immagini/GPU anche quando `CosineAnnealingLR` ha quasi azzerato il *learning rate*). Nel training generalizzato dello Step 4 ($12\,000$ immagini, `BATCH_SIZE = 16`, `weight_decay = 1e-4`), `CosineAnnealingLR` stabilizza invece la convergenza senza deriva.
2. **Compromesso Risoluzione vs Campo Recettivo (`T1`/`T2` vs `T3`):**  
   In `06_step3bis_mini_ablation_visual_comparison.png`, `T3` ($384\text{ px}$, griglia $48 \times 48$) recupera effettivamente una porzione maggiore delle mucche distanti nella Riga $1$, ma frammenta la fusoliera dell'aereo nella Riga $4$ (buchi interni dovuti al campo recettivo fisso della ResNet-50), fa scendere la `mIoU Obj` al $36.01\%$ e moltiplica il tempo di calcolo per **$4.30\times$** ($455.8\text{ s}$ vs $106.1\text{ s}$).
3. **Ponte verso lo Step 4 (*Ceteris Paribus*):**  
   Confermata la configurazione ottimale (`FCN-ResNet50` *from scratch*, `256 x 256`, `BATCH_SIZE = 16`, `AdamW` + `CosineAnnealingLR`, partenza da `shared_init_fcn_resnet50.pt`), il recupero degli oggetti piccoli e rari viene affrontato nello **Step 4** confrontando **`Cross-Entropy`**, **`Focal Loss`**, **`Dice Loss`**, **`Jaccard Loss`** e **`0.5 * CE + 0.5 * Dice`** a costo zero di VRAM aggiuntiva.