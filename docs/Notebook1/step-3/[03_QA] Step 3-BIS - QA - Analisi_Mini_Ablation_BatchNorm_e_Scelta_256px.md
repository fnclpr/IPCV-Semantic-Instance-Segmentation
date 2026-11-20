# \[03\_QA\] Step 3-BIS — Analisi Critica della Mini-Ablation, Comportamento di `BatchNorm2d` su `DataParallel` e Conferma di `CROP_SIZE = 256`

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Riferimento Codice:** `Notebook1_SetupDatasetCOCO2017_Babysitting_(Step1_Step3-BIS)` — Cella 5 (`Sezione 11-BIS`)  
**Collegamento Progettuale:** Step 3 (*Babysitting*) $\rightarrow$ Step 3-BIS (`run_step3bis_mini_ablation`) $\rightarrow$ Step 4 (Studio Comparativo Loss Functions *Ceteris Paribus*)  
**Tipo di Documento:** Scheda di Approfondimento Teorico-Metodologico (Q&A)

---

## 1. Quesiti Metodologici Formalizzati

La mini-ablation eseguita nella **Cella 5** (`Sezione 11-BIS`) ha prodotto risultati ricchi di implicazioni fisiche e statistiche. Questa scheda risponde a cinque domande chiave che emergono direttamente dalla lettura della tabella e della **Figura `06_step3bis_mini_ablation_visual_comparison.png`**:

1. **Paradosso apparente Step 3 vs Step 3-BIS (`79.13%` vs `43.98%`):** *Perché nello Step 3 la configurazione baseline registrava `Pixel Acc = 98.62%` e `mIoU = 79.13%`, mentre nello Step 3-BIS la medesima configurazione `T1` registra `PixAcc = 91.94%`, `mIoU All = 43.98%` e `mIoU Obj = 42.38%`?*
2. **Confronto `T1` vs `T2` (`CosineAnnealingLR` + $150$ epoche):** *Perché `T2` ottiene una training loss inferiore (`0.0597` vs `0.0664`) e rileva più classi oggetto (`19/29` vs `18/29`), ma mostra una `mIoU Obj` leggermente inferiore (`38.47%` vs `42.38%`) in `model.eval()` su questo mini-subset?*
3. **Confronto `T2` vs `T3` (Aumento di risoluzione $256\text{ px} \rightarrow 384\text{ px}$):** *Perché passando a $384 \times 384$ il tempo di esecuzione esplode di $4.30\times$ ($106.1\text{ s} \rightarrow 455.8\text{ s}$), mentre nella Figura `06` la mandria distante migliora ma compaiono buchi interni nella fusoliera dell'aereo?*
4. **Perché `CosineAnnealingLR` resta valido per lo Step 4:** *Nonostante la deriva su $16$ immagini in `T2`, perché `CosineAnnealingLR` è superiore a `StepLR` e `ReduceLROnPlateau` quando passiamo al dataset bilanciato da $12\,000$ immagini?*
5. **Sintesi Strategica per lo Step 4:** *In che modo le evidenze dello Step 3-BIS giustificano la conferma di `CROP_SIZE = 256` e spostano l'intervento sugli oggetti piccoli e rari alle funzioni di perdita (`Focal`, `Dice`, `Jaccard`, `CE + Dice`)?*

---

## 2. Riepilogo Quantitativo dei Tre Esperimenti Controllati (`Cella 5`)

Tutti e tre i tentativi partono dallo **stesso identico checkpoint iniziale** (`SHARED_INIT_WEIGHTS`) e vengono valutati in modalità deterministica **`model.eval()`** tramite accumulatore globale di intersezione e unione (`_evaluate_ablation_subset`) su tutte le $29$ classi *Thing* presenti nel mini-subset (più il `background` per `mIoU All`):

| Configurazione Tentativo | Loss Finale (`train`) | `PixAcc` (`eval`) | `mIoU All` ($0 \dots 80$) | `mIoU Obj` ($1 \dots 80$) | `Obj Rilevati` ($\text{IoU} > 0.10$) | Tempo Totale | Fattore Tempo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`T1`: Baseline ($256\text{ px}$, $80\text{ ep}$, LR fisso)** | $0.0664$ | **$91.94\%$** | **$43.98\%$** | **$42.38\%$** | $18 / 29$ | $106.1\text{ s}$ | $1.00\times$ |
| **`T2`: Scheduler ($256\text{ px}$, $150\text{ ep}$, `CosineLR`)** | $0.0597$ | $85.15\%$ | $39.94\%$ | $38.47\%$ | **$19 / 29$** | $199.0\text{ s}$ | $1.88\times$ |
| **`T3`: Risoluzione ($384\text{ px}$, $150\text{ ep}$, `CosineLR`)** | $0.0474$ | $87.45\%$ | $37.72\%$ | $36.01\%$ | $18 / 29$ | $455.8\text{ s}$ | $4.30\times$ |

---

## 3. Spiegazione del Divario `model.train()` (Step 3) vs `model.eval()` (Step 3-BIS)

Il confronto tra i numeri dell'Epoca $80$ dello Step 3 (`PixAcc = 98.62%`, `mIoU = 79.13%`) e quelli della riga `T1` dello Step 3-BIS (`PixAcc = 91.94%`, `mIoU All = 43.98%`) non indica un'incongruenza del codice, bensì **due differenze metodologiche fondamentali**:

### 3.1 Aggregazione *Batch-Wise* vs Accumulatore Globale su $30$ Classi

1. **Nel *babysitting* (Step 3):** `compute_batch_miou_and_acc` calcola la media dell'IoU separatamente su ciascuno dei $2$ mini-batch da $8$ immagini, mediando solo sulle poche classi presenti nel singolo batch (dove il `background` pesa molto nella media).
2. **Nello Step 3-BIS (`_evaluate_ablation_subset`):** I pixel di intersezione e unione vengono accumulati **globalmente sull'intero set di $16$ immagini** prima di calcolare l'IoU di ciascuna delle $30$ classi presenti ($29$ oggetti + `background`). Le classi presenti in una sola immagine con pochi pixel (es. una piccola auto sullo sfondo o una parte di cibo) hanno esattamente lo stesso peso $1/30$ del `background` nella media aritmetica `mIoU All` (e $1/29$ in `mIoU Obj`).

### 3.2 Effetto di `BatchNorm2d` sotto `nn.DataParallel` su Micro-Batch ($4$ Immagini/GPU)

* Durante il training (`model.train()`), ogni layer `nn.BatchNorm2d` normalizza le attivazioni usando **media e varianza esatte del mini-batch corrente**.
* Quando si passa in **`model.eval()`**, `BatchNorm2d` sostituisce le statistiche del batch con le medie e varianze mobili storiche (`running_mean`, `running_var`), aggiornate per default secondo la legge esponenziale:
  $$
  \hat{\mu}_{\text{running}} \leftarrow (1 - 0.1)\,\hat{\mu}_{\text{running}} + 0.1\,\mu_{\mathcal{B}}
  $$
* Sotto `nn.DataParallel` su $2$ GPU Tesla T4 con `batch_size = 8`, **ogni singola GPU elabora appena $4$ immagini per batch** (e solo $2$ batch per epoca). Su un modello inizializzato *from scratch* i cui pesi cambiano rapidamente, le statistiche mobili accumulano un inevitabile ritardo rispetto ai pesi correnti (*train-eval covariate shift*).
* **Perché questo problema scompare nello Step 4:** Nello Step 4 lavoreremo con `BATCH_SIZE = 16` ($8$ immagini per GPU) su **$12\,000$ immagini** ($750$ batch/epoca), consentendo a `running_mean` e `running_var` di convergere stabilmente verso la distribuzione reale del dataset.

---

## 4. Lettura Critica di `T1`, `T2` e `T3` e della Figura `06`

### 4.1 Perché `T2` (`150` Epoche + `CosineAnnealingLR`) Scende in `eval()` sul Micro-Subset

In `T2`, la *training loss* scende ulteriormente rispetto a `T1` (`0.0597` vs `0.0664`) e sale a **`19/29`** il numero di classi oggetto rilevate ($\text{IoU} > 0.10$), ma la `mIoU Obj` in `model.eval()` scende dal `42.38%` al `38.47%`:

* Prolungando il training a $150$ epoche senza regolarizzazione (`weight_decay = 0.0`) su sole $16$ immagini, nelle epoche finali in cui `CosineAnnealingLR` riduce il *learning rate* verso $\eta_{\min} = 10^{-5}$, i pesi convoluzionali sono ormai quasi fermi, ma i layer `BatchNorm2d` continuano ad aggiornare `running_mean` e `running_var` con `momentum = 0.1` fisso su micro-batch da $4$ immagini/GPU, accentuando l'over-memorizzazione di training.
* **Perché confermiamo `CosineAnnealingLR` per lo Step 4:** Sul dataset generalizzato da $12\,000$ immagini (`weight_decay = 1e-4`), la legge del coseno:
  $$
  \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)
  $$
  mantiene il *learning rate* alto nella prima metà per formare i filtri *from scratch* e lo accompagna dolcemente verso $\eta_{\min} = 10^{-5}$ nel finale, evitando sia le oscillazioni del LR costante sia le discontinuità brusche di `StepLR` e la dipendenza da soglie di pazienza di `ReduceLROnPlateau`.

### 4.2 Il Compromesso Fisico di `T3` ($384 \times 384$): Griglia Latente vs Campo Recettivo

L'analisi della **Figura `06_step3bis_mini_ablation_visual_comparison.png`** (colonna $5$, `T3: CosineLR 384px`) offre una dimostrazione visiva esemplare di due fenomeni opposti della visione artificiale:

1. **Guadagno locale sugli oggetti piccoli (Riga $1$ — mandria distante):**  
   Portando l'ingresso da $256 \times 256$ a $384 \times 384$, la griglia latente della FCN (*output stride* $s = 8$) cresce da $32 \times 32$ ($1\,024$ celle) a **$48 \times 48$ ($2\,304$ celle, $+125\%$ di risoluzione spaziale)**. Nella Riga $1$, `T3` recupera una porzione sensibilmente maggiore delle mucche distanti in magenta rispetto a `T1` e `T2`.
2. **Degrado globale sugli oggetti estesi e costo computazionale (Righe $3$ e $4$):**  
   * **Campo recettivo fisso:** Il campo recettivo effettivo della `ResNet-50` dipende dall'architettura ed è quindi **fisso in pixel**. Ingrandendo l'immagine a $384\text{ px}$, la fusoliera dell'aereo militare nella **Riga $4$** diventa più grande del campo recettivo dei neuroni centrali, che non vedono più i contorni esterni dell'aereo e generano **buchi interni** nella maschera; analogamente nella **Riga $3$** compaiono falsi positivi viola sul piatto. Di conseguenza `mIoU Obj` cala al **`36.01%`**.
   * **Esplosione dei tempi ($4.30\times$):** L'aumento quadratico dei pixel ($1.5^2 = 2.25\times$) porta il tempo di esecuzione da $106.1\text{ s}$ a **$455.8\text{ s}$**, rendendo insostenibile un confronto multi-loss su migliaia di immagini.

---

## 5. Verdetto Ingegneristico: Ponte verso lo Step 4 (*Ceteris Paribus*)

La mini-ablation dello Step 3-BIS consegna una giustificazione sperimentale solida per impostare lo **Step 4**:

1. **Confermiamo `CROP_SIZE = 256` e `BATCH_SIZE = 16`:** Garantisce il miglior compromesso tra campo recettivo, stabilità delle statistiche `BatchNorm2d` ($8$ immagini/GPU) e sostenibilità computazionale.
2. **Come recuperiamo gli oggetti piccoli e rari senza pagare il costo $4.30\times$ di $384\text{ px}$?**  
   * **Durante il training (Step 4):** Agendo sulla **funzione di perdita** a costo zero di VRAM aggiuntiva — confrontando *ceteris paribus* (partendo da `shared_init_fcn_resnet50.pt`) **`Cross-Entropy`**, **`Focal Loss`**, **`Dice Loss`**, **`Jaccard Loss`** e **`0.5 * CE + 0.5 * Dice`**, che normalizzano il gradiente rispetto alla dimensione della classe anziché pesare ogni pixel allo stesso modo.
   * **Durante la valutazione finale (Step 9):** Sfruttando la proprietà *resolution-agnostic* della FCN per valutare anche sull'immagine intera (`batch_size=1`) senza crop.

---

## 6. Riferimenti Documentali e Bibliografici Validati per il Colloquio

| Fonte / Riferimento | Dettaglio Specifico | Argomento a Supporto |
| :--- | :--- | :--- |
| **1. Ioffe & Szegedy (ICML 2015)** | *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* | Spiega la differenza tra normalizzazione su statistiche del mini-batch (`model.train()`) e uso di `running_mean`/`running_var` (`model.eval()`) sotto `DataParallel`. |
| **2. Loshchilov & Hutter (ICLR 2017)** | *SGDR: Stochastic Gradient Descent with Warm Restarts* (Cosine Annealing) | Formalizza il decadimento continuo del *learning rate* con legge del coseno per stabilizzare la convergenza finale senza gradini arbitrari. |
| **3. Luo et al. (NeurIPS 2016)** | *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks* | Spiega perché il campo recettivo effettivo è limitato in pixel e perché ingrandire l'input a $384\text{ px}$ (`T3`) frammenta gli oggetti estesi. |
| **4. Specifiche di Progetto (`H3.pdf`) & Scheletro (`Sez. 12`)** | *"Use different loss functions and their combination to train it"* + confronto *ceteris paribus* da pesi iniziali identici | Motiva il passaggio dallo Step 3-BIS allo Step 4 mantenendo fissi `256px`, `BATCH_SIZE = 16` e `shared_init_fcn_resnet50.pt`. |

---

## 7. Sintesi Pronta per la Discussione Orale

* **Risposta sintetica al professore:**  
  *"Nella mini-ablation dello Step 3-BIS abbiamo confrontato a parità di pesi iniziali Baseline (`T1`, 256px, 80 ep), Scheduler `CosineAnnealingLR` (`T2`, 256px, 150 ep) e Risoluzione 384px (`T3`, 150 ep), valutandoli in `model.eval()` con accumulatore globale sulle 29 classi oggetto presenti. Il divario numerico rispetto alle metriche batch-wise dello Step 3 (`79.13%` vs `43.98%`) deriva sia dal peso uniforme assegnato a tutte e 30 le classi nell'accumulatore globale, sia dal comportamento di `BatchNorm2d` su micro-batch da 4 immagini/GPU sotto DataParallel. Confrontando i tre tentativi, `T3` (384px) recupera effettivamente più mucche distanti nella Figura 06 grazie alla griglia latente $48 \times 48$, ma quadruplica i tempi ($455.8\text{ s}$) e introduce buchi interni sugli oggetti grandi come l'aereo perché il campo recettivo della ResNet-50 è fisso in pixel, portando la mIoU Obj al $36.01\%$. Confermiamo quindi `CROP_SIZE = 256` e `BATCH_SIZE = 16` per lo Step 4, demandando alle funzioni di perdita regionali e focali (Dice, Jaccard, Focal, CE+Dice) il recupero delle classi piccole e rare a costo zero di VRAM."*