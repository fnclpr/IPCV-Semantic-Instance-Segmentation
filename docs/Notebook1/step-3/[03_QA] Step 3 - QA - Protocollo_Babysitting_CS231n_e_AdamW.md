# \[03\_QA\] Step 3 — Protocollo di *Babysitting* (*CS231n*), Verifiche Analitiche e Ottimizzatore `AdamW`

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Riferimento Codice:** `Notebook1_SetupDatasetCOCO2017_Babysitting_(Step1_Step3-BIS)` — Celle 4 e 5 (`Sezioni 10–11`)  
**Collegamento Progettuale:** Step 2 (`COCOSemanticDataset`) $\rightarrow$ Step 3 (`run_step3_babysitting`) $\rightarrow$ Step 4 (Generalizzazione su $12\,000$ immagini)  
**Tipo di Documento:** Scheda di Approfondimento Teorico-Metodologico (Q&A)

---

## 1. Quesiti Metodologici Formalizzati

Questo documento illustra i fondamenti teorici ed empirici del protocollo di ***babysitting* del processo di apprendimento** implementato nella **Cella 5** (`run_step3_babysitting`), rispondendo a cinque domande critiche in vista dell'esame orale:

1. **Razionale Metodologico del *Babysitting*:** *Perché prima di addestrare su $12\,000$ immagini eseguiamo un test di overfitting su un mini-subset fisso di $16$ immagini per $80$ epoche invece di partire subito con poche epoche su migliaia di immagini?*
2. **Derivazione Matematica del *Sanity Check 1* (Step $0$):** *Come si ricava analiticamente la loss iniziale attesa a pesi casuali ($-\ln(1/81) = 4.3944$, totale con aux $6.5917$) e perché i valori misurati nel notebook ($4.4173$ e $6.5704$) confermano l'assenza di bug?*
3. **Diagnostica dei Gradienti (*Sanity Check 2* — Rapporto *Update-to-Weight*):** *Cosa misura il rapporto $\lVert \Delta W \rVert_2 / \lVert W \rVert_2 \sim 10^{-3}$ (*CS231n*), perché al Passo $1$ `backbone.layer4` segna $4.87 \times 10^{-6}$ mentre `backbone.conv1` segna $1.34 \times 10^{-2}$, e perché `AdamW` compensa automaticamente questa disparità?*
4. **Regime Diagnostico vs Regime di Generalizzazione:** *Perché nel test di overfitting disabilitiamo la regolarizzazione (`weight_decay=0.0`) e l'augmentation stocastica (`is_train=False`, `CenterCrop`), mentre nello Step 4 li riattiviamo?*
5. **Lettura Critica delle Curve e delle Predizioni (`Figure 04 e 05`):** *Perché nella Figura `04` la `Pixel Accuracy` raggiunge il $98.62\%$ mentre la `mIoU` si attesta al $79.13\%$, e che cosa dimostrano le quattro scene della Figura `05`?*

---

## 2. Perché il *Babysitting* su $16$ Immagini $\times$ $80$ Epoche (e Non Migliaia di Immagini $\times$ Poche Epoche)

Come prescritto dalle specifiche di progetto (`H3.pdf`: *"train from scratch (with babysitting)"*) e dai moduli teorici del corso (*CS231n — Babysitting the Learning Process*), il *babysitting* è un **controllo binario di integrità ingegneristica**, distinto dalla fase di generalizzazione:

1. **Isolamento dei bug di codice dal problema del *underfitting*:**  
   Se avviassimo subito un training su $12\,000$ o $118\,287$ immagini per $5\text{--}10$ epoche e la *loss* restasse alta ($\approx 3.5$) o la `mIoU` bassa ($\approx 5\%$), sarebbe impossibile distinguere tra un **bug strutturale** (es. maschere disallineate, indici di classe errati, gradienti interrotti) e un semplice **sotto-addestramento** dei $35.37\text{M}$ parametri inizializzati *from scratch*. Riducendo i dati a **$16$ immagini fisse**, trasformiamo il problema in un compito puramente meccanico di **memorizzazione**: poiché una rete da $35.37\text{M}$ parametri ha capacità ampiamente superiore a $16 \times 256 \times 256 \approx 1.05 \times 10^6$ pixel, l'incapacità di memorizzarle rivelerebbe con certezza matematica un errore implementativo.
2. **Numero di aggiornamenti per singolo campione (Matematica del Gradiente):**  
   * Nel *babysitting* ($16$ immagini, `batch_size=8`, $80$ epoche), la rete osserva **ciascuna immagine $80$ volte** ($160$ passi di ottimizzazione complessivi, completati in soli **$109.3\text{ s}$**).
   * Su $12\,000$ immagini per $5$ epoche, la rete osserverebbe ogni immagine solo $5$ volte su scene sempre diverse, senza il tempo necessario a formare i filtri convoluzionali di basso livello partendo da pesi casuali.
3. **Selezione intelligente del mini-subset (`Cella 5`):**  
   Anziché prendere le prime $16$ immagini arbitrarie di `train2017` (che conterrebbero quasi solo `person`), utilizziamo `select_class_balanced_subset(coco_train, max_samples=100, seed=42)[:16]`, ottenendo un pool compatto di $16$ scene che espone **$29$ categorie *Thing* distinte** oltre al `background`.

---

## 3. Derivazione Matematica dei *Sanity Check 1 e 2* (`Cella 5`)

### 3.1 Sanity Check 1 — Loss Iniziale a Pesi Casuali (Step $0$)

Prima di eseguire qualsiasi passo di ottimizzazione (`Step 0`), i pesi del classificatore finale (`classifier.4` e `aux_classifier.4`) sono inizializzati con valori casuali piccoli e simmetrici attorno allo zero. Di conseguenza, per ogni pixel valido $(y, x)$, i *logit* $z_c$ sono prossimi a zero ($z_c \approx 0 \;\forall c \in \{0, \dots, 80\}$) e la funzione *Softmax* assegna una distribuzione quasi uniforme:

$$
p_c = \frac{\exp(z_c)}{\sum_{j=0}^{80} \exp(z_j)} \approx \frac{1}{81} \qquad \forall c \in \{0, \dots, 80\}
$$

La *Cross-Entropy Loss* teorica attesa per singolo pixel sul ramo principale è quindi:

$$
\mathcal{L}_{\text{main}}^{(0)} = -\ln(p_{y^*}) \approx -\ln\left(\frac{1}{81}\right) = \ln(81) \approx \mathbf{4.3944}
$$

Includendo il ramo ausiliario pesato con `AUX_LOSS_WEIGHT = 0.5`, il valore teorico composito atteso è:

$$
\mathcal{L}_{\text{tot}}^{(0)} = \mathcal{L}_{\text{main}}^{(0)} + 0.5\,\mathcal{L}_{\text{aux}}^{(0)} \approx 1.5 \times 4.3944 = \mathbf{6.5917}
$$

**Confronto con i valori empirici misurati nella Cella 5:**

| Componente della Loss al Passo $0$ | Valore Teorico Atteso | Valore Empirico Misurato | Scostamento | Diagnostica |
| :--- | :---: | :---: | :---: | :--- |
| **Ramo Principale (`out`)** | $4.3944$ | **$4.4173$** | $+0.52\%$ | Coerente: per la disuguaglianza di Jensen, *logit* casuali a varianza finita aumentano lievemente la CE media rispetto al caso $z_c \equiv 0$. |
| **Totale (`main + 0.5 * aux`)** | $6.5917$ | **$6.5704$** | $-0.32\%$ | Certifica che dimensionalità `(B, 81, 256, 256)`, mascheramento `IGNORE_INDEX = 255` e media sui pixel validi sono esatti. |

### 3.2 Sanity Check 2 — Rapporto *Update-to-Weight* ($\sim 10^{-3}$) e Ruolo di `AdamW`

Come indicato nelle dispense *CS231n*, per evitare sia la divergenza (passo troppo grande) sia lo stallo dell'apprendimento (passo troppo piccolo), il rapporto tra la norma $L_2$ dell'aggiornamento dei parametri e la norma $L_2$ dei pesi stessi al primo step deve collocarsi intorno all'ordine di grandezza di **$10^{-3}$**:

$$
r_l = \frac{\lVert \Delta W_l \rVert_2}{\lVert W_l \rVert_2} = \frac{\lVert \eta \cdot \nabla_{W_l} \mathcal{L} \rVert_2}{\lVert W_l \rVert_2} \sim 10^{-3}
$$

Nella **Cella 5** (`compute_update_to_weight_ratios` con $\eta = 10^{-3}$), abbiamo misurato questo indicatore su tre livelli di profondità della rete:

| Layer Monitorato | Posizione Architetturale | Rapporto Misurato al Passo $1$ | Spiegazione Fisica e Ruolo di `AdamW` |
| :--- | :--- | :---: | :--- |
| `backbone.conv1` | Primo strato convoluzionale ($7 \times 7$, ingresso) | **$1.34 \times 10^{-2}$** | Riceve l'intero flusso di gradiente sommato su tutti i pixel; un ordini di grandezza superiore a $10^{-3}$ accelera la formazione dei primi filtri di bordo. |
| `backbone.layer4` | Ultimo blocco residuo profondo ($3 \times 3$, $512$ ch) | **$4.87 \times 10^{-6}$** | Fisiologico al Passo $1$ *from scratch*: prima che le statistiche dei $50$ strati `BatchNorm2d` si stabilizzino, il gradiente attraverso i blocchi residui non pre-addestrati è attenuato e la norma $\lVert W \rVert_2$ dei tensori a $512$ canali è elevata. |
| `classifier.4` | Testa di segmentazione finale ($1 \times 1$, $81$ ch) | **$1.16 \times 10^{-3}$** | **Centrato esattamente sul target teorico $\sim 10^{-3}$**, confermando il corretto condizionamento del classificatore finale. |

**Perché `AdamW` è decisivo in questo scenario *from scratch*:**  
Se usassimo `SGD` semplice con un unico *learning rate* scalare $\eta$, `layer4` avanzerebbe $2\,700\times$ più lentamente di `conv1`. L'ottimizzatore **`AdamW`** (*Loshchilov & Hutter, ICLR 2019*) normalizza invece l'aggiornamento di ciascun parametro $\theta_i$ tramite la radice della media mobile esponenziale dei gradienti al quadrato $\sqrt{\hat{v}_{t,i}} + \epsilon$, riequilibrando istantaneamente le diverse scale di gradiente tra `conv1`, `layer4` e `classifier` fin dalle prime iterazioni.

---

## 4. Regime Diagnostico (Step 3) vs Regime di Generalizzazione (Step 4)

Un punto fondamentale da chiarire al colloquio è perché le impostazioni del `babysit_loader` e dell'ottimizzatore nello Step 3 sono intenzionalmente diverse da quelle dello Step 4:

| Parametro / Scelta | **Step 3 (*Babysitting* Diagnostico)** | **Step 4 (Training Comparativo Generalizzato)** | Ragione Metodologica |
| :--- | :--- | :--- | :--- |
| **Dimensione Dataset** | $16$ immagini (`cache_in_ram=True`) | $12\,000$ immagini (`select_class_balanced_subset`) | Lo Step 3 verifica la memorizzazione meccanica; lo Step 4 apprende la generalizzazione su $80$ classi. |
| **Trasformazione Spaziale** | `is_train=False` (`CenterCrop` fisso, no flip) | `is_train=True` (`RandomCrop` + `HorizontalFlip`) | Nel *babysitting* il target deve restare **stazionario** affinché la *loss* possa tendere a zero senza rumore. |
| **Regolarizzazione (`weight_decay`)** | `weight_decay = 0.0` | `weight_decay = 1e-4` | Nel *babysitting* non vogliamo frenare la capacità di memorizzazione; nello Step 4 preveniamo l'*overfitting*. |
| **Batch Size / Epoche** | `batch_size = 8` ($2$ batch/epoca), $80$ epoche | `BATCH_SIZE = 16` ($750$ batch/epoca), epoche fisse | $2$ batch/epoca su $16$ immagini raddoppiano gli aggiornamenti per epoca ($160$ step in $109.3\text{ s}$). |
| **Metrica di Monitoraggio** | *Batch-wise* in `model.train()` (`0.53%` $\rightarrow$ `79.13%`) | Accumulatore globale $81 \times 81$ in `model.eval()` | Su $16$ immagini in `DataParallel` ($4$ img/GPU), il calcolo *batch-wise* evita il ritardo delle *running stats* di `BatchNorm2d`. |

---

## 5. Interpretazione delle Curve e delle Maschere (`Figure 04 e 05`)

1. **Curva di Loss e Metriche (`04_step3_babysitting_loss_metrics_curves.png`):**
   * **Loss (pannello sinistro):** Scende monotonamente da `6.6100` (Epoca 1) a `1.5603` (Epoca 10), `0.2487` (Epoca 30) fino a **`0.0680`** all'Epoca 80 ($-99.0\%$), dimostrando assenza totale di instabilità numeriche.
   * **Divario `Pixel Accuracy` (`98.62%`) vs `mIoU` (`79.13%`, pannello destro):** La `Pixel Accuracy` supera il $95\%$ già all'Epoca 30 perché la classe `0 = background` occupa l'$80\text{--}90\%$ dei pixel. La `mIoU` pesa invece equamente ogni classe presente indipendentemente dall'area, crescendo gradualmente fino al $79.13\%$ e fermandosi sotto il $100\%$ a causa della risoluzione discreta della griglia latente $32 \times 32$ ($s = 8$).
2. **Griglia Qualitativa (`05_step3_babysitting_overfit_predictions.png`):**
   * Conferma che la rete memorizza fedelmente le macro-strutture (`car`, `dog`, `surfboard`, `bowl`, `broccoli`, `airplane`), mentre perde le istanze minute della mandria distante nella Riga 1 (`cow`), fornendo la diagnosi visiva che motiva lo Step 3-BIS e lo Step 4.

---

## 6. Riferimenti Documentali e Bibliografici Validati per il Colloquio

| Fonte / Riferimento | Dettaglio Specifico | Argomento a Supporto |
| :--- | :--- | :--- |
| **1. Dispense Teoriche del Corso (*CS231n*)** | *Training Neural Networks I & II — Babysitting the Learning Process* | Prescrive il controllo della loss iniziale $-\ln(1/C)$, il test di overfitting su un piccolo batch senza regolarizzazione e il monitoraggio del rapporto $\lVert \Delta W \rVert / \lVert W \rVert \approx 10^{-3}$. |
| **2. Specifiche di Progetto (`H3.pdf`)** | *"Implement and train from scratch (with babysitting) FCN network on COCO dataset"* | Rende obbligatoria l'esecuzione documentata della fase di *babysitting* prima del training comparativo delle *loss*. |
| **3. Loshchilov & Hutter (ICLR 2019)** | *Decoupled Weight Decay Regularization* (`AdamW`) | Giustifica l'uso di `AdamW` per normalizzare adattivamente i gradienti sbilanciati tra `conv1` e `layer4` nei modelli *from scratch* e disaccoppiare il *weight decay*. |
| **4. He et al. (ICCV 2015)** | *Delving Deep into Rectifiers* (Kaiming Initialization) | Spiega l'inizializzazione casuale dei pesi convoluzionali da cui deriva la distribuzione quasi uniforme dei *logit* al Passo $0$. |

---

## 7. Sintesi Pronta per la Discussione Orale

* **Risposta sintetica al professore:**  
  *"Prima di avviare addestramenti onerosi su migliaia di immagini, abbiamo applicato il protocollo di babysitting CS231n su un mini-subset deterministico di $16$ immagini ($29$ classi oggetto distinte) per $80$ epoche senza regolarizzazione (`weight_decay=0.0`, `CenterCrop` fisso). Questo ci ha permesso di verificare tre proprietà fondamentali: (1) al Passo 0 la loss iniziale empirica ($4.4173$ main, $6.5704$ totale) coincide con il valore analitico $-\ln(1/81) = 4.3944$ ($6.5917$ con aux); (2) al Passo 1 il rapporto Update-to-Weight sulla testa (`classifier.4`) è $1.16 \times 10^{-3}$, centrato sul target teorico $\sim 10^{-3}$, mentre l'ottimizzatore AdamW compensa automaticamente l'attenuazione iniziale dei gradienti tra `conv1` ($1.34 \times 10^{-2}$) e `layer4` ($4.87 \times 10^{-6}$); (3) in $80$ epoche ($109.3\text{ s}$) la loss scende monotonamente a $0.0680$ con Pixel Accuracy del $98.62\%$ e mIoU del $79.13\%$, dimostrando che l'architettura e il flusso dei gradienti sono completamente sani."*