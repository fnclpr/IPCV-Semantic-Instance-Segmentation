# \[03\_QA\] Step 3 — Architettura `FCN-ResNet50` *from scratch*, *Output Stride*, Ramo Ausiliario e Calibrazione Multi-GPU

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Riferimento Codice:** `Notebook1_SetupDatasetCOCO2017_Babysitting_(Step1_Step3-BIS)` — Celle 1, 4 e 5 (`Sezioni 9–10`)  
**Collegamento Progettuale:** Step 2 (`COCOSemanticDataset`, $256 \times 256$) $\rightarrow$ Step 3 (`FCN-ResNet50` *from scratch*) $\rightarrow$ Step 4 (Confronto Loss Functions) $\rightarrow$ Step 6 (*Transfer Learning* su Cityscapes)  
**Tipo di Documento:** Scheda di Approfondimento Teorico-Metodologico (Q&A)

---

## 1. Quesiti Metodologici Formalizzati

Questa scheda approfondisce i meccanismi interni dell'architettura `torchvision.models.segmentation.fcn_resnet50` e la sua integrazione nel nostro progetto, rispondendo a cinque domande tipiche del colloquio orale:

1. **Trasformazione *Fully Convolutional*:** *In che modo `FCN-ResNet50` trasforma un classificatore d'immagine (ResNet-50) in un modello di segmentazione densa pixel-per-pixel, e cosa comporta l'istanziazione completamente from scratch (`weights=None`, `weights_backbone=None`)?*
2. **Convoluzioni Dilatate (*Atrous*) e *Output Stride* ($s = 8$):** *Perché la griglia latente interna della nostra FCN misura $32 \times 32$ anziché $8 \times 8$ su ingressi $256 \times 256$, e quale legame matematico sussiste con il $15.21\%$ di istanze COCO aventi lato equivalente $< 8\text{ px}$?*
3. **Confronto Architetturale con U-Net (Scheletro del Corso):** *Qual è la differenza strutturale tra la `FCN-ResNet50` richiesta dalle specifiche `H3.pdf` e la `U-Net` dello scheletro di lezione, e perché i contorni predetti nella Figura `05_step3_babysitting_overfit_predictions.png` appaiono leggermente smussati?*
4. **Ruolo del Ramo Ausiliario (`aux_loss=True`, peso $\lambda = 0.5$):** *A cosa serve `out["aux"]` estratto da `layer3` durante l'addestramento from scratch, e perché scompare quando si imposta `model.eval()`?*
5. **Calibrazione Multi-GPU e Persistenza Pesi Iniziali:** *Come abbiamo calibrato `BATCH_SIZE = 16` sulle $2$ GPU NVIDIA Tesla T4 (`nn.DataParallel`) e perché salviamo su disco `shared_init_fcn_resnet50.pt`?*

---

## 2. Come Funziona Internamente `FCN-ResNet50` (*from scratch*)

### 2.1 Dalla Classificazione Globale alla Predizione Densa

Una rete **ResNet-50** standard progettata per la classificazione su ImageNet termina con un livello di *Global Average Pooling* e un livello *Fully Connected* (`nn.Linear(2048, 1000)`), che collassa l'intera dimensione spaziale $(H, W)$ in un singolo vettore $1\text{D}$, perdendo l'informazione sulla posizione degli oggetti.

Come introdotto da *Long, Shelhamer & Darrell (CVPR 2015)* e implementato in `torchvision.models.segmentation.fcn_resnet50` (`Cella 4`, `build_fcn_from_scratch`):

1. **Rimozione dei layer densi:** I livelli di *pooling* globale e *fully connected* vengono rimossi.
2. **Testa convoluzionale (`FCNHead`):** L'uscita profonda del backbone viene elaborata da una convoluzione $3 \times 3$ (`Conv2d(2048, 512, kernel_size=3, padding=1, bias=False)` + `BatchNorm2d` + `ReLU` + `Dropout(0.1)`) seguita da una **convoluzione $1 \times 1$** (`Conv2d(512, 81, kernel_size=1)`), che agisce come un classificatore lineare indipendente applicato a ciascuna cella spaziale.
3. **Upsampling Bilineare:** La mappa dei *logit* a bassa risoluzione viene riportata alle dimensioni esatte dell'immagine di ingresso $(B, 81, 256, 256)$ tramite interpolazione bilineare (`F.interpolate(..., size=(256, 256), mode='bilinear', align_corners=False)`).

### 2.2 Vincolo *From Scratch* (`weights=None`, `weights_backbone=None`)

In conformità alle specifiche di progetto (`H3.pdf`), istanziamo la rete imponendo entrambi i flag a `None` (`35,368,418` parametri totali, tutti addestrabili):

* **Significato tecnico:** Né i filtri convoluzionali di basso livello del backbone (`conv1`, `layer1..4`) né le teste di classificazione (`classifier`, `aux_classifier`) ricevono pesi pre-addestrati su ImageNet o COCO. Tutti i pesi vengono inizializzati casualmente (inizializzazione di Kaiming/He per le convoluzioni e $\gamma = 1, \beta = 0$ per i layer `BatchNorm2d`).
* **Implicazione pratica:** A differenza del *fine-tuning* (dove i filtri di Gabor per bordi e texture sono già formati), nei primi step di training la rete deve apprendere simultaneamente sia l'estrazione delle feature visive sia la loro aggregazione semantica sulle $81$ classi di COCO.

---

## 3. *Output Stride* ($s = 8$), Convoluzioni Dilatate (*Atrous*) e Limite sugli Oggetti Piccoli

### 3.1 Perché la Griglia Latente è $32 \times 32$ e non $8 \times 8$

Nella ResNet-50 originale per classificazione, ogni stadio (`stem` $/4$, `layer2` $/8$, `layer3` $/16$, `layer4` $/32$) dimezza la risoluzione spaziale, producendo un **Output Stride** pari a $s = 32$. Applicato a un ingresso $256 \times 256$, ciò genererebbe una mappa latente di appena $8 \times 8 = 64$ celle, troppo grossolana per segmentare oggetti complessi.

Nell'implementazione `fcn_resnet50` di `torchvision`, il backbone modifica gli ultimi due stadi (`replace_stride_with_dilation=[False, True, True]`):

* **`layer3` (stride $1$, *dilation rate* $d = 2$):** Mantiene la risoluzione spaziale bloccata a $H/8 \times W/8 = 32 \times 32$ ($1\,024$ celle).
* **`layer4` (stride $1$, *dilation rate* $d = 4$):** Mantiene anch'esso la risoluzione spaziale a $32 \times 32$ ($2\,048$ canali).

Le **convoluzioni dilatate (*atrous*, *Chen et al., ICLR 2015*)** inseriscono $d - 1$ zeri tra i pesi del kernel $3 \times 3$. In questo modo il filtro espande il proprio campo recettivo effettivo senza ridurre la risoluzione della griglia spaziale e senza aggiungere parametri addestrabili.

### 3.2 Connessione Diretta con i Dati di COCO 2017 (`Cella 3` $\rightarrow$ `Cella 5`)

Con un **Output Stride $s = 8$**, ogni singola cella della mappa latente $32 \times 32$ corrisponde a una regione di **$8 \times 8 = 64\text{ pixel}$** nell'immagine di ingresso $256 \times 256$:

* Nell'analisi statistica integrale della **Cella 3** (`analyze_coco_split`), abbiamo misurato che in `train2017` ben **$130\,797 / 860\,001$ istanze ($15.21\%$)** e in `val2017` **$5\,526 / 36\,781$ istanze ($15.02\%$)** possiedono un lato equivalente $\sqrt{A_{\text{post}}} < 8\text{ px}$ dopo il ridimensionamento a $256\text{ px}$.
* Tali istanze occupano **meno di una singola cella** della griglia latente prima dell'upsampling bilineare, risultando fisicamente difficili da separare dallo sfondo circostante.
* **Conferma visiva nel notebook (`05_step3_babysitting_overfit_predictions.png`):** Nella **Riga 1**, la FCN segmenta nitidamente le due automobili estese (`car` in azzurro), mentre cattura solo una minima frazione della mandria di bovini distanti (`cow` in magenta), dimostrando sul campo il limite fisico imposto dall'*output stride* $s = 8$.

---

## 4. Confronto Architetturale: `FCN-ResNet50` vs `U-Net` (Scheletro del Corso)

Poiché il Kaggle Notebook di lezione utilizza una `U-Net` mentre la traccia `H3.pdf` richiede una `FCN`, è importante saperne confrontare l'architettura durante l'orale:

| Caratteristica Architetturale | **`FCN-ResNet50` (`torchvision`)** — *Nostro Progetto* | **`U-Net` (`UNet` nello Scheletro)** — *Lezione* |
| :--- | :--- | :--- |
| **Encoder (Backbone)** | **ResNet-50** ($50$ strati con blocchi residui *Bottleneck* e convoluzioni *atrous* $d=2, 4$). | Encoder simmetrico a $4$ livelli di `MaxPool2d(2)` ($8$ convoluzioni $3 \times 3$). |
| **Risoluzione minima latente** | $32 \times 32$ (**Output Stride $s = 8$**). | $16 \times 16$ (*bottleneck* $/16$ su ingressi $256 \times 256$). |
| **Decoder / Upsampling** | **Singolo upsampling bilineare $8\times$** direttamente dai *logit* $32 \times 32$ a $256 \times 256$. | Decoder simmetrico progressivo ($/8 \rightarrow /4 \rightarrow /2 \rightarrow /1$) con **`ConvTranspose2d`**. |
| **Skip Connections ad alta risoluzione** | **Assenti** tra encoder (`conv1`, `layer1`) e decoder. | **Presenti** a ogni livello (`torch.cat([x_up, x_enc], dim=1)`). |
| **Supervisione intermedia** | **Presente** (`aux_classifier` su `layer3` con peso $0.5$). | **Assente** (singola uscita finale `conv_f`). |
| **Implicazione sui contorni** | Contorni leggermente arrotondati/smussati (visibile nella Figura `05` attorno alle zampe dei cani), ma forte capacità semantica sulle $81$ classi grazie ai $35.37\text{M}$ parametri residui. | Contorni molto taglienti grazie alle *skip connections*, ma priva della profondità residua necessaria a discriminare $81$ categorie complesse *from scratch*. |

---

## 5. Il Ramo Ausiliario (`aux_loss=True`, Peso $\lambda = 0.5$)

### 5.1 Perché è Fondamentale nell'Addestramento *From Scratch*

Quando `aux_loss=True` (`Cella 4`), la `FCN-ResNet50` istanzia una seconda testa convoluzionale (`aux_classifier`: `Conv2d(1024, 256, 3x3)` + `BatchNorm2d` + `ReLU` + `Dropout(0.1)` + `Conv2d(256, 81, 1x1)`) collegata direttamente all'uscita di **`backbone.layer3`**. Durante il forward pass in training (`model.train()`), la funzione obiettivo complessiva diventa:

$$
\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{main}}\big(\text{out}[\text{'out'}],\, y\big) + 0.5 \cdot \mathcal{L}_{\text{aux}}\big(\text{out}[\text{'aux'}],\, y\big)
$$

Questo meccanismo (*Deep Supervision*, analogo a GoogLeNet / PSPNet) apporta tre vantaggi decisivi:

1. **Iniezione diretta del gradiente agli strati intermedi:** Nelle prime iterazioni *from scratch*, il gradiente che parte da `classifier` deve attraversare tutti i blocchi residui di `layer4` prima di raggiungere `layer3`. Il ramo ausiliario fornisce una **scorciatoia di gradiente diretta** a `layer3`, accelerando e stabilizzando la formazione dei filtri intermedi.
2. **Campo recettivo più locale:** Poiché `layer3` precede le convoluzioni dilatate $d = 4$ di `layer4`, il suo campo recettivo effettivo è più compatto e localizzato, agendo da supervisore complementare per gli oggetti di dimensioni medio-piccole.
3. **Zero costi in inferenza:** In modalità valutazione (`model.eval()`), `torchvision` disattiva automaticamente `aux_classifier` e restituisce un dizionario contenente **solo** la chiave `'out'` (trappola implementativa gestita esplicitamente nella funzione `_evaluate_ablation_subset` della Cella 5).

---

## 6. Calibrazione Multi-GPU (`BATCH_SIZE = 16`) e Persistenza `shared_init_fcn_resnet50.pt`

### 6.1 Scelta del `BATCH_SIZE = 16` su $2 \times \text{NVIDIA Tesla T4}$ (`Cella 4`)

La funzione `calibrate_batch_size_and_device` ha testato empiricamente i candidati $\{8, 16, 32\}$ con un passo reale *forward + backward* su $2 \times \text{Tesla T4}$ ($14.56\text{ GB}$ VRAM ciascuna, `DataParallel=True`):

| `BATCH_SIZE` | VRAM Picco | Tempo per Step | Immagini per GPU | Valutazione Progettuale |
| :---: | :---: | :---: | :---: | :--- |
| **$8$** | $3.25\text{ GB}$ | $682.0\text{ ms}$ | $4$ | Sottoutilizza le $2$ GPU e rende rumorose le statistiche locali di `BatchNorm2d`. |
| **$16$** | $5.17\text{ GB}$ | $1152.3\text{ ms}$ | $8$ | **Ottimo operativo:** $< 36\%$ VRAM occupata, $8$ immagini/GPU e doppia frequenza di aggiornamento rispetto a $32$. |
| **$32$** | $9.00\text{ GB}$ | $2226.6\text{ ms}$ | $16$ | Dimezza gli aggiornamenti per epoca e riduce il margine VRAM per i tensori *one-hot* `(B, 81, 256, 256)` dello Step 4. |

### 6.2 Perché Persistiamo `shared_init_fcn_resnet50.pt` su Disco

In conformità alla **Sezione 12 dello scheletro** (*"every experiment starts with identical initial weights"*), dopo aver istanziato `fcn_base` con `SEED = 42` salviamo il suo `state_dict()` su disco (`/kaggle/working/checkpoints/shared_init_fcn_resnet50.pt`). In questo modo sia la mini-ablation dello Step 3-BIS sia tutti gli esperimenti comparativi sulle *loss* nello Step 4 partono **esattamente dallo stesso punto nello spazio dei parametri**, garantendo un confronto scientifico *ceteris paribus* anche in caso di riavvio della sessione Kaggle.

---

## 7. Riferimenti Documentali e Bibliografici Validati per il Colloquio

| Fonte / Riferimento | Dettaglio Specifico | Argomento a Supporto |
| :--- | :--- | :--- |
| **1. Long, Shelhamer & Darrell (CVPR 2015)** | *Fully Convolutional Networks for Semantic Segmentation* | Sostituzione dei layer *Fully Connected* con convoluzioni $1 \times 1$, predizione densa pixel-per-pixel e upsampling bilineare. |
| **2. He et al. (CVPR 2016)** | *Deep Residual Learning for Image Recognition* (ResNet-50) | Struttura dei blocchi residui *Bottleneck* che permette di addestrare reti profonde $50$ strati completamente *from scratch*. |
| **3. Chen et al. (ICLR 2015 / TPAMI 2018)** | *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets and Atrous Convolution* | Uso delle convoluzioni dilatate (*atrous*) negli stadi finali per mantenere l'*Output Stride* $s = 8$ senza perdita di risoluzione. |
| **4. Zhao et al. (CVPR 2017) & Documentazione PyTorch** | *Pyramid Scene Parsing Network* (Auxiliary Loss) e `torchvision.models.segmentation.fcn_resnet50` | Supervisione ausiliaria su `layer3` pesata a $\lambda = 0.5$ durante il training e rimozione automatica di `'aux'` in `model.eval()`. |
| **5. Specifiche di Progetto (`H3.pdf`) & Scheletro (`Sez. 10, 12`)** | Addestramento *from scratch* di `FCN`, supporto `nn.DataParallel` e identici pesi iniziali tra esperimenti | Giustifica `weights=None`, `weights_backbone=None`, calibrazione `BATCH_SIZE = 16` e persistenza di `shared_init_fcn_resnet50.pt`. |

---

## 8. Sintesi Pronta per la Discussione Orale

* **Risposta sintetica al professore:**  
  *"In conformità con le specifiche H3, abbiamo istanziato `FCN-ResNet50` completamente from scratch (`weights=None`, `weights_backbone=None`, $35.37\text{M}$ parametri addestrabili). La rete sostituisce i layer densi della ResNet-50 con convoluzioni $1 \times 1$ e impiega convoluzioni dilatate (atrous) in `layer3` e `layer4` per mantenere un Output Stride pari a $s = 8$, comprimendo l'ingresso $256 \times 256$ in una griglia latente $32 \times 32$ prima dell'upsampling bilineare. Questo spiega sia il perché i contorni siano leggermente più smussati rispetto a una U-Net (priva di skip connections ad alta risoluzione), sia il perché il $15.21\%$ di istanze COCO con lato equivalente $< 8\text{ px}$ rappresenti un limite fisico strutturale. Durante il training sfruttiamo inoltre il ramo ausiliario su `layer3` (`AUX_LOSS_WEIGHT = 0.5`) per iniettare gradiente diretto negli strati intermedi, e abbiamo calibrato `BATCH_SIZE = 16` ($5.17\text{ GB}$ VRAM) sulle $2$ Tesla T4 salvando i pesi iniziali in `shared_init_fcn_resnet50.pt` per garantire confronti ceteris paribus nello Step 4."*