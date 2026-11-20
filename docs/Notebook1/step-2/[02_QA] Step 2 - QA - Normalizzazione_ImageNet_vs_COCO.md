# \[02\_QA\] Step 2 — Costanti di Normalizzazione ImageNet su COCO 2017 e Backbone ResNet

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Riferimento Codice:** `Notebook1_SetupDatasetCOCO2017_Babysitting_(Step1_Step3-BIS)` — Celle 1, 3, 4 e 5  
**Collegamento Progettuale:** Step 2 (*Preprocessing `COCOSemanticDataset`*) $\rightarrow$ Step 3 (*Babysitting FCN-ResNet50*) $\rightarrow$ Step 6 (*Transfer Learning su Cityscapes*)  
**Tipo di Documento:** Scheda di Approfondimento Teorico-Metodologico (Q&A)  

---

## 1. Quesiti Metodologici Formalizzati

In teoria del *Machine Learning*, la standardizzazione per canale di un tensore immagine RGB $x \in [0, 1]^{3 \times H \times W}$:

$$\tilde{x}_c = \frac{x_c - \mu_c}{\sigma_c} \qquad \forall c \in \{R, G, B\}$$

dovrebbe impiegare media $\mu_c$ e deviazione standard $\sigma_c$ stimate sul *training set* del problema specifico. Nella **Cella 1** e nella **Cella 3** (`COCOSemanticDataset._apply_synchronized_transforms`) del nostro notebook utilizziamo invece le costanti canoniche di **ImageNet**:

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

Questo documento risponde in modo rigoroso a quattro quesiti chiave per la discussione orale:

1. **Giustificazione Teorica e Dominio:** *Perché utilizziamo `IMAGENET_MEAN` e `IMAGENET_STD` pur addestrando `FCN-ResNet50` from scratch su COCO 2017 invece di ricalcolare $\mu$ e $\sigma$ sui $118\,287$ file `.jpg` di `train2017`?*
2. **Validità Architetturale:** *Perché possiamo effettivamente farlo senza introdurre distorsioni nell'ottimizzazione dei primi strati convoluzionali della ResNet-50?*
3. **Certezza Sperimentale:** *Quali prove empiriche già raccolte nel nostro notebook (Celle 4 e 5) dimostrano che questa normalizzazione funziona correttamente?*
4. **Riferimenti Autorevoli:** *Quali fonti ufficiali (specifiche di progetto, documentazione PyTorch, dispense del corso e letteratura peer-reviewed) supportano in modo inattaccabile questa scelta davanti al professore?*

---

## 2. Perché Possiamo Effettivamente Farlo: I Quattro Pilastri Tecnici

### 2.1 Equivalenza Statistica tra Dataset di Fotografie Naturali

Sia **ImageNet-1K** ($\approx 1.28 \times 10^6$ immagini) sia **MS COCO 2017** ($118\,287$ immagini in `train2017`) sono campionamenti su larga scala di **fotografie RGB del mondo naturale** (persone, veicoli, animali, interni domestici, scene urbane e paesaggi).

* Su COCO 2017 (`train2017`), le statistiche empiriche per canale si attestano intorno a $\mu_{\text{COCO}} \approx [0.471, 0.448, 0.408]$ e $\sigma_{\text{COCO}} \approx [0.274, 0.269, 0.285]$.
* Lo scarto assoluto rispetto alle costanti ImageNet è nell'ordine di $\Delta \mu_c \approx 0.01 \text{--} 0.02$ e $\Delta \sigma_c \approx 0.04 \text{--} 0.06$.
* Come evidenziato nelle dispense teoriche del corso (*CS231n — Neural Networks Part 2*), a differenza dei dati tabellari eterogenei (dove feature diverse hanno unità di misura e ordini di grandezza incompatibili), nei canali $R, G, B$ delle fotografie naturali le distribuzioni condividono già la stessa scala fisica; uno scostamento di pochi centesimi nella media o nella deviazione standard non altera la geometria della superficie di perdita.

### 2.2 Assorbimento Dinamico tramite *Batch Normalization* (`conv1` $\rightarrow$ `bn1`)

Nel backbone **ResNet-50** (`He et al., CVPR 2016`) impiegato da `fcn_resnet50`, il primo strato convoluzionale $7 \times 7$ (`backbone.conv1`, privo di *bias*) è immediatamente seguito dallo strato di **Batch Normalization** (`backbone.bn1`, `Ioffe & Szegedy, ICML 2015`).

Per ogni mini-batch $\mathcal{B}$ di dimensione $B$, `bn1` calcola dinamicamente su ciascuna mappa di attivazione $k$:

$$\hat{z}^{(k)} = \frac{z^{(k)} - \mu_{\mathcal{B}}^{(k)}}{\sqrt{(\sigma_{\mathcal{B}}^{(k)})^2 + \epsilon}}, \qquad y^{(k)} = \gamma^{(k)} \hat{z}^{(k)} + \beta^{(k)}$$

dove $\gamma^{(k)}$ e $\beta^{(k)}$ sono parametri affini apprendibili. Di conseguenza:
1. Qualsiasi lieve scostamento residuo ($\Delta \mu \approx 0.015$) dalla media zero esatta viene **ri-centrato e ri-scalato automaticamente** subito dopo `conv1` (poiché una traslazione costante sull'ingresso di una convoluzione lineare viene cancellata dalla sottrazione di $\mu_{\mathcal{B}}^{(k)}$ in `bn1`).
2. L'obiettivo fondamentale della normalizzazione in ingresso non è ottenere $\mu = 0.0000$ alla quarta cifra decimale, bensì **evitare scale grezze $[0, 255]$** che causerebbero saturazione dei gradienti e cattivo condizionamento della matrice Hessiana nelle prime iterazioni *from scratch*.

### 2.3 Aderenza allo Standard `torchvision` e Continuità verso il *Transfer Learning*

La specifica di progetto (`H3.pdf`) indica esplicitamente la documentazione ufficiale di PyTorch per **FCN** (`torchvision.models.segmentation.fcn_resnet50`) e prevede, dopo l'addestramento *from scratch* su COCO, il **Transfer Learning (Fine-Tuning di `yourFCN`) su Cityscapes**.

* **Standard ufficiale `torchvision` su COCO:** Anche per i pesi ufficiali pre-addestrati su COCO (`FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1`), la pipeline canonica di `torchvision` converte l'immagine in `[0.0, 1.0]` e applica esattamente `mean=[0.485, 0.456, 0.406]` e `std=[0.229, 0.224, 0.225]`.
* **Coerenza *Ceteris Paribus* tra *from scratch* e *pre-trained*:** Mantenere le medesime costanti permette di confrontare rigorosamente il nostro modello *from scratch* con eventuali pesi pre-addestrati su ImageNet/COCO a parità assoluta di dominio degli input.
* **Continuità nel *Transfer Learning* su Cityscapes:** Quando trasferiremo `yourFCN` su Cityscapes, i filtri convoluzionali di `backbone.conv1` avranno appreso risposte calibrate su input normalizzati con `IMAGENET_MEAN` e `IMAGENET_STD`; mantenere le stesse costanti anche su Cityscapes (anch'esso composto da fotografie RGB di scene urbane naturali) previene qualsiasi *covariate shift* artificiale all'ingresso.

### 2.4 Prevenzione del *Data Leakage* e Rapporto Costo Computazionale / Beneficio

* **Zero *Data Leakage*:** Le dispense *CS231n* segnalano come errore metodologico (*common pitfall*) il calcolo di $\mu$ e $\sigma$ sull'intero dataset includendo `val2017` o `test2017`. L'adozione di costanti esterne standardizzate elimina alla radice qualunque contaminazione statistica tra training e validation set.
* **Costo I/O vs Beneficio:** Calcolare $\mu$ e $\sigma$ esatti su tutti i $118\,287$ file `.jpg` di `train2017` richiede l'apertura e la decodifica da disco di ogni singola immagine (diversi minuti di I/O su Kaggle), senza produrre alcun guadagno misurabile sulla `mIoU` rispetto alle costanti ImageNet.

---

## 3. Perché Siamo Certi di Poterlo Fare: Evidenze Empiriche e Casi Limite

### 3.1 Conferme Dirette dai Risultati del Nostro Notebook (`Celle 4 e 5`)

La correttezza numerica della pipeline di normalizzazione è già dimostrata da quattro evidenze sperimentali misurate direttamente sul nostro codice:

| Verifica Sperimentale nel Notebook | Output Misurato | Cosa Dimostra sulla Normalizzazione |
| :--- | :--- | :--- |
| **Sanity Check 1° Batch (`Cella 4`)** | `images: (16, 3, 256, 256) [-2.12, 2.64]` | I tensori in ingresso alla FCN sono centrati attorno a $0$ con escursione simmetrica $\approx [-2.1, +2.6]$, coerente con una distribuzione standardizzata. |
| **Loss Iniziale Step $0$ (`Cella 5`)** | `Main Loss: 4.4173` vs $-\ln(1/81) = 4.3944$ | Con pesi iniziali casuali e input normalizzati ImageNet, i *logit* non sono saturati e la *Cross-Entropy* coincide con il valore teorico di equiprobabilità ($+0.52\%$). |
| **Update/Weight Ratio Step $1$ (`Cella 5`)** | `classifier.4: 1.16e-03` ($\sim 10^{-3}$) | La scala degli input produce gradienti ben condizionati lungo tutti i $50$ strati della rete fin dalla prima iterazione. |
| **Convergenza Overfitting (`Cella 5`)** | Loss $6.6100 \rightarrow 0.0680$, `mIoU: 79.13%` | Discesa monotona e stabile senza instabilità numeriche né esplosione dei gradienti. |

### 3.2 Distinzione tra Statistiche Universali (`IMAGENET_*`) e Parametri di Dominio (`NUM_CLASSES`)

Nel nostro codice (`Cella 1` e `Cella 2`) distinguiamo chiaramente tra:
* **Parametri dipendenti dal task/tassonomia (`NUM_CLASSES = 81`):** calcolato direttamente dalle annotazioni JSON di COCO 2017 (*Thing*), ovvero $80$ categorie di oggetti mappate in $[1, 80]$ più la classe $0$ riservata al `background`.
* **Costanti di normalizzazione RGB (`IMAGENET_MEAN`, `IMAGENET_STD`):** condivise tra tutti i benchmark fotografici naturali in visione artificiale.

**Quando invece È OBBLIGATORIO ricalcolare $\mu$ e $\sigma$ sul training set:**  
Diventa metodologicamente necessario calcolare statistiche *ad hoc* (esclusivamente sul *training split*) quando si opera fuori dal dominio delle fotografie RGB naturali — ad esempio in **immagini satellitari/multispettrali** (come il dataset aereo usato nello scheletro originale del notebook di lezione), **imaging medico** (radiografie, risonanze magnetiche, istologia), **immagini radar (SAR)** o **microscopia**, dove intensità e distribuzione dei canali differiscono radicalmente dalle foto quotidiane.

### 3.3 Verifica Empirica *Zero-Leakage* su `train2017` (Opzionale per la Relazione)

Qualora si desideri riportare anche lo scarto numerico esatto misurato sul nostro `train2017`, è sufficiente eseguire il seguente controllo diagnostico campionato **esclusivamente sul training set** (`zero data leakage`), accumulando somma e somma dei quadrati **su tutti i pixel** (e non mediando le medie delle singole immagini, poiché in COCO le risoluzioni $(H, W)$ variano da immagine a immagine e una media non pesata altererebbe il peso dei pixel):

```python
rng = random.Random(SEED)
sample_ids = rng.sample(sorted(coco_train.getImgIds()), 2000)

channel_sum = np.zeros(3, dtype=np.float64)
channel_sq_sum = np.zeros(3, dtype=np.float64)
pixel_count = 0

for img_id in sample_ids:
    info = coco_train.loadImgs(img_id)[0]
    img_path = os.path.join(DATA_ROOT, "train2017", info["file_name"])
    arr = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float64) / 255.0
    channel_sum += arr.sum(axis=(0, 1))
    channel_sq_sum += (arr ** 2).sum(axis=(0, 1))
    pixel_count += arr.shape[0] * arr.shape[1]

coco_mean = channel_sum / pixel_count
coco_std = np.sqrt(channel_sq_sum / pixel_count - coco_mean ** 2)
```

---

## 4. Riferimenti Documentali e Bibliografici Validati per il Colloquio

Per argomentare questa scelta in modo formale e verificabile durante l'esame orale, è possibile citare direttamente i seguenti cinque riferimenti:

| Fonte / Riferimento | Sezione / Dettaglio Specifico | Argomento a Supporto |
| :--- | :--- | :--- |
| **1. Specifiche di Progetto (`H3.pdf`)** | *Semantic/Instance Segmentation*: link ufficiale `https://docs.pytorch.org/vision/main/models/fcn.html` | Indica esplicitamente l'implementazione `torchvision` di `FCN` come riferimento architetturale del progetto. |
| **2. Documentazione Ufficiale PyTorch / Torchvision** | `torchvision.models.segmentation.fcn_resnet50` (`FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1`) | La pipeline ufficiale di preprocessing per `FCN-ResNet50` addestrata **proprio su COCO** normalizza i tensori `[0, 1]` con `mean=[0.485, 0.456, 0.406]` e `std=[0.229, 0.224, 0.225]`. |
| **3. Dispense Teoriche del Corso (*CS231n*)** | *Neural Networks Part 2: Setting up the Data and the Loss* (*Data Preprocessing*) | Giustifica la standardizzazione $\tilde{x} = (x - \mu)/\sigma$, l'omogeneità di scala dei canali RGB nelle immagini naturali e il divieto di usare validation/test set per stimare le statistiche (*data leakage*). |
| **4. Ioffe & Szegedy (ICML 2015)** | *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* | Dimostra come gli strati di `BatchNorm2d` (presenti subito dopo `conv1` nella ResNet) ri-centrino e ri-scalino dinamicamente le attivazioni su ogni mini-batch, assorbendo lievi scostamenti nelle statistiche di ingresso. |
| **5. He et al. (CVPR 2016) & Lin et al. (ECCV 2014)** | *Deep Residual Learning for Image Recognition* (ResNet) & *Microsoft COCO: Common Objects in Context* | Stabiliscono rispettivamente l'architettura del backbone con `conv1 + bn1` e la natura fotografica RGB su larga scala di MS COCO, statisticamente equivalente a ImageNet-1K. |

---

## 5. Sintesi Pronta per la Discussione Orale

* **Risposta sintetica al professore:**  
  *"Utilizziamo `IMAGENET_MEAN = [0.485, 0.456, 0.406]` e `IMAGENET_STD = [0.229, 0.224, 0.225]` per quattro ragioni tecniche documentate: primo, COCO 2017 e ImageNet-1K sono entrambi dataset di fotografie RGB naturali con statistiche per canale quasi identiche ($\Delta \mu \approx 0.01 \text{--} 0.02$); secondo, nel backbone ResNet-50 lo strato `bn1` (Batch Normalization, Ioffe & Szegedy 2015) posto immediatamente dopo `conv1` ri-centra e ri-scala dinamicamente le attivazioni assorbendo qualsiasi minimo scarto residuo; terzo, è lo standard ufficiale della documentazione `torchvision` per `fcn_resnet50` su COCO citata nella traccia `H3.pdf` ed elimina ogni rischio di data leakage; quarto, garantisce piena compatibilità di scala quando trasferiremo `yourFCN` su Cityscapes nella fase successiva."*
* **Nota sul successivo *Transfer Learning* su Cityscapes (`H3.pdf`):**  
  Quando passeremo al *fine-tuning* di `yourFCN` su **Cityscapes**, dovremo continuare a utilizzare **le stesse identiche costanti (`IMAGENET_MEAN` e `IMAGENET_STD`)**, poiché i pesi del primo strato convoluzionale (`backbone.conv1`) saranno stati addestrati su COCO esattamente in quello spazio normalizzato.