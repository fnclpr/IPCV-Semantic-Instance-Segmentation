# \[02\_QA\] Step 2 — Preprocessing Geometrico (**Opzione C**), Scelta di `CROP_SIZE = 256`, Architettura FCN-ResNet50 e Validazione Deterministica

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Riferimento Codice:** `Notebook1_SetupDatasetCOCO2017_Babysitting_(Step1_Step3-BIS)` — Celle 1, 3, 4 e 5  
**Collegamento Progettuale:** Step 1 (*Profilazione Risoluzioni*) $\rightarrow$ Step 2 (*`COCOSemanticDataset`*) $\rightarrow$ Step 3-BIS (*Mini-Ablation Risoluzione*) $\rightarrow$ Step 6 (*Transfer Learning su Cityscapes*)  
**Tipo di Documento:** Scheda di Approfondimento Teorico-Metodologico (Q&A)  

---

## 1. Quesiti Metodologici Formalizzati

Questo documento affronta in modo rigoroso e critico i cinque nodi progettuali relativi alla geometria dei dati in ingresso e al loro legame con l'architettura **FCN-ResNet50**:

1. **Strategia Geometrica (**Opzione C**):** *Che cosa facciamo fisicamente ai pixel in `COCOSemanticDataset._apply_synchronized_transforms` (Cella 3) e perché l'Opzione C è superiore alle alternative su COCO 2017?*
2. **Confronto Critico con le Alternative:** *Che cosa sarebbe successo adottando il ritaglio puro dello scheletro di lezione (**Opzione A**) oppure il ridimensionamento quadrato diretto (**Opzione B**)?*
3. **Giustificazione della Soglia `CROP_SIZE = 256`:** *Sulla base di quali criteri abbiamo fissato `256` pixel pur avendo rilevato in COCO immagini più piccole (fino a $230 \times 246\text{ px}$)?*
4. **Fondamenti Architetturali di `FCN-ResNet50`:** *Che cosa significano in pratica termini come Fully Convolutional Network, Output Stride ($s = 8$), Convoluzioni Dilatate (Atrous), Campo Recettivo (Receptive Field) e Upsampling Bilineare, e come interagiscono con i dati di COCO?*
5. **Validazione Deterministica e Gestione Memoria:** *Perché in validazione sostituiamo `RandomCrop` con `CenterCrop` deterministico, come valuteremo l'intera immagine a fine progetto e perché adottiamo una strategia ibrida RAM/Disco?*

---

## 2. Che Cosa Facciamo Fisicamente alle Immagini (**Opzione C**)

Nella **Cella 3** (`COCOSemanticDataset._apply_synchronized_transforms`), ogni coppia `(immagine RGB, maschera semantica)` subisce una pipeline geometrica a due stadi rigorosamente sincronizzata tramite l'API funzionale `torchvision.transforms.functional` (`TF`):

### 2.1 Stadio 1 — Ridimensionamento Proporzionale del Lato Corto (`TF.resize(..., size=256)`)

* **Passaggio di un intero (`size=256`) vs tupla (`size=(256, 256)`):** In `torchvision`, passare un singolo **intero** `256` a `TF.resize` ridimensiona il **lato più corto** dell'immagine a $256\text{ px}$ e scala l'altro lato dello stesso identico fattore $r = 256 / \min(H, W)$, preservando al $100\%$ l'**aspect ratio** originale ($W/H$). *(Al contrario, passare la tupla `(256, 256)` forzerebbe un ridimensionamento anisotropo deformante).*
* **Esempi numerici reali sul nostro dataset:**
  * **Immagine mediana di COCO ($480 \times 640\text{ px}$, rapporto $4:3$):** il lato corto è $480\text{ px}$ ($r = 256 / 480 \approx 0.533$). L'immagine diventa **$256 \times 341\text{ px}$**.
  * **Immagine più piccola rilevata in COCO ($230 \times 246\text{ px}$, Step 1):** il lato corto è $230\text{ px}$ ($r = 256 / 230 \approx 1.113$). L'immagine viene ingrandita leggermente a **$256 \times 274\text{ px}$**, portando entrambi i lati $\ge 256\text{ px}$ **senza aggiungere alcun bordo fittizio**.
* **Differenziazione delle modalità di interpolazione:**
  * **Immagine RGB:** `InterpolationMode.BILINEAR` con **`antialias=True`** (previene artefatti di *aliasing* ad alta frequenza durante il sottocampionamento di texture fini come erba, ringhiere o tessuti).
  * **Maschera semantica:** rigorosamente **`InterpolationMode.NEAREST`**. Se si usasse l'interpolazione bilineare su una maschera di indici discreti, sul confine tra la classe `1` (`person`) e la classe `5` (`airplane`) verrebbe calcolata la media $(1 + 5)/2 = 3$, generando una **classe fittizia inesistente (`3: car`)** lungo tutto il contorno dell'oggetto.

### 2.2 Stadio 2 — Ritaglio e *Data Augmentation* Sincronizzati

Dopo il resize proporzionale, almeno uno dei due lati misura esattamente $256\text{ px}$ e l'altro misura $\ge 256\text{ px}$:

* **In Training (`is_train=True`):**
  1. `transforms.RandomCrop.get_params(img, output_size=(256, 256))` estrae una sola coppia di coordinate $(i, j)$ e `TF.crop` ritaglia **esattamente la stessa finestra** $256 \times 256$ su immagine e maschera. Sull'immagine mediana $256 \times 341$, la finestra scorre lungo i $341\text{ px}$ orizzontali inquadrando **il $75.1\%$ dell'intera scena** e fungendo da *data augmentation* traslazionale naturale.
  2. `if torch.rand(1).item() < 0.5:` applica `TF.hflip` (ribaltamento orizzontale speculare) a entrambi i tensori. L'uso di `torch.rand(1)` al posto di `random.random()` garantisce che l'intera stocasticità sia governata dal `torch.Generator` passato al `DataLoader`.
  3. **Asimmetria fotometrica:** La normalizzazione per canale (`TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)`) viene applicata **esclusivamente** all'immagine RGB, lasciando intatti gli indici interi $0 \dots 80$ della maschera.
* **In Validation (`is_train=False`):**
  * Viene applicato **`TF.center_crop(img, (256, 256))`** deterministico (nessun flip, nessuno spostamento casuale).

---

## 3. Analisi Comparativa Critica: Opzione C vs Opzione A e Opzione B

Per giustificare davanti al professore perché abbiamo contestualizzato lo scheletro di lezione anziché copiarlo meccanicamente, confrontiamo le tre alternative progettuali possibili:

| Criterio di Confronto | **Opzione A: Crop Puro Scheletro** (`Pad` se $<256$ + `RandomCrop(256)`) | **Opzione B: Resize Quadrato Diretto** (`Resize((256, 256))` anisotropo) | **Opzione C: Scelta Adottata** (`Resize(lato_corto=256)` + `Crop(256)`) |
| :--- | :--- | :--- | :--- |
| **Gestione immagini $< 256\text{ px}$** ($822$ img in `train2017`, $42$ in `val2017`) | Senza `Pad` lancia **`ValueError`**; con `Pad` introduce cornici artificiali `IGNORE_INDEX = 255`. | Nessun `ValueError` e nessun padding artificiale. | Nessun `ValueError` e **nessun padding artificiale `255`**. |
| **Proporzioni (*Aspect Ratio*)** | $100\%$ preservate. | **Distorte** ($4:3 \rightarrow 1:1$): schiaccia orizzontalmente auto, persone e animali. | **$100\%$ preservate** (fattore di scala identico su $H$ e $W$). |
| **Porzione di scena visibile** (su immagine mediana $480 \times 640$) | Solo **$21.3\%$** dell'area (*effetto buco della serratura*). | $100\%$ dell'area, ma con geometria alterata. | **$75.1\%$** dell'area con geometria naturale intatta. |
| **Rischio patch vuote (solo `background`)** | **Alto:** in COCO *Thing* l'$80\text{--}90\%$ dei pixel è sfondo (`0`); molti crop pescano solo muro/cielo. | Minimo (l'intera scena è sempre inquadrata). | **Basso:** la finestra copre tre quarti dell'immagine. |
| **Comportamento su oggetti grandi in primo piano** | **Critico:** un elefante o un divano $400 \times 500\text{ px}$ appare solo come texture interna senza bordi. | Vede l'intera sagoma, ma deformata. | **Ottimale:** inquadra l'intera sagoma con proporzioni reali. |
| **Comportamento su oggetti molto piccoli (*Small Objects*)** | Preserva la risoluzione nativa degli oggetti piccoli che cadono nel crop. | Riduce gli oggetti piccoli e ne altera la forma. | **Limite noto (analizzato criticamente):** riduce l'area mediana di $\sim 3.5\times$, portando il **$15.21\%$** delle istanze sotto $8\text{ px}$ equivalenti. |
| **Compatibilità con *Transfer Learning* su Cityscapes (Step 6)** | Buona (proporzioni intatte), ma filtri addestrati su viste molto ravvicinate. | **Scarsa:** Cityscapes ha formato panoramico $2048 \times 1024$ ($2:1$); i filtri hanno appreso forme compresse. | **Elevata:** preserva la geometria reale degli oggetti prima del *fine-tuning* su Cityscapes. |

### Valutazione Critica Onesta dell'Opzione C

In sede d'esame è fondamentale non presentare l'Opzione C come priva di compromessi, ma mostrare di averne **misurato quantitativamente il limite**:
* Ridimensionando un'immagine $480 \times 640$ a $256 \times 341$, le dimensioni lineari si riducono di un fattore $1.875\times$ (area $\approx 3.52\times$).
* Nella **Cella 3** (`analyze_coco_split`), abbiamo misurato direttamente che **`130,797 / 860,001` istanze in `train2017` ($15.21\%$) e `5,526 / 36,781` in `val2017` ($15.02\%$)** scendono sotto un lato equivalente $\sqrt{A_{\text{post}}} < 8\text{ px}$.
* Come vedremo nella Sezione 5, questo fenomeno interagisce direttamente con l'**Output Stride $s = 8$** della `FCN-ResNet50` e spiega perché nella **Riga 1** di `05_step3_babysitting_overfit_predictions.png` le mucche distanti vengono segmentate solo parzialmente.

---

## 4. Perché Abbiamo Scelto Esattamente `CROP_SIZE = 256` (Anche in Presenza di Immagini $< 256\text{ px}$)

La scelta di `CROP_SIZE = 256` non è un retaggio passivo dello scheletro del professore, bensì il punto di equilibrio tra **quattro vincoli fisici, architetturali e sperimentali**:

### 4.1 Vincolo Fisico di Batching su GPU (`133` Risoluzioni Distinte)

Per impilare $B$ immagini in un unico tensore PyTorch $(B, 3, H, W)$ ed eseguire operazioni matriciali parallele su GPU, **tutti i campioni di un mini-batch devono possedere esattamente le stesse dimensioni spaziali $(H, W)$**. Poiché nella **Cella 1** abbiamo rilevato **$133$ risoluzioni diverse su $500$ immagini** ($H \in [230, 640]$, $W \in [246, 640]$), è obbligatorio fissare una dimensione target comune $S$.

### 4.2 Perché Non Ridurre la Soglia Sotto `256` (es. `224` o `192 px`) per Includere le Immagini Minime?

* Dalla **Cella 3**, le immagini con lato corto $< 256\text{ px}$ sono soltanto **$822$ su $118\,287$ in `train2017` ($0.69\%$)** e **$42$ su $5\,000$ in `val2017` ($0.84\%$)**, con minimo assoluto $230 \times 246\text{ px}$.
* Abbassare `CROP_SIZE` a `224` o `192 px` per evitare un upscaling minimo ($1.11\times$) sullo **$0.69\%$** dei dati avrebbe significato **degradare ulteriormente la risoluzione del restante $99.31\%$ del dataset**, facendo salire ben oltre il $20\%$ la quota di oggetti piccoli irriconoscibili.
* Con `CROP_SIZE = 256`, il $99.31\%$ delle immagini subisce un downsampling controllato mentre il restante $0.69\%$ subisce un upscaling lievissimo ($1.004\times \text{--} 1.113\times$) privo di sfocature significative.

### 4.3 Vincolo Architetturale di Divisibilità (`Output Stride = 8` e `32`)

Il backbone **ResNet-50** dimezza progressivamente la risoluzione attraverso stadi di downsampling a potenze di $2$ ($/2, /4, /8$, e nelle ResNet standard $/16, /32$). Fissare $S = 256 = 2^8$ garantisce che ogni divisione spaziale restituisca un intero esatto ($256 \rightarrow 128 \rightarrow 64 \rightarrow 32$), evitando qualsiasi disallineamento di fase o arrotondamento asimmetrico durante l'upsampling finale $32 \times 32 \rightarrow 256 \times 256$.

### 4.4 Legge Quadratica VRAM/Tempo ($O(H \cdot W)$) e Validazione Empirica nello Step 3-BIS

Il numero di pixel da elaborare in ogni forward/backward pass cresce **quadraticamente** con il lato $S$:

| Risoluzione Candidata | Pixel per Immagine ($S^2$) | Fattore Rispetto a $256$ | Griglia Latente FCN ($S/8 \times S/8$) | Evidenza Empirica Misurata nel Notebook |
| :--- | :---: | :---: | :---: | :--- |
| **`256 x 256` (Scelta)** | $65\,536$ | **$1.00\times$** | $32 \times 32$ ($1\,024$ celle) | **`BATCH_SIZE = 16` usa $5.17\text{ GB}$ VRAM** (Cella 4); in Step 3-BIS (`T1`) ottiene **`mIoU Obj = 42.38%`** in **$106.1\text{ s}$**. |
| **`384 x 384` (Testata in `T3`)** | $147\,456$ | **$2.25\times$** | $48 \times 48$ ($2\,304$ celle) | Tempo esplode a **$455.8\text{ s}$ ($4.30\times$)** e **`mIoU Obj` scende a `36.01%`** (buchi interni sugli oggetti estesi). |
| **`512 x 512` (Scartata)** | $262\,144$ | **$4.00\times$** | $64 \times 64$ ($4\,096$ celle) | Supererebbe la VRAM disponibile con `BATCH_SIZE = 16` ($\sim 20\text{ GB} > 14.56\text{ GB}$ per Tesla T4), quadruplicando i tempi dello Step 4. |

La mini-ablation dello **Step 3-BIS (`Cella 5`)** fornisce dunque la **prova sperimentale definitiva** che `CROP_SIZE = 256` rappresenta l'ottimo globale di compromesso tra accuratezza semantica, campo recettivo e tempi di calcolo.

---

## 5. Glossario Architetturale: Come Funzionano ResNet-50 e FCN sui Nostri Dati

In sede di colloquio, è cruciale saper spiegare cosa avviene all'interno di `torchvision.models.segmentation.fcn_resnet50` quando riceve un tensore $(B, 3, 256, 256)$ e quali implicazioni pratiche ne derivano sul dataset COCO:

### 5.1 Cosa Significa *Fully Convolutional Network* (FCN — Long et al., CVPR 2015)

* **Definizione:** Una rete convoluzionale tradizionale per classificazione (come la ResNet-50 originale) termina con un *Global Average Pooling* e un layer *Fully Connected* (`Linear(2048, 1000)`), che collassa l'intera immagine in un singolo vettore $1\text{D}$ distruggendo la posizione spaziale degli oggetti. Una **FCN** sostituisce tutti i layer densi con **convoluzioni $1 \times 1$** (`Conv2d(2048, 512, 3x3)` $\rightarrow$ `Conv2d(512, 81, 1x1)`).
* **Implicazione pratica 1 (Predizione densa):** La rete conserva la topologia $2\text{D}$ e produce un tensore di *logit* $\hat{Y} \in \mathbb{R}^{B \times 81 \times H \times W}$, classificando simultaneamente tutti i $65\,536$ pixel in un solo passaggio.
* **Implicazione pratica 2 (Indipendenza dalla risoluzione di ingresso — *Resolution-Agnostic*):** Poiché i filtri convoluzionali scorrono localmente sull'immagine senza pesi legati a una matrice di dimensioni fisse, una FCN addestrata su crop $256 \times 256$ **può essere valutata in inferenza su immagini a risoluzione arbitraria** (ad esempio l'intera foto originale COCO $480 \times 640$ o un fotogramma panoramico Cityscapes $1024 \times 2048$ con `batch_size=1`).

### 5.2 Backbone ResNet-50 e Stadi Gerarchici (`conv1` $\rightarrow$ `layer1..4`)

Il *backbone* (spina dorsale estrattore di feature) è composto da $50$ strati convoluzionali raggruppati in blocchi residui (*Bottleneck* con connessioni *skip* $y = \mathcal{F}(x) + x$ che prevengono la scomparsa del gradiente nei modelli profondi addestrati *from scratch*):

* **Stadi iniziali (`conv1` + `maxpool` + `layer1`, stride $/4$, risoluzione $64 \times 64$):** Estraggono feature geometriche a basso livello (bordi orientati, contrasti cromatici, texture locali) con elevata precisione spaziale ma bassa astrazione semantica.
* **Stadi intermedi (`layer2` + `layer3`, stride $/8$, risoluzione $32 \times 32$, $1\,024$ canali):** Catturano parti di oggetti (ruote, finestre, zampe, volti). Proprio all'uscita di `layer3` si innesta il **ramo ausiliario (`aux_classifier`)** pesato a `AUX_LOSS_WEIGHT = 0.5`, che inietta gradiente diretto negli strati intermedi stabilizzando il training *from scratch*.
* **Stadio profondo (`layer4`, $2\,048$ canali):** Codifica l'identità semantica globale dell'oggetto e il contesto della scena.

### 5.3 Convoluzioni Dilatate (*Atrous Convolutions*) e *Output Stride* ($s = 8$)

* **Il problema del downsampling nella ResNet-50 standard:** Nella ResNet-50 per classificazione, anche `layer3` e `layer4` applicano stride $2$, riducendo l'immagine di un fattore **$32\times$** (*Output Stride* $s = 32$). Un ingresso $256 \times 256$ diventerebbe una griglia microscopica $8 \times 8$, troppo grossolana per ricostruire i contorni degli oggetti.
* **La soluzione adottata da `fcn_resnet50` (`Output Stride = 8`):** In `torchvision`, gli stadi `layer3` e `layer4` **disattivano lo stride $2$** (mantenendo la risoluzione spaziale bloccata a $H/8 \times W/8 = 32 \times 32$) e inseriscono al suo posto **convoluzioni dilatate (*atrous*, con *dilation rate* $d = 2$ in `layer3` e $d = 4$ in `layer4`)**, che inseriscono $d - 1$ zeri tra i pesi del kernel $3 \times 3$. In questo modo il filtro espande il proprio campo visivo senza ridurre la risoluzione della griglia e senza aggiungere parametri.
* **Implicazione pratica sui dati COCO ($15.21\%$ istanze $< 8\text{ px}$):** Poiché ogni cella della mappa latente $32 \times 32$ corrisponde a un blocco di **$8 \times 8 = 64\text{ pixel}$** nell'immagine di ingresso, qualsiasi istanza di COCO che dopo il resize ha diametro inferiore a $8\text{ px}$ ($15.21\%$ del totale) occupa **meno di una singola cella latente** e tende a essere assorbita dallo sfondo circostante.

### 5.4 Campo Recettivo (*Receptive Field*) e Perché `T3 (384 px)` Ha Fallito sugli Oggetti Grandi

* **Cos'è il Campo Recettivo:** È l'area dell'immagine di ingresso $(H_{\text{RF}} \times W_{\text{RF}})$ che influenza l'attivazione di un singolo neurone nell'ultimo strato `layer4`. Il campo recettivo dipende **esclusivamente dall'architettura** (numero di strati, kernel $3 \times 3$, dilation rate), ed è quindi **fisso in pixel**.
* **Implicazione pratica osservata nello Step 3-BIS (`06_step3bis_mini_ablation_visual_comparison.png`):** Quando nello Step 3-BIS (`T3`) abbiamo ingrandito l'immagine da $256\text{ px}$ a $384\text{ px}$, un oggetto esteso come la fusoliera dell'aereo militare (Riga 4) è diventato più grande del campo recettivo effettivo della rete: i neuroni al centro della fusoliera vedevano solo lamiera grigia uniforme senza scorgere le ali o i contorni dell'aereo, generando **buchi interni** nella predizione e facendo crollare la `mIoU Obj` dal $42.38\%$ (`T1`, $256\text{ px}$) al $36.01\%$ (`T3`, $384\text{ px}$).

### 5.5 Upsampling Bilineare vs Assenza di *Skip Connections* (FCN-8s / U-Net)

* A differenza di **U-Net** (usata nello scheletro del professore), che fonde feature ad alta risoluzione ($128 \times 128$, $64 \times 64$) provenienti dall'encoder tramite *skip connections*, l'implementazione standard `fcn_resnet50` di `torchvision` risale direttamente dalla mappa dei *logit* $32 \times 32$ alla risoluzione $256 \times 256$ tramite **interpolazione bilineare** (`F.interpolate(..., size=(256, 256), mode='bilinear', align_corners=False)`).
* **Implicazione pratica:** Questa semplicità rende la rete veloce e leggera in VRAM, ma spiega perché le maschere predette (es. `05_step3_babysitting_overfit_predictions.png`) presentino **contorni leggermente arrotondati** attorno alle zampe dei cani o ai bordi delle auto rispetto alla *ground truth* poligonale.

---

## 6. Perché la Validazione Deve Essere Deterministica (`CenterCrop`) e Come Valutare l'Intera Immagine

### 6.1 Difetto del `RandomCrop` in Validazione (Sezioni 10–11 dello Scheletro)

Come evidenziato esplicitamente nello scheletro del Kaggle Notebook (*"What is wrong with RandomCrop validation?"*), usare `RandomCrop(256, 256)` durante la validazione rende le curve di `val_loss` e `val_mIoU` **rumorose e non confrontabili** tra epoche diverse o tra funzioni di perdita diverse: se all'epoca $t$ il crop casuale inquadra un oggetto facile e all'epoca $t+1$ inquadra uno sfondo ambiguo o un oggetto minuscolo al bordo, la metrica oscilla per pura varianza di campionamento anziché per reale apprendimento dei pesi.

Con **`TF.resize(size=256)` + `TF.center_crop((256, 256))`** in `COCOSemanticDataset(is_train=False)`:
1. Ogni immagine di `val2017` ($5\,000$ campioni) espone **esattamente gli stessi identici pixel** in ogni epoca e in ogni esperimento dello Step 4.
2. Ogni variazione della `val_mIoU` è imputabile **al $100\%$ all'aggiornamento dei pesi del modello** (*ceteris paribus*).

### 6.2 Come Valuteremo l'Intera Immagine a Fine Progetto (Domanda *Discussion* dello Scheletro)

Il `CenterCrop(256, 256)` in validazione taglia comunque circa il $25\%$ laterale delle immagini panoramiche ($256 \times 341 \rightarrow 256 \times 256$). Nella fase di valutazione finale (Step 9), sfrutteremo la proprietà **resolution-agnostic** della FCN per valutare **l'intero fotogramma senza alcun crop** (`batch_size=1`):
* Ridimensioniamo solo il lato corto a $256\text{ px}$ (`256 x 341`, oppure multiplo di $8$), passiamo il tensore $(1, 3, 256, W')$ attraverso `model(img)` e confrontiamo la mappa predetta $(256, W')$ con l'intera maschera ridimensionata $(256, W')$, recuperando il $100\%$ degli oggetti periferici.

### 6.3 Strategia Ibrida Memoria/Disco (`CACHE_IN_RAM` vs `preload()` dello Scheletro)

Nello scheletro del professore, `preload()` carica l'intero dataset in RAM prima del training. Su COCO 2017 (`118,287` immagini), precaricare tutti i tensori occuperebbe $> 95\text{ GB}$ di RAM causando un crash *Out-Of-Memory* (OOM). Nel nostro `COCOSemanticDataset` (Celle 1 e 3):
* **`CACHE_IN_RAM = True`** è attivato **esclusivamente** per subset controllati $\le 15\,000$ immagini (`16` immagini nel *babysitting* in $0.1\text{ s}$; `12\,000` immagini nello Step 4 $\approx 9.5\text{ GB}$ RAM), eliminando il collo di bottiglia I/O del disco su epoche ripetute.
* **`CACHE_IN_RAM = False`** (caricamento *on-the-fly* in `__getitem__`) è attivato sull'intero `train2017` (`118,287` immagini), garantendo zero rischio OOM.

---

## 7. Riferimenti Documentali e Bibliografici Validati per il Colloquio

| Fonte / Riferimento | Dettaglio Specifico | Argomento a Supporto |
| :--- | :--- | :--- |
| **1. Long, Shelhamer & Darrell (CVPR 2015)** | *Fully Convolutional Networks for Semantic Segmentation* | Introduce la sostituzione dei layer densi con convoluzioni $1 \times 1$, l'upsampling bilineare e la capacità di processare input di risoluzione arbitraria. |
| **2. He et al. (CVPR 2016)** | *Deep Residual Learning for Image Recognition* (ResNet-50) | Definisce i blocchi residui *Bottleneck* e la gerarchia spaziale degli stadi `conv1` $\rightarrow$ `layer4` che consente il training stabile *from scratch*. |
| **3. Chen et al. (ICLR 2015 / TPAMI 2018)** | *Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs* (DeepLab — Convoluzioni *Atrous*) | Formalizza l'uso delle **convoluzioni dilatate (*atrous*)** per mantenere l'*Output Stride* $s = 8$ espandendo il campo recettivo senza ridurre la risoluzione spaziale. |
| **4. Documentazione PyTorch / Torchvision** | `torchvision.models.segmentation.fcn_resnet50` & `torchvision.transforms.functional.resize` | Documenta l'architettura con `output_stride=8`, il classificatore ausiliario su `layer3` e la semantica `size=int` (lato corto) vs `size=tuple`. |
| **5. Kaggle Notebook di Riferimento (`Sezioni 7, 10, 11, Discussion`)** | Sincronizzazione `TF`, difetti del `RandomCrop` in validazione e valutazione dell'intera immagine | Giustifica l'uso di `CenterCrop` deterministico durante il training comparativo e dell'inferenza *full-image* (`batch_size=1`) nella valutazione finale. |

---

## 8. Sintesi Pronta per la Discussione Orale

* **Risposta sintetica al professore:**  
  *"In COCO 2017 abbiamo rilevato 133 risoluzioni diverse con un 0.69% di immagini aventi lato corto inferiore a 256 px (fino a 230 x 246 px). Abbiamo quindi adottato una pipeline geometrica sincronizzata (Opzione C) che ridimensiona proporzionalmente il lato corto a 256 px (usando `BILINEAR + antialias` sull'immagine RGB e rigorosamente `NEAREST` sulla maschera per non generare classi interpolate spurie), seguita da `RandomCrop(256, 256) + HorizontalFlip` in training e `CenterCrop(256, 256)` deterministico in validazione. Questa scelta preserva al 100% l'aspect ratio (fondamentale anche per il successivo transfer learning su Cityscapes), evita bordi artificiali di padding e rispetta la divisibilità per l'Output Stride s = 8 della FCN-ResNet50 (griglia latente 32 x 32 ottenuta tramite convoluzioni dilatate atrous in layer3 e layer4). Infine, la mini-ablation dello Step 3-BIS ha dimostrato empiricamente che salire a 384 px quadruplica i tempi di calcolo e degrada la mIoU sugli oggetti estesi a causa del campo recettivo fisso della rete, confermando 256 px come configurazione ottimale."*