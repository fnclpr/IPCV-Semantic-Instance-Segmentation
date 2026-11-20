# \[01\_QA\] Step 1–2 — Scelta di COCO 2017 (*Thing* vs *Stuff*), Mappatura Classi e Rasterizzazione delle Maschere

**Progetto:** Semantic & Instance Segmentation (Specifiche H3)  
**Riferimento Codice:** `Notebook1_SetupDatasetCOCO2017_Babysitting_(Step1_Step3-BIS)` — Celle 1, 2, 3 e 4  
**Collegamento Progettuale:** Step 1 (*Ispezione Dati Grezzi*) $\rightarrow$ Step 2 (*Parsing JSON `pycocotools`, `COCOSemanticDataset` e `DataLoader`*) $\rightarrow$ Step 6 (*Transfer Learning su Cityscapes*)  
**Tipo di Documento:** Scheda di Approfondimento Teorico-Metodologico (Q&A)

---

## 1. Quesiti Metodologici Formalizzati

Questo documento risponde in modo organico a cinque domande fondamentali emerse durante la progettazione delle **Celle 1–4** del nostro notebook e particolarmente rilevanti per la discussione orale con il professore:

1. **Allineamento con la Traccia (`H3.pdf` vs Ambiente Kaggle):** *Perché la specifica progettuale cita `COCO 2017 (Stuff)` e lo scheletro di lezione presuppone maschere `.png` pre-renderizzate, mentre il nostro modello è addestrato su `COCO 2017 (Thing)` con $C = 81$ classi estratte dai file `instances_*.json`?*
2. **Necessità della Rasterizzazione:** *Perché le annotazioni vettoriali nei file `.json` non possono essere utilizzate direttamente per calcolare la funzione di perdita e come avviene la conversione in maschera densa 2D tramite `pycocotools`?*
3. **Sovrapposizione tra Istanze (*Painter's Algorithm*):** *Perché in `COCOSemanticDataset._load_raw_pair` (Cella 3) ordiniamo le istanze di ciascuna immagine per area decrescente prima di disegnarle sulla maschera?*
4. **Rimappatura Contigua ($1 \dots 80$) e Ruolo di `IGNORE_INDEX = 255`:** *Perché non usiamo direttamente i `category_id` originali di COCO ($1 \dots 90$) e perché manteniamo `IGNORE_INDEX = 255` pur senza introdurre bordi di padding?*
5. **Integrità dei Canali RGB e dei Tipi Numerici:** *Perché imponiamo `.convert("RGB")` su ogni immagine caricata e convertiamo la maschera da `uint8` a `torch.int64` (`torch.long`)?*

---

## 2. Disallineamento Dataset (`COCO Thing` vs `COCO Stuff`) e Integrazione di `pycocotools`

### 2.1 Differenza Tassonomica tra Categorie *Thing* e *Stuff*

In Computer Vision (formalizzata da *Lin et al., ECCV 2014* e *Caesar et al., CVPR 2018*), le etichette semantiche di una scena si dividono in due famiglie concettuali:

| Proprietà | **COCO 2017 (*Thing*)** — *Adottato nel Notebook* | **COCO 2017 (*Stuff*)** — *Citato in `H3.pdf`* |
| :--- | :--- | :--- |
| **Natura semantica** | Oggetti contabili con forma definita e identità d'istanza individuale (`person`, `car`, `dog`, `chair`, `cup`). | Regioni amorfe, sfondi e materiali non contabili privi di istanze distinte (`sky`, `road`, `grass`, `wall`, `sea`). |
| **Numero di classi nel task** | **$C = 81$**: $0 = \text{background}$ + $80$ categorie *Thing* ($1 \dots 80$). | $91$ classi *Stuff* (o $172$ nella formulazione unificata *Stuff+Thing* + `unlabeled = 255`). |
| **Formato nativo su disco Kaggle** | Immagini `.jpg` + annotazioni vettoriali poligonali/RLE in `instances_train2017.json` ($448.02\text{ MB}$) e `instances_val2017.json` ($19.06\text{ MB}$). | Maschere semantiche `.png` pre-renderizzate (`stuffthingmaps_trainval2017`), **assenti** nel dataset montato su Kaggle. |
| **Copertura dei pixel dell'immagine** | Sparsa: gli oggetti coprono il $10\text{--}20\%$ dei pixel; l'$80\text{--}90\%$ residuo appartiene alla classe $0$ (`background`). | Densa: quasi ogni pixel dell'immagine possiede un'etichetta semantica specifica. |

### 2.2 Giustificazione Ingegneristica della Scelta (`Celle 1 e 2`)

L'ispezione empirica eseguita da `inspect_raw_coco_directory(DATA_ROOT)` nella **Cella 1** ha accertato la struttura fisica del dataset ufficiale montato su Kaggle (`/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017`):
* **Cartelle presenti:** `['annotations', 'test2017', 'train2017', 'val2017']`
* **Cartelle maschere semantiche pre-renderizzate:** `NESSUNA`
* **Split verificati:** $118\,287$ file `.jpg` in `train2017`, $5\,000$ file `.jpg` in `val2017`, $40\,670$ file `.jpg` in `test2017`.

Anziché introdurre dipendenze da mirror esterni non verificati, abbiamo formulato il problema di *Semantic Segmentation* multi-classe direttamente sul dataset ufficiale **COCO 2017 (*Thing*)** con **$C = 81$ classi**, integrando esplicitamente la libreria **`pycocotools`** (`from pycocotools.coco import COCO` nella Cella 1, indicizzazione in $23.52\text{ s}$ nella Cella 2) per decodificare poligoni e maschere compresse RLE (*Run-Length Encoding*).

---

## 3. Perché è Indispensabile Rasterizzare le Annotazioni JSON

Le informazioni contenute nei file `instances_*.json` **non possono** essere confrontate direttamente con le predizioni di `FCN-ResNet50` per tre ragioni matematiche e architetturali:

1. **Rappresentazione Vettoriale vs Tensore Denso Pixel-per-Pixel:**
   * Nel file JSON, ogni annotazione è memorizzata come lista di vertici poligonali $\{(x_1, y_1), (x_2, y_2), \dots\}$ oppure come stringa binaria compressa RLE.
   * La rete `FCN-ResNet50` produce in uscita un tensore denso di *logit* $\hat{Y} \in \mathbb{R}^{B \times 81 \times H \times W}$, assegnando a **ogni singola coordinata spaziale $(y, x)$** un vettore di $81$ punteggi di classe.
2. **Allineamento Spaziale nella Funzione di Perdita:**
   * Tutte le funzioni di perdita della segmentazione semantica (*Cross-Entropy*, *Focal*, *Dice*, *Jaccard*) operano **pixel per pixel**, confrontando i *logit* $\hat{Y}[b, :, y, x]$ con l'indice di classe reale $M[b, y, x] \in \{0, 1, \dots, 80\}$ alla medesima coordinata $(y, x)$.
3. **Sincronizzazione delle Trasformazioni Geometriche (`Opzione C`):**
   * Solo dopo aver materializzato la maschera bidimensionale $M \in \{0, \dots, 80\}^{H \times W}$ è possibile applicare in modo sincronizzato `TF.resize(..., InterpolationMode.NEAREST)`, `TF.crop` e `TF.hflip` (Cella 3), garantendo la corrispondenza sub-pixel tra immagine RGB e *ground truth* (verificata visivamente in `01_step2_dataloader_sanity_overlay.png`).

---

## 4. Ordinamento per Area Decrescente (*Painter's Algorithm* / *Z-Ordering*)

Una differenza cruciale tra `COCO Stuff` (mappe semantiche mutuamente esclusive) e `COCO Thing` (annotazioni d'istanza indipendenti) è che **in `COCO Thing` più istanze della stessa immagine possono sovrapporsi spazialmente**.

Nella funzione `COCOSemanticDataset._load_raw_pair` (Cella 3), prima di riempire la maschera `mask_np`, ordiniamo esplicitamente le annotazioni in **ordine decrescente di area**:

```python
anns_sorted = sorted(anns, key=lambda a: a["area"], reverse=True)
for ann in anns_sorted:
    c_idx = COCO_ID_TO_CONTIGUOUS[ann["category_id"]]
    binary_mask = self.coco.annToMask(ann)
    mask_np[binary_mask == 1] = c_idx
```

### Perché questo ordinamento è essenziale:
* **Comportamento senza ordinamento:** Se nel JSON un oggetto esteso in secondo piano (es. `dining table`, `couch` o `person`) fosse elencato dopo un oggetto piccolo in primo piano (es. `cup`, `tie` o `dog`), la scrittura sequenziale sovrascriverebbe e cancellerebbe completamente l'oggetto più piccolo.
* **Comportamento con *Painter's Algorithm* (`reverse=True`):** Disegnando prima le superfici più estese e successivamente le istanze di area minore, gli oggetti piccoli in primo piano restano nitidamente visibili nella maschera semantica.
* **Prova Visiva nel Nostro Notebook:** Nella **Riga 3** di `01_step2_dataloader_sanity_overlay.png` (Cella 4), l'istanza `dog` (giallo ocra) sovrapposta davanti a `person` (viola) viene rasterizzata per ultima e rimane perfettamente segmentata anziché essere inghiottita dalla maschera della persona.

---

## 5. Mappatura Contigua ($1 \dots 80$), `IGNORE_INDEX = 255` e Integrità dei Tipi

### 5.1 Perché Rimappiamo `category_id` $[1 \dots 90] \rightarrow [1 \dots 80]$ (`Cella 2`)

Nella tassonomia originale di MS COCO 2017, le $80$ categorie *Thing* **non** possiedono identificativi contigui: gli ID sono distribuiti nell'intervallo $[1, 90]$ a causa della rimozione storica di $10$ categorie intermedie (es. ID `12`, `26`, `29`, `30`, `45`, `66`, `68`, `69`, `71`, `83`).

* In PyTorch, un classificatore finale `Conv2d(..., out_channels=81)` produce canali indicizzati rigorosamente da $0$ a $80$.
* Se passassimo direttamente un'etichetta con valore `90` a `nn.CrossEntropyLoss()`, PyTorch solleverebbe immediatamente un errore `IndexError: Target 90 is out of bounds` (o un `device-side assert triggered` su CUDA).
* Nella **Cella 2**, `COCO_ID_TO_CONTIGUOUS` stabilisce quindi una corrispondenza biunivoca deterministica ordinata sull'intero vocabolario $[1 \dots 80]$, riservando $0$ al `background` (`CONTIGUOUS_TO_NAME[0] = "background"`).

### 5.2 Ruolo di `IGNORE_INDEX = 255` e Perché Non Introduciamo Padding

Nella **Cella 1** definiamo `IGNORE_INDEX = 255` (il valore sentinella standard in `torchvision` e `nn.CrossEntropyLoss(ignore_index=255)`):

1. **Separazione dal dominio valido $[0, 80]$:** Il valore $255$ rientra perfettamente nel tipo `uint8` ($[0, 255]$) della maschera rasterizzata senza collidere con alcuna delle $81$ classi valide.
2. **Assenza di bordi artificiali grazie all'Opzione C:** L'analisi integrale della **Cella 3** ha mostrato che $822$ immagini in `train2017` ($0.69\%$) e $42$ immagini in `val2017` ($0.84\%$) hanno lato corto $< 256\text{ px}$ (fino a $230 \times 246\text{ px}$). Adottando il **ridimensionamento proporzionale del lato corto a `256 px`** (**Opzione C**) anziché il padding con cornici nere, evitiamo di introdurre pixel fittizi `255` sui bordi di COCO.
3. **Compatibilità Architetturale per lo Step 6 (Cityscapes):** Mantenere `ignore_index=255` parametrizzato fin dallo Step 1 rende le funzioni di perdita e il valutatore (`SegmentationEvaluator`) immediatamente riutilizzabili nel successivo **Transfer Learning su Cityscapes**, dove i pixel non annotati (*void* / *ego-vehicle*) sono codificati proprio con `255`.

### 5.3 Conversione Forzata `.convert("RGB")` e Gestione dei `dtype` (`Cella 3`)

* **Prevenzione di Crash di Canale (`PIL mode == 'L'`):** L'ispezione su $500$ immagini di `val2017` nella **Cella 1** ha rilevato $2$ immagini in scala di grigi a singolo canale (`{'RGB': 498, 'L': 2}`, $0.40\%$). La chiamata `im.convert("RGB")` in `_load_raw_pair` replica il canale di luminanza su $3$ canali $(3, H, W)$, evitando errori di impilamento nel `DataLoader` e garantendo la compatibilità con `backbone.conv1` ($3$ canali in ingresso).
* **Pipeline dei Tipi Numerici:**
  $$\text{maschera JSON} \xrightarrow{\text{annToMask}} \texttt{np.uint8 } (H, W) \xrightarrow{\text{TF.resize (NEAREST)}} \texttt{PIL Image ('L')} \xrightarrow{\text{torch.from\_numpy}} \texttt{torch.int64 } (256, 256)$$
  L'interpolazione `InterpolationMode.NEAREST` durante il resize preserva l'integrità discreta degli indici di classe (evitando che l'interpolazione bilineare generi classi spurie intermedie lungo i contorni), mentre la conversione finale a `torch.int64` (`torch.long`) soddisfa il requisito nativo delle *loss* multi-classe di PyTorch.

---

## 6. Riferimenti Documentali e Bibliografici Validati per il Colloquio

| Fonte / Riferimento | Dettaglio Specifico | Argomento a Supporto |
| :--- | :--- | :--- |
| **1. Specifiche di Progetto (`H3.pdf`)** | *Semantic Segmentation*: addestramento *from scratch* di FCN su COCO 2017 e successivo *Transfer Learning* su Cityscapes. | Giustifica la formulazione multi-classe con `ignore_index=255` e la continuità della pipeline dati verso Cityscapes. |
| **2. Ispezione Empirica del Dataset (`Celle 1–3`)** | Assenza di cartelle `.png` pre-renderizzate; presenza di `instances_*.json` ($80$ classi *Thing*), $0.40\%$ immagini `'L'` e $0.69\%$ immagini con lato $< 256\text{ px}$. | Dimostra che la scelta di COCO 2017 (*Thing*, $C=81$) + `pycocotools` deriva da una verifica oggettiva dell'ambiente Kaggle. |
| **3. Lin et al. (ECCV 2014)** | *Microsoft COCO: Common Objects in Context* | Definisce la tassonomia delle $80$ categorie *Thing*, gli ID originali non contigui $[1, 90]$ e il formato poligonale/RLE delle annotazioni d'istanza. |
| **4. Caesar et al. (CVPR 2018)** | *COCO-Stuff: Thing and Stuff Classes in Context* | Formalizza la distinzione teorica tra classi *Thing* (oggetti contabili sovrapponibili) e classi *Stuff* (regioni amorfe mutuamente esclusive). |
| **5. Documentazione PyTorch** | `torch.nn.CrossEntropyLoss` e `torchvision.transforms.functional.resize` | Conferma il requisito di target `torch.int64` in $[0, C-1] \cup \{\text{ignore\_index}\}$ e la necessità di `InterpolationMode.NEAREST` per maschere categoriche. |

---

## 7. Sintesi Pronta per la Discussione Orale

* **Risposta sintetica al professore:**  
  *"Dall'ispezione preliminare del dataset COCO 2017 montato su Kaggle abbiamo verificato l'assenza di maschere semantiche `.png` pre-renderizzate e la presenza delle annotazioni vettoriali ufficiali di COCO 2017 (*Thing*) in `instances_*.json`. Abbiamo quindi formulato il task di Semantic Segmentation su $C = 81$ classi ($0 = \text{background}$ più le $80$ categorie *Thing* rimappate deterministicamente dagli ID sparsi $[1, 90]$ all'intervallo contiguo $[1, 80]$), integrando `pycocotools` per rasterizzare i poligoni e le RLE in maschere dense 2D. Poiché in COCO Thing più istanze possono sovrapporsi nella stessa scena, rasterizziamo le annotazioni in ordine decrescente di area (Painter's Algorithm) affinché gli oggetti più piccoli in primo piano non vengano coperti dalle superfici più estese, imponendo inoltre `.convert("RGB")` per le rare immagini in scala di grigi e `InterpolationMode.NEAREST` durante il ridimensionamento."*