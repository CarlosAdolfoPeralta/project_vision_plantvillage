🇪🇸 [Leer en español](README.es.md)

# Leaf Disease Classification: From 99% in the Lab to 9% in the Field

A computer vision project on PlantVillage: a custom CNN baseline, transfer learning with MobileNetV3-Small, interpretability with Grad-CAM, and a robustness test on field images (PlantDoc). All training was done **on CPU only**.

**Author:** Carlos Adolfo Peralta · [LinkedIn](https://www.linkedin.com/in/carlosperalta1993/) · [GitHub](https://github.com/CarlosAdolfoPeralta)

## Summary

- A transfer learning model (MobileNetV3-Small) reaches **accuracy 0.987 and macro F1 0.982** on the PlantVillage test set (39 classes) when trained on the full train set. With the same 25% of train used by the baseline, it gets 0.968 and 0.959, vs. **0.803 and 0.746** for the CNN trained from scratch.
- On field images (PlantDoc, 2,569 images, 28 mapped classes) the same model drops to **top-1 accuracy of 0.088 and macro F1 of 0.095**. More than half of the images are predicted as "background without leaves".
- **The model's confidence cannot flag those errors**: it is just as high on errors (0.91) as on correct predictions (0.89).
- Grad-CAM shows that, on lab images, the model looks at the leaf and not the background. The failure in the field seems to be due to a **domain shift** (untested hypothesis) rather than a problem of attention to the background.
- Conclusion: metrics on lab datasets do not guarantee performance in real-world conditions.

## Data

| | |
|---|---|
| Main dataset | PlantVillage (`DScomp380/plant_village` version on Hugging Face) |
| Images | 55,447, RGB, almost all 256x256 |
| Classes | 39 (38 crop/condition + `Background_without_leaves`) |
| Composition | 39,221 diseased, 15,083 healthy, 1,143 background |
| Imbalance | 36.2x between the largest class (5,507) and the smallest (152, `Potato___healthy`) |
| Split | stratified 70/15/15, seed 42: train 38,812, val 8,317, test 8,318 |
| External data | PlantDoc (`Project-AgML/plant_doc_classification`), 2,569 images cropped from search engines, mapped to the PlantVillage classes |

**Data leakage warning:** this version of the dataset has no leaf identifier, so the split could not be grouped by physical leaf and there may be nearly identical photos between train and test. A conservative estimate using perceptual hashing detected at least 1.3% near-duplicates between test and train (19 of 1,500 test images; this is a floor, not a measure of the real problem). The PlantVillage metrics should be read as probably optimistic.

## Methodology

1. **EDA** (`01_eda`): class distribution, healthy/diseased/background composition, image sizes and leakage estimate.
2. **Baseline** (`02_baseline_cnn`): a 4-block CNN (103K parameters) trained from scratch at 128x128 on 25% of train (9,703 images) with data augmentation. Compared with and without class weights.
3. **Transfer learning** (`03_transfer_learning`): MobileNetV3-Small pretrained on ImageNet and frozen. Each image goes through the network only once at 224x224, the feature vectors are saved, and only a dense head (159K parameters) is trained. This allows training in seconds without a GPU. It uses no data augmentation.
4. **Interpretability and robustness** (`04_gradcam_robustez`): Grad-CAM computed on the logit of the predicted class, an approximate measure of how much attention falls on the leaf, and evaluation on PlantDoc without retraining.

The main metric is **macro F1**, because of the class imbalance.

## Results

### Baseline vs. transfer learning (PlantVillage test)

| Model | Train | Class weights | Accuracy | Macro F1 | Training time |
|---|---|---|---|---|---|
| CNN from scratch | 25% | no | 0.803 | 0.746 | 18.3 min |
| CNN from scratch | 25% | yes (balanced) | 0.758 | 0.723 | 20.0 min |
| MobileNetV3-Small | 25% | no | 0.968 | 0.959 | < 1 min |
| MobileNetV3-Small | 25% | soft (exp. 0.5) | 0.968 | 0.960 | < 1 min |
| MobileNetV3-Small | 100% | no | 0.985 | 0.980 | < 1 min |
| **MobileNetV3-Small** | **100%** | **soft (exp. 0.5)** | **0.987** | **0.982** | **< 1 min** |

The rows with 25% of train use the same stratified subset, so they are comparable with each other. The training time of the transfer learning models covers only the head; feature extraction is done once and cached.

- Data matters more than any other change: going from 25% to 100% of train raises macro F1 by about 2 points.
- Full balanced class weights made the baseline worse (they improve recall on small classes at the cost of precision). Soft weights add 0.1 to 0.2 points with transfer learning, a difference that is probably within run-to-run noise (one run per configuration).
- The background class does not inflate the results: macro F1 without it is practically the same.
- The remaining errors are mostly between similar diseases within the same crop (for example, mites ↔ `Target_Spot` in tomato, `Cercospora` ↔ `Northern_Leaf_Blight` in corn).

### Robustness on field images (PlantDoc)

| Metric | PlantVillage (test) | PlantDoc |
|---|---|---|
| Top-1 accuracy | 0.987 | **0.088** |
| Top-3 accuracy | n/a | 0.229 |
| Macro F1 | 0.982 | 0.095 |

- 1,397 of 2,569 images (54%) are predicted as `Background_without_leaves`.
- Ignoring the background class when predicting only raises top-1 to 0.123; using a square crop instead of stretching the image brings it to 0.104. The problem is not explained by those two factors.
- **Confidence is useless for rejecting errors:** mean confidence is 0.894 on correct predictions and 0.908 on errors. Accepting only predictions with confidence ≥ 0.99 (53% of the images), the accuracy of the accepted ones is still 0.075.
- The best class is `Tomato___Late_blight` (0.459); no healthy class exceeds 0.06.

### Grad-CAM

- On the PlantVillage test set, the map falls on the leaf (sometimes on lesions, sometimes on nearby healthy tissue) and not on the background. With an approximate leaf mask based on color saturation, the **median attention ratio is 1.40** (247 images): the activation on the leaf is 1.4 times what the area it occupies would predict.
- On PlantDoc, when the model is right the map falls on the lesions; when it is wrong, it shows one or two scattered spots unrelated to the lesions.
- Grad-CAM was computed on the logit rather than on the softmax probability: with very confident predictions, the softmax gradient flattens out and produces empty maps.

## Limitations

- Split not grouped by leaf: the PlantVillage metrics are probably optimistic.
- A single run per configuration, with no repetitions across seeds: differences of a few tenths are not conclusive.
- CPU-only training: small baseline, with a subset and reduced resolution; the transfer learning model uses no augmentation or fine-tuning.
- The PlantDoc → PlantVillage class mapping is manual and may contain judgment errors (for example, which specific disease corresponds to a generic label such as "leaf spot"). PlantDoc comes from image search engines and may have label noise. In addition, some classes have very few images (for example, 2 for tomato mites), so their per-class accuracy is not informative.
- The leaf mask used to measure attention is a color-saturation heuristic, and the Grad-CAM maps have low resolution (7x7). The review of the grids was visual and on a small sample; the sample contained only 4 errors on PlantVillage, so correct and incorrect predictions could not be compared.
- Why the model fails on PlantDoc was not tested: the most likely explanation is a general domain shift (lighting, scale, backgrounds, quality), but it is a hypothesis.

## Future work

- Train with strong data augmentation (color, background, crops) and fine-tune the last layers.
- Incorporate part of the field images into training and evaluate on the rest (making sure crops of the same original image do not end up on both sides).
- Add a class or an out-of-distribution detection method, since softmax confidence is not useful for that.
- Repeat the experiments with several seeds to estimate variability.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   └── 04_gradcam_robustez.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py          # data loading, splits and array cache
│   └── features.py      # feature extraction with pretrained networks
├── reports/             # results of each experiment (.json)
├── models/              # trained models (not versioned)
└── data/                # data and cache (not versioned)
```

## How to reproduce

1. Create an environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Download the dataset from the command line, with resume if interrupted:
   ```bash
   hf download DScomp380/plant_village --repo-type dataset --local-dir data/plant_village
   ```
   If the `data/plant_village` folder does not exist, `src/data.py` downloads it automatically from Hugging Face.
3. Run the notebooks in order from the `notebooks/` folder. The first run of 02 and 03 takes longer because it generates caches in `data/cache/`; as a rough guide, each baseline run takes between 18 and 20 minutes on CPU. Notebook 04 uses the model saved by 03 with `USE_SUBSET=False` and `CW_POWER=0.5`, so that configuration must have been run beforehand.
4. For the robustness section, notebook 04 uses PlantDoc: it reads the `data/plantdoc` folder if it exists (it can be downloaded with `hf download Project-AgML/plant_doc_classification --repo-type dataset --local-dir data/plantdoc`) or, otherwise, downloads it from Hugging Face. It also accepts your own photos in `data/external/<ClassName>/`.

Developed with Python 3.13, TensorFlow 2.21 and `datasets` 5.1.0. Seeds are fixed (42), but results may vary slightly between runs and machines.

## References and licenses

- Mohanty, Hughes and Salathé (2016). *Using deep learning for image-based plant disease detection*. Frontiers in Plant Science.
- Singh et al. (2019). *PlantDoc: A Dataset for Visual Plant Disease Detection*. arXiv:1911.10317. License CC BY 4.0.
- The PlantVillage version used is listed on Hugging Face under CC0 1.0, according to its dataset card.
- The PlantDoc version used (Hugging Face) has 2,569 images and 28 classes; the original paper reports 2,598 images and 27 classes.
- The data is not included in the repository: it is downloaded from Hugging Face.
