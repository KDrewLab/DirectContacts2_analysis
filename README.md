# DirectContacts2_analysis
Proteins carry out cellular functions by self-assembling into functional complexes, a process that depends on direct physical interactions between components. Predicting whether proteins are directly interacting is an outstanding challenge. DirectContacts2 addresses this challenge by integrating diverse large-scale protein interaction datasets, including AP/MS (BioPlex1–3, Boldt et al., Hein et al.), biochemical fractionation (Wan et al.), proximity labeling (Gupta et al., Youn et al.), and RNA pulldown (Treiber et al.), to predict whether ~26 million human protein pairs interact directly or indirectly.

## Funding

    NIH R00, NSF/BBSRC

## Citation

    Erin R. Claussen, Miles D Woodcock-Girard, Samantha N Fischer, Kevin Drew


## Code examples for DirectContacts2 training, testing, and generating predictions

All code is available as Jupyter notebooks. There are notebooks which outline:
  1. Generating the benchmark
  2. Training the model
  3. Testing the model (with advice to use [ProteinComplexMaps](https://github.com/KDrewLab/protein_complex_maps.git) to generate Precision-Recall curves to evaluate performance.
  4. Making predictions for ~26 million protein pairs in the full feature matrix

All files for training, testing, and evaluating the model can be found on [HuggingFace](https://huggingface.co/datasets/sfisch/DirectContacts2/tree/main). Our model is built using [AutoGluon]() and to utilize our trained model **note that version 0.4.0 is needed**. If you decide to train a model from scratch, you may use newer versions of AutoGluon.

  ```bash
  $ pip install autogluon==0.4.0
  ```

### Accessing feature matrix files

All feature matrices can be pulled from our [dataset](https://huggingface.co/datasets/sfisch/DirectContacts2) on HuggingFace
and examples of using the *datasets* module can be seen in **DirectContacts2_testing.ipynb** and **DirectContacst2_training.ipynb**

In brief:
  ```python
  from autogluon.tabular import TabularPredictor
  from datasets import load_dataset

  dataset = load_dataset('sfisch/DirectContacts2')
  train = dataset["train"].to_pandas()
  test = dataset["test"].to_pandas()
  ```

The full feature matrix can be pulled using the *huggingface_hub* module as seen in **generating_predictions_w_DirectContacts2.ipynb** 

To pull from HuggingFace and use as a pandas dataframe:
  ```python
  from huggingface_hub import hf_hub_download
  import pandas as pd

  full_file = hf_hub_download(repo_id="sfisch/DirectContacts2", filename='full/humap3_full_feature_matrix_20220625.csv.gz', repo_type='dataset')
  full_featmat = pd.read_csv(full_file, compression="gzip")
  ```
### Accessing the DirectContacts2 model
The [DirectContacts2 model](https://huggingface.co/sfisch/DirectContacts2_AutoGluon) can also be downloaded from HuggingFace using the *huggingface_hub* module as seen in **generating_predictions_w_DirectContacts2.ipynb** and **DirectContacts2_testing.ipynb**. **Reminder:** That version 0.4.0 of AutoGluon is required for making predictions with our model.

  ```python
  from autogluon.tabular import TabularPredictor
  from huggingface_hub import snapshot_download

  model_dir = snapshot_download(repo_id="sfisch/hu.MAP3.0_AutoGluon")
  predictor = TabularPredictor.load(f"{model_dir}/huMAP3_20230503_complexportal_subset10kNEG_notScaled_accuracy")
  ```
### Accessing other testing/training data
All test/train complexes and interactions can be found on [HuggingFace](https://huggingface.co/datasets/sfisch/DirectContacts2/tree/main/reference_interactions)

These will be necessary for running performance analysis, such as Precision-Recall with [ProteinComplexMaps](https://github.com/KDrewLab/protein_complex_maps.git) 

In brief, after you have made your predictions on the test set (see **DirectContacts2_testing.ipynb**), you can use the files for the [test negatives](https://huggingface.co/datasets/sfisch/DirectContacts2/blob/main/reference_interactions/test_INdirect_interactions_pdbsize5_20240326.txt) and the [test positives](https://huggingface.co/datasets/sfisch/DirectContacts2/blob/main/reference_interactions/test_direct_interactions_pdbsize5_20240326.txt) for PR analysis with [prcurve.py](https://github.com/KDrewLab/protein_complex_maps/blob/master/protein_complex_maps/evaluation/plots/prcurve.py)

Example commandline:
  ```bash
(python2) user@computer$ python protein_complex_maps/protein_complex_maps/evaluation/plots/prcurve.py \
--results_wprob ./DirectContacts2_autogluon_test.pairsWprob \
--labels DirectContacts2 \
--input_positives ./test_direct_interactions_pdbsize5_20240326.txt \
--input_negatives ./test_INdirect_interactions_pdbsize5_20240326.txt \
--output_file ./DirectContacts2_test_eval_PR_curve_02JUL2025.pdf \
--complete_benchmark --add_tiny_noise
  ```
