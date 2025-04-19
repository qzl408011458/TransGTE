This is the repository for the paper "TransGTE: A Transformer-based Model with Geographical Trajectory Embedding for the Individual Trip Destination Prediction".

## Data Preparation
The original data used in the paper are open-source with links attached at the REMADE.md of data folder. We provide processed datasets used in the experiments at the following link: 
https://drive.google.com/file/d/1GllK2bhk4VuwyVl3yBCz89n1JFl0Cff0/view?usp=drive_link

## Model Training
The TransGTE model used in the experiments can be trained by following command: 
```python
python train.py --pr <trajectory completion ratio> --data <dataset name> --g <grid granularity> --batch_size <batch size>
```
The argparse argument pr can be selected from [0.1, 0.3, 0.5, 0.7, 0.9]. The pairs of data and g can be selected from
(porto, 50), (chengdu, 70), (shenzhen, 70), and (sanfran, 60).
The recommended hyper-parameters in Table 3 of the paper are all initially set in the train.py. 

# Model Evaluation
The model parameters will be saved at a new folder in the **modelsave** after training. The trained model can be evaluation by following command:
```python
python eval.py --model_path <a path of model checkpoint>
```
Note that the checkpoint path should be enclosed with '' or "".

