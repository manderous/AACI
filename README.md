Improving long-tail relation extraction via adaptive adjustment and causal inference
==========

This project includes the code implemented with PyTorch and the paper 'Improving Long-tail Relation Extraction via Adaptive Adjustment and Causal Inference' published in the Neurocomputing 2023 journal.

https://doi.org/10.1016/j.neucom.2023.126563

## Training
```
python train_tjy_la.py --id [saved_models/your_model_name] --seed 1234 --effect_type None --lr 1 --num_epoch 40 --pooling max --mlp_layers 2 --pooling_l2 0
```

## Evaluation

To run evaluation on the test set for CGCN, run:
```
python eval.py [saved_models/your_model_name] --dataset test
```
