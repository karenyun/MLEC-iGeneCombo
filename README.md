# MLEC-iGeneCombi

### Installtion  
Main packages should be installed, and `Python>=3.10.14`.
```txt
torch==2.1.2
torch-geometric==2.3.1
numpy==1.24.3
pandas==2.2.0
spicy==1.14.1
scikit-learn==1.2.2
networkx==3.2.1
```

### Train
`python train.py -c config_res.json -t "22RV1" --network false --cell true --lr 0.0025`

### Test
`python test.py -c config_res.json -t "22RV1" -m "test" --network false --cell true -r "saved_model_path"`