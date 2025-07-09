# GeneCom

### Train
python train.py -c config_res.json -t "22RV1" --network false --cell true 

### Test
python test.py -c config_res.json -t "22RV1" -m "test" --network false --cell true -r "saved_model_path"