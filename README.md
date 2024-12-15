# UC Berkeley MEng Capstone Project - Time Series
Advisor: Yunkai Zhang, yunkai_zhang@berkeley.edu.
This repository applies time series models to the [M5 dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy).

## Run Instructions
0. Install the required packages using Python 3.9.6.
```
pip install -r requirements.txt
```
1. Download data files from [Google Drive](https://drive.google.com/drive/folders/1-A45kVC1mssG7bJeUTLJISOcM4QEATaL?usp=drive_link).
Put these files in the`./dataset/m5` folder.
2. Log in to Weight and Biases. Follow the prompt.
```
wandb login
```
You will need to create your own W&B account. Once you do, you should update entity and project [here](https://github.com/zhykoties/Time-Series-Capstone/blob/9106dd57da80fcbf72d68a67ed778618fde77116/run.py#L146).
3. Run the script.
```
bash scripts/forecast/m5/Linear.sh
```

# M5 Data Versions
- v0: Predict the sales for each product category in each store.
Meta variables: state_id, store_id, product_category_id.
Given variables: time_from_start, snap_accepted, is_sporting_event, is_cultural_event, is_national_event, is_religious_event.

## Models
- **Linear:** We process all the meta variables, the given variables, the target variables (only in the context range),
and the time covariates, and concatenate them together. We feed the concatenated vector into a linear layer to predict 
the target values in the forecast range.
- **Enc_Only_Transformer:** The variables at each time step is treated as one token. The variables
are the meta variables, the given variables, the target variables, and the time covariates.
The encoder only takes in values in the context range. We take the output embeddings from the encoder,
flatten them, and feed them into a linear layer to predict the target values in the forecast range.
- **DeepAR (TODO):** Predict one step ahead each time, and feed back into the LSTM to predict the next step.
- **Enc_Dec_Transformer (TODO):** We treat the variables at each time step as one token,
similarly as the Enc_Only_Transformer. The encoder takes in values in the context range, and the decoder
takes in the values in the forecast range, except that for the target variables (which we don't have access to 
in the forecast range), for which we pad with zeros. Instead of flattening, we apply a linear layer 
independently to each output token in the decoder to get the prediction for the target variable at
the corresponding time step.

## Acknowledgements
- Time Series Library: https://github.com/thuml/Time-Series-Library
