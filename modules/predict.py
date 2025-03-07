import glob
import json
import os

import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '/Users/ekaterinakomarova/airflow_hw')

def predict():
    model_files = glob.glob(os.path.join(path, "data", "models", "*"))
    last_model_file = max(model_files, key=os.path.getmtime)
    with open(last_model_file, "rb") as f:
        model = dill.load(f)
        print(f)

        pred_df = pd.DataFrame(columns=['id', 'price_category'])

    for datapath in glob.glob(f"{path}/data/test/*.json"):
        with open(datapath) as fin:
            form = json.load(fin)
            df_f = pd.DataFrame.from_dict([form])
            y = model.predict(df_f)
            x = {
                'id': df_f.id,
                'price_category': y
            }
            result = pd.DataFrame([x])
            pred_df = pd.concat([pred_df, result], ignore_index=True)

    print(pred_df)
    pred_df.to_csv(os.path.join(path, "data", "predictions", "predictions.csv"), index=False)

pass


if __name__ == '__main__':
    predict()
