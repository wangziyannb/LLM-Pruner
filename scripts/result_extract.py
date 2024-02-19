import os
import json
# Extracting and converting specified fields to percentages


if __name__ == "__main__":
    files=os.listdir(os.getcwd())
    for f in files:
        if '.Json' in f.title():
            print(f.title())

            with open(f) as r:
                data=json.load(r)
                extracted_data_latest_final_percentage = {
                    "BoolQ_acc": data["results"]["boolq"]["acc"] * 100,
                    "PIQA_acc": data["results"]["piqa"]["acc"] * 100,
                    "Hellaswag_acc_norm": data["results"]["hellaswag"]["acc_norm"] * 100,
                    "winogrande_acc": data["results"]["winogrande"]["acc"] * 100,
                    "arc_easy_acc": data["results"]["arc_easy"]["acc"] * 100,
                    "arc_challenge_acc_norm": data["results"]["arc_challenge"]["acc_norm"] * 100,
                    "openbookqa_acc_norm": data["results"]["openbookqa"]["acc_norm"] * 100
                }
                
                print(extracted_data_latest_final_percentage)
                print()