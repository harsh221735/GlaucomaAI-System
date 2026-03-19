import json

DB_PATH = "data/patient_records.json"


def save_visit(patient_id, cdr):

    with open(DB_PATH, "r") as f:
        data = json.load(f)

    if patient_id not in data:
        data[patient_id] = []

    data[patient_id].append({"cdr": cdr})

    with open(DB_PATH, "w") as f:
        json.dump(data, f)