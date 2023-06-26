import pdb
import sys
import numpy as np
import pandas as pd

sys.path.append(".")

configs = [{
        "type": "classification",
        "code": "breast_cancer",
        "name": "Breast cancer",
        "filepath": "data/breast_cancer.csv",
        "n_attributes": 9,
        "attributes": [
            "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
            "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei",
            "Bland Chromatin", "Normal Nucleoli", "Mitoses"],
        "n_classes": 2,
        "classes": [(2, "Benign"), (4, "Malignant")]
    },
    {
        "type": "classification",
        "code": "car",
        "name": "Car evaluation",
        "filepath": "data/car.csv",
        "n_attributes": 6,
        "attributes": ["buying", "maint", "doors", "persons", "lug_boot", "safety"],
        "n_classes": 4,
        "classes": [(0, "Unacceptable"), (1, "Acceptable"), (2, "Good"), (3, "Very good")]
    },
    {
        "type": "classification",
        "code": "banknote",
        "name": "Banknote authentication",
        "filepath": "data/banknote.csv",
        "n_attributes": 4,
        "attributes": ["variance", "skewness", "curtosis", "entropy"],
        "n_classes": 2,
        "classes": [(0, "Authentic"), (1, "Forged")]
    },
    {
        "type": "classification",
        "code": "balance",
        "name": "Balance scale",
        "filepath": "data/balance.csv",
        "n_attributes": 4,
        "attributes": ["left weight", "left distance", "right weight", "right distance"],
        "n_classes": 3,
        "classes": [(0, "Left"), (1, "Balanced"), (2, "Right")]
    },
    {
        "type": "classification",
        "code": "acute-1",
        "name": "Acute inflammations 1",
        "filepath": "data/acute-1.csv",
        "n_attributes": 6,
        "attributes": ["temperature", "nausea", "lumbar pain", "urine pushing", "micturition", "burning urethra"],
        "n_classes": 2,
        "classes": [(0, "No inflammation"), (1, "Inflammation")]
    },
    {
        "type": "classification",
        "code": "acute-2",
        "name": "Acute inflammations 2",
        "filepath": "data/acute-2.csv",
        "n_attributes": 6,
        "attributes": ["temperature", "nausea", "lumbar pain", "urine pushing", "micturition", "burning urethra"],
        "n_classes": 2,
        "classes": [(0, "No nephritis"), (1, "Nephritis")]
    },
    {
        "type": "classification",
        "code": "transfusion",
        "name": "Blood transfusion",
        "filepath": "data/transfusion.csv",
        "n_attributes": 4,
        "attributes": ["recency", "frequency", "monetary", "time"],
        "n_classes": 2,
        "classes": [(0, "Not donor"), (1, "Donor")]
    },
    {
        "type": "classification",
        "code": "climate",
        "name": "Climate model crashes",
        "filepath": "data/climate.csv",
        "n_attributes": 18,
        "attributes": ["vconst_corr", "vconst_2", "vconst_3", "vconst_4", "vconst_5", "vconst_7", "ah_corr", "ah_bolus",
                       "slm_corr", "efficiency_factor", "tidal_mix_max", "vertical_decay_scale", "convect_corr",
                       "bckgrnd_vdc1", "bckgrnd_vdc_ban", "bckgrnd_vdc_eq", "bckgrnd_vdc_psim", "Prandtl"],
        "n_classes": 2,
        "classes": [(0, "Failure"), (1, "Success")]
    },
    {
        "type": "classification",
        "code": "sonar",
        "name": "Connectionist bench sonar",
        "filepath": "data/sonar.csv",
        "n_attributes": 60,
        "attributes": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
                       "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29",
                       "x30", "x31", "x32", "x33", "x34", "x35", "x36", "x37", "x38", "x39", "x40", "x41", "x42", "x43",
                       "x44", "x45", "x46", "x47", "x48", "x49", "x50", "x51", "x52", "x53", "x54", "x55", "x56", "x57",
                       "x58", "x59", "x60"],
        "n_classes": 2,
        "classes": [(0, "Rock"), (1, "Mine")]
    },
    {
        "type": "classification",
        "code": "optical",
        "name": "Optical recognition",
        "filepath": "data/optical.csv",
        "n_attributes": 64,
        "attributes": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
                       "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29",
                       "x30", "x31", "x32", "x33", "x34", "x35", "x36", "x37", "x38", "x39", "x40", "x41", "x42", "x43",
                       "x44", "x45", "x46", "x47", "x48", "x49", "x50", "x51", "x52", "x53", "x54", "x55", "x56", "x57",
                       "x58", "x59", "x60", "x61", "x62", "x63", "x64"],
        "n_classes": 10,
        "classes": [(0, "Number 0"), (1, "Number 1"), (2, "Number 2"), (3, "Number 3"), (4, "Number 4"), (5, "Number 5"),
                    (6, "Number 6"), (7, "Number 7"), (8, "Number 8"), (9, "Number 9")]
    },
    {
        "type": "classification",
        "code": "drybean",
        "name": "Drybeans",
        "filepath": "data/drybean.csv",
        "n_attributes": 16,
        "attributes": ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation", "Eccentricity",
                       "ConvexArea", "EquivDiameter", "Extent", "Solidity", "roundness", "Compactness", "ShapeFactor1",
                       "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"],
        "n_classes": 7,
        "classes": [(0, "Seker"), (1, "Barbunya"), (2, "Bombay"), (3, "Cali"), (4, "Dermosan"), (5, "Horoz"), (6, "Sira")]
    },
    {
        "type": "classification",
        "code": "avila",
        "name": "Avila bible",
        "filepath": "data/avila.csv",
        "n_attributes": 10,
        "attributes": ["intercolumnar distance", "upper margin", "lower margin", "exploitation", "row number",
                       "modular ratio", "interlinear spacing", "weight", "peak number",
                       "modular ratio/ interlinear spacing"],
        "n_classes": 12,
        "classes": [(0, "A"), (1, "B"), (2, "C"), (3, "D"), (4, "E"), (5, "F"), (6, "G"), (7, "H"), (8, "I"), (9, "W"),
                    (10, "X"), (11, "Y")]
    },
    {
        "type": "classification",
        "code": "wine-red",
        "name": "Wine quality red",
        "filepath": "data/wine-red.csv",
        "n_attributes": 11,
        "attributes": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                       "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
        "n_classes": 6,
        "classes": [(3, "Score 3"), (4, "Score 4"), (5, "Score 5"), (6, "Score 6"), (7, "Score 7"), (8, "Score 8")]
    },
    {
        "type": "classification",
        "code": "wine-white",
        "name": "Wine quality white",
        "filepath": "data/wine-white.csv",
        "n_attributes": 11,
        "attributes": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                       "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
        "n_classes": 7,
        "classes": [(3, "Score 3"), (4, "Score 4"), (5, "Score 5"), (6, "Score 6"), (7, "Score 7"), (8, "Score 8"),
                    (9, "Score 9")]
    },
    {
        "type": "regression",
        "code": "qsar",
        "name": "Qsar fish toxicity",
        "filepath": "data/qsar.csv",
        "n_attributes": 6,
        "attributes": ["CIC0", "SM1_Dz", "GATS1i", "NdsCH", "NdssC", "MLOGP", "LC50"]
    }]

def get_config(dataset_code):
    for config in configs:
        if config["code"] == dataset_code:
            return config

    raise Exception(f"Invalid dataset code {dataset_code}.")

def load_dataset(config, classification=True):
    df = pd.read_csv(config["filepath"])

    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values

    if classification:
        y = y.astype(np.int64)

    return X, y