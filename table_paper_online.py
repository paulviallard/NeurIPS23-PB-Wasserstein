import sys
import os
import pandas as pd

sys.path.append("sourcecode/")
from core.nd_data import NDData

###############################################################################


def latex_fun(x, r, c, data):
    i = data.columns.get_loc(c)

    x = "{:.3f}".format(float(x))
    if(x[:1] == "0"):
        x = x[1:]

    test_alg = float("{:.3f}".format(float(data.iloc[r, 1])))
    test_ogd = float("{:.3f}".format(float(data.iloc[r, 3])))

    if(i == 1 and test_alg <= test_ogd):
        x = "{{\\bf {}}}".format(x)
    elif(i == 3 and test_ogd <= test_alg):
        x = "{{\\bf {}}}".format(x)

    if(i == 1 and abs(test_alg-test_ogd) <= 0.05):
        x = "\\underline{{{}}}".format(x)
    if(i == 3 and abs(test_alg-test_ogd) <= 0.05):
        x = "\\underline{{{}}}".format(x)

    return x

###############################################################################


os.makedirs("tables/", exist_ok=True)
data_list = ["linear_online.csv", "nn_online.csv"]

for data in data_list:
    table_path = "paper_"+data.replace(".csv", "")+".tex"
    data = NDData(data)

    d = data.get("train_risk", "test_risk", "ogd_train_risk", "ogd_test_risk")

    d.rename(columns={
        "train_risk": (
            r"\cref{alg:online}", r"{\scriptsize $\Cfrak_{\S}$}"),
        "test_risk": (
            r"\cref{alg:online}", r"{\scriptsize $\Cfrak_{\D}$}"),
        "ogd_train_risk": (
            r"OGD", r"{\scriptsize $\Cfrak_{\S}$}"),
        "ogd_test_risk": (
            r"OGD", r"{\scriptsize $\Cfrak_{\D}$}")},
        inplace=True)
    d.columns = pd.MultiIndex.from_tuples(d.columns)

    table_str = NDData.to_latex(
        d, latex_fun, col_format=r"||@{\ }c@{\ }c@{\ }|@{\ }c@{\ }c")
    table_str = table_str.replace(
        r"\multicolumn{2}{r}", r"\multicolumn{2}{c}")

    with open(f"tables/{table_path}", "w") as f:
        f.write(table_str)
    print(table_str)
