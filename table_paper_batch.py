import sys
import os
import pandas as pd

sys.path.append("sourcecode/")
from core.nd_data import NDData

###############################################################################


def latex_fun(x, r, c, data):
    i = data.columns.get_loc(c)

    if(i != 0):
        x = "{:.3f}".format(float(x))
        if(x[:1] == "0"):
            x = x[1:]

        test_m = float("{:.3f}".format(float(data.iloc[r, 2])))
        test_sqrtm = float("{:.3f}".format(float(data.iloc[r, 4])))
        test_erm = float("{:.3f}".format(float(data.iloc[r, 6])))

        if(i == 2 and test_m <= test_sqrtm and test_m <= test_erm):
            x = "{{\\bf {}}}".format(x)
        elif(i == 4 and test_sqrtm <= test_m and test_sqrtm <= test_erm):
            x = "{{\\bf {}}}".format(x)
        elif(i == 6 and test_erm <= test_m and test_erm <= test_sqrtm):
            x = "{{\\bf {}}}".format(x)
    else:
        x = "\\textsc{{\\footnotesize {}}}".format(x)

    return x

###############################################################################


os.makedirs("tables/", exist_ok=True)
data_list = ["linear_batch.csv", "nn_batch.csv"]

for data in data_list:

    table_path = "paper_"+data.replace(".csv", "")+".tex"

    data = NDData(data)

    d_m = data.get(
        "data", "train_risk", "test_risk", epsilon="m", ratio_set_size=0.2)
    d_m.rename(columns={
        "train_risk": "train_risk_m",
        "test_risk": "test_risk_m"},
        inplace=True)
    d_sqrt = data.get(
        "data", "train_risk", "test_risk", "erm_train_risk", "erm_test_risk",
        epsilon="sqrtm", ratio_set_size=0.2)
    d_sqrt.rename(columns={
        "data": "data_2",
        "train_risk": "train_risk_sqrtm",
        "test_risk": "test_risk_sqrtm",
    }, inplace=True)

    d = pd.concat([d_m, d_sqrt], axis=1, join="inner")
    d = d.drop(['data_2'], axis=1)
    d.rename(columns={
        "data": ("", "Dataset"),
        "train_risk_m": (
            r"\cref{alg:batch} {\small ($\frac{1}{m}$)}",
            r"{\scriptsize $\Rfrak_{\S}(h)$}"),
        "test_risk_m": (
            r"\cref{alg:batch} {\small ($\frac{1}{m}$)}",
            r"{\scriptsize $\Rfrak_{\D}(h)$}"),
        "train_risk_sqrtm": (
            r"\cref{alg:batch} {\small ($\frac{1}{\sqrt{m}}$)}",
            r"{\scriptsize $\Rfrak_{\S}(h)$}"),
        "test_risk_sqrtm": (
            r"\cref{alg:batch} {\small ($\frac{1}{\sqrt{m}}$)}",
            r"{\scriptsize $\Rfrak_{\D}(h)$}"),
        "erm_train_risk": (
            r"ERM", r"{\scriptsize $\Rfrak_{\S}(h)$}"),
        "erm_test_risk": (
            r"ERM", r"{\scriptsize $\Rfrak_{\D}(h)$}")}, inplace=True)
    d.columns = pd.MultiIndex.from_tuples(d.columns)

    table_str = NDData.to_latex(
        d, latex_fun,
        col_format=(
            r"c@{\ }|@{\ }c@{\ }c@{\ }|"
            + r"@{\ }c@{\ }c@{\ }|@{\ }c@{\ }c@{\ }||"))
    table_str = table_str.replace(
        r"\multicolumn{2}{r}", r"\multicolumn{2}{c}")

    with open(f"tables/{table_path}", "w") as f:
        f.write(table_str)
    print(table_str)
