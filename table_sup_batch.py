import sys
import os
import pandas as pd

sys.path.append("sourcecode/")
from core.nd_data import NDData

###############################################################################


def latex_fun(x, r, c, data):
    i = data.columns.get_loc(c)

    if(i == 0):
        x = "\\textsc{{\\footnotesize {}}}".format(x)
    else:
        x = "{:.3f}".format(float(x))
    return x

###############################################################################


data_list = ["linear_batch.csv", "nn_batch.csv"]

for ratio_set_size in [0.0, 0.4, 0.6, 0.8, 1.0]:

    for data in data_list:

        ratio_set_size_ = f"{ratio_set_size}".replace("0.", "")
        ratio_set_size_ = f"{ratio_set_size_}".replace("1.0", "10")

        table_path = ("sup_"+data.replace(".csv", "")
                      + f"_{ratio_set_size_}"+".tex")

        data = NDData(data)

        d_m = data.get(
            "data", "train_risk", "test_risk",
            epsilon="m", ratio_set_size=ratio_set_size)
        d_m.rename(columns={
            "train_risk": "train_risk_m",
            "test_risk": "test_risk_m"},
            inplace=True)
        d_sqrt = data.get(
            "data", "train_risk", "test_risk",
            epsilon="sqrtm", ratio_set_size=ratio_set_size)
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
                r"{\scriptsize $\Rfrak_{\D}(h)$}")}, inplace=True)
        d.columns = pd.MultiIndex.from_tuples(d.columns)

        table_str = NDData.to_latex(
            d, latex_fun,
            col_format=r"c|cc|cc")
        table_str = table_str.replace(
            r"\multicolumn{2}{r}", r"\multicolumn{2}{c}")

        with open(f"tables/{table_path}", "w") as f:
            f.write(table_str)
        print(table_str)

###############################################################################

data_list = ["linear_batch.csv", "nn_batch.csv"]

for data in data_list:

    ratio_set_size_ = f"{ratio_set_size}".replace("0.", "")
    ratio_set_size_ = f"{ratio_set_size_}".replace("1.0", "10")

    table_path = ("sup_"+data.replace(".csv", "")
                  + f"_ERM"+".tex")

    data = NDData(data)

    d = data.get(
        "data", "erm_train_risk", "erm_test_risk",
        epsilon="sqrtm", ratio_set_size=0.2)
    d.rename(columns={
        "data": "Dataset",
        "erm_train_risk": r"{\scriptsize $\Rfrak_{\S}(h)$}",
        "erm_test_risk": r"{\scriptsize $\Rfrak_{\D}(h)$}",
    }, inplace=True)

    table_str = NDData.to_latex(
        d, latex_fun,
        col_format=r"c|cc")

    with open(f"tables/{table_path}", "w") as f:
        f.write(table_str)
    print(table_str)

###############################################################################
###############################################################################
###############################################################################
###############################################################################

data_list = ["linear_batch.csv", "nn_batch.csv"]

for data in data_list:

    table_path = ("sup_"+data.replace(".csv", "")
                  + f"_L2"+".tex")

    data = NDData(data)

    d_m = data.get(
        "data", "reg_L2_train_risk", "reg_L2_test_risk",
        epsilon="m", ratio_set_size=0.2)
    d_m.rename(columns={
        "reg_L2_train_risk": "reg_L2_train_risk_m",
        "reg_L2_test_risk": "reg_L2_test_risk_m",
    }, inplace=True)

    d_sqrt = data.get(
        "data", "reg_L2_train_risk", "reg_L2_test_risk",
        epsilon="sqrtm", ratio_set_size=0.2)
    d_sqrt.rename(columns={
        "data": "data_2",
        "reg_L2_train_risk": "reg_L2_train_risk_sqrtm",
        "reg_L2_test_risk": "reg_L2_test_risk_sqrtm",
    }, inplace=True)

    d = pd.concat([d_m, d_sqrt], axis=1, join="inner")
    d = d.drop(['data_2'], axis=1)

    d.rename(columns={
        "data": (" ", "Dataset"),
        "reg_L2_train_risk_m": (
            r"L2 Reg. {\small ($\frac{1}{m}$)}",
            r"{\scriptsize $\Rfrak_{\S}(h)$}"),
        "reg_L2_test_risk_m": (
            r"L2 Reg. {\small ($\frac{1}{m}$)}",
            r"{\scriptsize $\Rfrak_{\D}(h)$}"),
        "reg_L2_train_risk_sqrtm": (
            r"L2 Reg. {\small ($\frac{1}{\sqrt{m}}$)}",
            r"{\scriptsize $\Rfrak_{\S}(h)$}"),
        "reg_L2_test_risk_sqrtm": (
            r"L2 Reg. {\small ($\frac{1}{\sqrt{m}}$)}",
            r"{\scriptsize $\Rfrak_{\D}(h)$}")}, inplace=True)

    d.columns = pd.MultiIndex.from_tuples(d.columns)

    table_str = NDData.to_latex(
        d, latex_fun,
        col_format=r"c|cc|cc")
    table_str = table_str.replace(
        r"\multicolumn{2}{r}", r"\multicolumn{2}{c}")

    with open(f"tables/{table_path}", "w") as f:
        f.write(table_str)
    print(table_str)
