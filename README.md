This repository contains the source code of the paper entitled:

> **Learning via Wasserstein-Based High Probability Generalization Bounds**<br/>
> Paul Viallard, Maxime Haddouche, Umut Şimşekli, Benjamin Guedj<br/>
> NeurIPS, 2023


### Running the experiments

**Generating the data:**
```bash
cd data
python ../run.py local generate_data.ini
cd ../
```

**Running the algorithms:**
```bash
python run.py local run.ini
```

**Generating the tables:**
```bash
python run.py local generate_tables.ini
```

### Conda environment

The sourcecode was tested with the packages provided in _env.yml_.
