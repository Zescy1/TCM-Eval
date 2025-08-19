from mmengine.config import read_base
with read_base():
    from opencompass.configs.datasets.TCMBench.tcm_demo import \
        tcmbench_datasets as tcmbench_datasets
    from opencompass.configs.models.TCMmodel.spark_x1 import \
        models as spark   

datasets = tcmbench_datasets
models=spark
work_dir='/home/opencompass/outputs/' # define which path you like

