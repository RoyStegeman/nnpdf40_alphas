executable   = /data/theorie/rstegeman/miniconda3/envs/nnpdf/bin/python
arguments    = /data/theorie/rstegeman/github/nnpdf40_alphas/closuretest/compute_csv_tcm_closure.py $(item)

output       = /data/theorie/rstegeman/github/nnpdf40_alphas/closuretest/logs/$(item).out
error        = /data/theorie/rstegeman/github/nnpdf40_alphas/closuretest/logs/$(item).err
log          = /data/theorie/rstegeman/github/nnpdf40_alphas/closuretest/logs/$(item).log

getenv = true

request_cpus   = 8
request_memory = 28G
request_disk   = 8G

+JobCategory            = "medium"
+UseOS                  = "el9"
accounting_group        = smefit

max_idle = 20
queue from seq 1 100 |
