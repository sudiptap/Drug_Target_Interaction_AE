import subprocess

dataset = ['nr_admat_dgc.txt', 'gpcr_admat_dgc.txt', 'ic_admat_dgc.txt', 'e_admat_dgc.txt']

## run for different noise for dim = 2000
#noise = [0.001, 0.005, 0.01, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4]
#
#for d in dataset:
#  for n in noise:
#    cmd = 'python dae2.py '+d+' '+str(n)
#    subprocess.check_output(cmd, shell=True)

# run for different dim for n = 0.1
dims = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,2400, 2500, 2600, 2700, 2800, 2900, 3000]
for d in dataset:
  for dim in dims:
    n = 0.1
    cmd = 'python dae2.py '+d+' '+str(n)+' '+str(dim)
    subprocess.check_output(cmd, shell=True)



