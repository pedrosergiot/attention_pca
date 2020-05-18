import lps_maestro as maestro

config = 'user.pedrosergiot.pca_without_reconstruction'
task = 'user.pedrosergiot.task.pca_without_reconstruction'
data = 'user.pedrosergiot.Exames2_sem1segundo.mat'
image = 'pedrosergiot/attention_pca:attention_pca_image'

command = "python3 tunning_jobs_pca_without_reconstruction.py -d %DATA -c %IN -o %OUT"

maestro.task.create(task, data, config, command, command)