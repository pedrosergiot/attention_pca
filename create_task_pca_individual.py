import lps_maestro as maestro

config = 'user.pedrosergiot.attention_pca_individual'
task = 'user.pedrosergiot.task.attention_pca_individual'
data = 'user.pedrosergiot.Exames2_sem1segundo.mat'
image = 'pedrosergiot/attention_pca:attention_pca_image'

command = "python3 tunning_jobs_pca_individual.py -d %DATA -c %IN -o %OUT"

maestro.task.create(task, data, config, command, command)