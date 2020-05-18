import lps_maestro as maestro

config = 'user.pedrosergiot.attention_pca_filter_first'
task = 'user.pedrosergiot.task.attention_pca_filter_first'
data = 'user.pedrosergiot.Exames2_sem1segundo.mat'
image = 'pedrosergiot/attention_pca:attention_pca_image'

command = "python3 tunning_jobs_filter_first.py -d %DATA -c %IN -o %OUT"

maestro.task.create(task, data, config, command, command)