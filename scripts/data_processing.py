import os
import json
import numpy as np
from drl_implementation.agent.utils import plot as plot
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
cwd = os.path.dirname(os.path.realpath(__file__))

"""generate mean and deviation data"""
# num_books = str(2)
# case_dir = os.path.join(cwd, '..', 'results_hm_C_30', f'{num_books}hooks', 'cs_coef_0.0_-1.0')
# file_prefix = f'run-{num_books}hooks_cs_coef_0.0_-1.0_seed'
# file_suffix = '_data-tag-Test_'
# data_types = [
#     'num_is_exceeded_workspace',
#     # 'num_is_object_dropped',
#     'avg_num_is_grasped',
#     'return',
#     'avg_scene_alteration',
#     'num_is_separated'
# ]
# seeds = [11, 22, 33]
# for data_type in data_types:
#     data = []
#     for seed in seeds:
#         with open(os.path.join(case_dir, file_prefix + str(seed) + file_suffix + data_type + '.json'), 'rb') as f:
#             d = json.load(f)
#         data.append(np.array(d)[:, -1])
#     plot.get_mean_and_deviation(np.array(data),
#                                 save_data=True,
#                                 file_name=os.path.join(case_dir, data_type + '.json'))
# exit()

"""generate legend"""
# plt.rcParams.update({'font.size': 27})
# plot.smoothed_plot_mean_deviation(
#     file=os.path.join(cwd, '..', 'result_figs', 'legend.pdf'), data_dict_list=[None, None, None],
#     legend=['Ours', 'Moosmann et al.'], legend_ncol=2, legend_frame=False,
#     legend_bbox_to_anchor=(-0.1, 2.25), linewidth=5,
#     legend_loc='upper left', legend_only=True, legend_file=os.path.join(cwd, '..', 'result_figs', 'e2e-legend.pdf'),
#     x_axis_off=True, y_axis_off=True
# )
# exit()

"""generate plots"""
plt.rcParams.update({'font.size': 30})
num_hooks = 2
baselines = [
    'results_C_30',
    'results_hm_C_30'
]
cases = ['cs_coef_0.0_-1.0']
data_to_show = [
    'num_is_separated',
    'avg_num_is_grasped',
    'num_is_exceeded_workspace',
    'avg_scene_alteration',
    'return']
# straight_up_baseline = [[0.42488], [0.0011], None, [0.0241], None]  # 2C
# straight_up_baseline = [[0.63796], [0.0009], None, [0.0223], None]  # 3C
# straight_up_baseline = [[0.50942], [0.0016], None, [0.0256], None]  # 4C
# straight_up_baseline = [[0.29258], [0.0004], None, [0.0313], None]  # 3C+
# straight_up_baseline = [[0.2227], [0.0004], None, [0.0275], None]  # 3S
xoffs = [False, False, False, False, False]
titles = ['2 C Hooks', None, None, None, None]
y_labels = ['Success Rate', 'Num. Grasped', 'Exceed Workspace Rate', 'NG-Obj. Movement', 'Return']
y_lims = [(-0.1, 0.5), (-0.2, 4), (-0.1, 1.1), (-0.03, 0.3), (-12, 8)]
for i in range(len(data_to_show)):
    data_type = data_to_show[i]
    y_label = y_labels[i]
    ylim = y_lims[i]
    stat_dicts = []
    for baseline in baselines:
        case_dir = os.path.join(cwd, '..', baseline, str(num_hooks)+'hooks')
        for case in cases:
            with open(os.path.join(case_dir, case, data_type+'.json'), 'rb') as f:
                d = json.load(f)
            if i <= 2:
                d['mean'] = (np.array(d['mean']) / 30).tolist()
                d['lower'] = (np.array(d['lower']) / 30).tolist()
                d['upper'] = (np.array(d['upper']) / 30).tolist()
            stat_dicts.append(d)
    plot.smoothed_plot_mean_deviation(
        file=os.path.join(cwd, '..', 'result_figs', str(num_hooks)+'Chooks_e2e', data_type+'.pdf'),
        data_dict_list=stat_dicts,
        horizontal_lines=None, linestyle='--', linewidth=5,
        legend=None, legend_ncol=2, legend_frame=False,
        legend_bbox_to_anchor=(-0.1, 1.25),
        legend_loc='upper left',
        x_label='Timesteps (x 1e4)', x_axis_off=xoffs[i],
        y_label=None, ylim=ylim, y_axis_off=False,
        title=y_label
    )

"""generate histograms"""
# plt.rcParams.update({'font.size': 65})
# def histo(file, data, tasks, x_label, title=None,
#           y_axis_off=False, xlim=(None, None), xticks=None,
#           legend=False, legend_only=False, legend_file=None, legend_ncol=4):
#     colors = ['#262626', '#5C5C5C', '#989898',
#               '#06592A', '#22BB3B', '#40AD5A', '#6CBA7D',
#               '#FC4E2A', '#FD8D3C', '#feb24c',
#               '#0D4A70', '#3C93C2', '#9EC9E2',
#               '#C40F5B', '#E32977', '#E95694', '#ED85B0']
#     # colors.reverse()
#     fig = plt.figure(figsize=(9, 45), dpi=500)
#     baselines = list(data.keys())
#     # baselines.reverse()
#     width = 0.9 / len(baselines)
#     ticks = np.arange(len(tasks))
#
#     for index, baseline in enumerate(baselines):
#         bars = plt.barh(ticks + index * width, data[baseline], width,
#                         label=baseline, color=colors[index])
#
#     plt.xlabel(x_label)
#     if title is not None:
#         plt.title(title)
#     if y_axis_off:
#         plt.ylabel(None)
#         plt.yticks([])
#     else:
#         plt.yticks(ticks + 0.7, labels=tasks, rotation=-90)
#     if xlim[0] is not None:
#         plt.xlim(xlim)
#         plt.xticks(xticks)
#
#     if legend:
#         legend_plot = plt.legend(loc='best', bbox_to_anchor=(-2, 2), ncol=legend_ncol,
#                                  handlelength=1, handleheight=1)
#         if legend_only:
#             assert legend_file is not None, 'specify legend save path'
#             fig = legend_plot.figure
#             fig.canvas.draw()
#             bbox = legend_plot.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#             fig.savefig(legend_file, dpi=500, bbox_inches=bbox)
#             plt.close()
#             return
#
#     plt.savefig(file, bbox_inches='tight', dpi=500)
#     plt.close()
#
#
# # Columns -> tasks: ['2 C hooks', '3 C hooks', '4 C hooks', '3 C+ hooks', '3 S hooks']
# success_rate_data = {
#     'SLM': [42.5, 63.8, 50.9, 29.3, 22.3],
#     'HM': [68.5, 65.6, 56.4, 32.6, 26.1],
#     'CM': [65.5, 73.7, 61.1, 32.5, 34.4],
#     'CM-I:0.8': [91.4, 93.4, 90.2, 87.3, 88.2],
#     'CM-I:0.9': [92.1, 93.8, 90.7, 87.8, 88.6],
#     'CM-I:0.95': [92.5, 94.2, 91.1, 88.1, 88.8],
#     'CM-I:0.98': [93.1, 94.5, 91.6, 88.6, 89.1],
#     'CM-II:0.3' : [69.1, 77.6, 64.2, 34.7, 35.7],
#     'CM-II:0.2' : [69.2, 77.8, 64.2, 35.0, 35.8],
#     'CM-II:0.1' : [69.5, 78.0, 64.5, 34.7, 35.9],
#     'CM-III:0.3': [75.3, 80.0, 68.5, 51.5, 55.7],
#     'CM-III:0.2': [86.9, 90.6, 85.7, 74.0, 77.3],
#     'CM-III:0.1': [92.9, 97.7, 94.4, 80.2, 86.0],
#     'CM-IV:0.8-0.2': [97.0, 97.0, 95.2, 91.6, 92.7],
#     'CM-IV:0.8-0.1': [97.3, 98.9, 97.1, 90.7, 92.3],
#     'CM-IV:0.9-0.2': [97.0, 97.2, 95.4, 91.7, 92.8],
#     'CM-IV:0.9-0.1': [97.4, 98.9, 97.2, 90.7, 93.6]
# }
# obj_drop_data = {
#     'SLM': [0.0, 0.0, 0.0, 0.0, 0.0],
#     'HM': [13.6, 18.3, 19.6, 8.7, 8.2],
#     'CM': [7.8, 6.3, 6.0, 7.8, 5.5],
#     'CM-I:0.8': [2.6, 1.6, 1.8, 2.3, 1.8],
#     'CM-I:0.9': [2.5, 1.5, 1.8, 2.3, 1.8],
#     'CM-I:0.95': [2.5, 1.5, 1.7, 2.3, 1.9],
#     'CM-I:0.98': [2.4, 1.4, 1.7, 2.2, 1.8],
#     'CM-II:0.3': [3.8, 2.6, 2.8, 3.9, 2.8],
#     'CM-II:0.2': [3.7, 2.6, 2.7, 3.9, 2.8],
#     'CM-II:0.1': [3.6, 2.5, 2.6, 3.8, 2.8],
#     'CM-III:0.3': [9.6, 6.2, 6.7, 8.4, 7.4],
#     'CM-III:0.2': [7.5, 4.3, 5.9, 6.9, 9.2],
#     'CM-III:0.1': [5.0, 1.2, 3.2, 7.2, 7.8],
#     'CM-IV:0.8-0.2': [1.7, 0.9, 1.3, 1.6, 2.5],
#     'CM-IV:0.8-0.1': [1.5, 0.2, 0.9, 0.9, 2.6],
#     'CM-IV:0.9-0.2': [1.7, 0.9, 1.2, 1.6, 2.6],
#     'CM-IV:0.9-0.1': [1.5, 0.2, 0.9, 0.9, 1.9]
# }
# ng_obj_move_data = {
#     'SLM': [0.0241, 0.0223, 0.0256, 0.0313, 0.0275],
#     'HM': [0.6168, 0.3920, 0.4273, 0.7318, 0.4673],
#     'CM': [0.3092, 0.1905, 0.2556, 0.3433, 0.3110],
#     'CM-I:0.8': [0.2780, 0.1570, 0.2167, 0.2813, 0.2623],
#     'CM-I:0.9': [0.2748, 0.1556, 0.2143, 0.2800, 0.2608],
#     'CM-I:0.95': [0.2710, 0.1540, 0.2115, 0.2797, 0.2595],
#     'CM-I:0.98': [0.2638, 0.1519, 0.2073, 0.2779, 0.2582],
#     'CM-II:0.3': [0.3135, 0.1894, 0.2565, 0.3455, 0.3133],
#     'CM-II:0.2': [0.3135, 0.1892, 0.2565, 0.3456, 0.3132],
#     'CM-II:0.1': [0.3134, 0.1888, 0.2563, 0.3455, 0.3129],
#     'CM-III:0.3': [0.1936, 0.1573, 0.2198, 0.2623, 0.2430],
#     'CM-III:0.2': [0.1353, 0.1065, 0.1477, 0.1974, 0.1591],
#     'CM-III:0.1': [0.0907, 0.0580, 0.0973, 0.1782, 0.0815],
#     'CM-IV:0.8-0.2': [0.1249, 0.1000, 0.1385, 0.1855, 0.1509],
#     'CM-IV:0.8-0.1': [0.0876, 0.0571, 0.0958, 0.1724, 0.0828],
#     'CM-IV:0.9-0.2': [0.1247, 0.0997, 0.1383, 0.1857, 0.1502],
#     'CM-IV:0.9-0.1': [0.0876, 0.0570, 0.0959, 0.1724, 0.0792]
# }
# DG_data = {
#     'SLM': [0.0, 0.0, 0.0, 0.0, 0.0],
#     'HM': [0.0, 0.0, 0.0, 0.0, 0.0],
#     'CM': [0.0, 0.0, 0.0, 0.0, 0.0],
#     'CM-I:0.8': [36.8, 26.8, 42.7, 70.1, 68.2],
#     'CM-I:0.9': [38.7, 27.8, 45.2, 71.4, 69.9],
#     'CM-I:0.95': [40.7, 28.9, 47.9, 72.9, 71.8],
#     'CM-I:0.98': [43.7, 30.8, 51.6, 75.3, 74.4],
#     'CM-II:0.3': [7.2, 6.5, 6.5, 9.3, 6.0],
#     'CM-II:0.2': [7.6, 6.9, 6.9, 9.8, 6.5],
#     'CM-II:0.1': [8.3, 7.3, 7.6, 10.9, 7.4],
#     'CM-III:0.3': [50.9, 18.6, 27.7, 62.8, 65.8],
#     'CM-III:0.2': [75.9, 52.2, 71.6, 91.6, 96.5],
#     'CM-III:0.1': [93.3, 90.7, 94.3, 99.5, 99.6],
#     'CM-IV:0.8-0.2': [79.3, 56.4, 75.7, 93.7, 97.2],
#     'CM-IV:0.8-0.1': [93.8, 90.8, 94.7, 99.5, 99.6],
#     'CM-IV:0.9-0.2': [79.3, 56.6, 76.0, 93.8, 97.5],
#     'CM-IV:0.9-0.1': [93.8, 90.9, 94.5, 99.5, 99.6]
# }
#
# xlims = [(0.0, 105), (0.0, 20), (0.0, 0.75), (0.0, 105)]
# x_labels = ['Success rate (%)', 'Object dropped (%)', 'NGOM (meters)', 'Discarded grasp (%)']
# file_names = ['success_rate', 'obj_drop', 'ngom', 'discarded_grasp']
# y_axis_offs = [False, True, True, True]
# tasks = ['2 C hooks', '3 C hooks', '4 C hooks', '3 C+ hooks', '3 S hooks']
# datas = [success_rate_data, obj_drop_data, ng_obj_move_data, DG_data]

# for n in range(4):
#     histo(file=os.path.join(cwd, '..', 'result_figs', 'histograms', file_names[n]+'.pdf'),
#           legend=False, legend_only=True, legend_ncol=1, xlim=xlims[n],
#           legend_file=os.path.join(cwd, '..', 'result_figs', 'histograms', 'legend.pdf'),
#           y_axis_off=y_axis_offs[n],
#           data=datas[n], tasks=tasks,
#           x_label=x_labels[n])

# n = 0
# histo(file=os.path.join(cwd, '..', 'result_figs', 'histograms', file_names[n]+'.pdf'),
#       legend=True, legend_only=True, legend_ncol=1, xlim=xlims[n],
#       legend_file=os.path.join(cwd, '..', 'result_figs', 'histograms', 'legend.pdf'),
#       y_axis_off=y_axis_offs[n],
#       data=datas[n], tasks=tasks,
#       x_label=x_labels[n])