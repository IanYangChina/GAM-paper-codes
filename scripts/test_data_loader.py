import os
from manipulation_project.env.sg_data_loader import DataLoader


loader = DataLoader(
    path=os.path.dirname(__file__)+'/../SG_data', shape='S', num_hooks=3,
    mani_labels=False,
    start_dir_ind=4,
    end_dir_ind=4, end_file_ind=4
)

while not loader.finished:
    print(loader.data['scene_alteration_values'].min(), loader.data['scene_alteration_values'].max())
    loader.load_next_file()
