from graph import draw_combined_graphs
from analyse import calculate_smape_and_save, process_log_files, combine_graphs

process_log_files(folder_path="env")
process_log_files(folder_path="buffer")
process_log_files(folder_path="noise")
process_log_files(folder_path="policy")

pt=-400
pb=200
ss= 10

draw_combined_graphs(path_to_super="env\\super", path_to_anti="env\\anti",source="env",padding_top=pt,padding_bottom=pb,smoothing_step=ss)
draw_combined_graphs(path_to_super="buffer\\super", path_to_anti="buffer\\anti",source="buffer",padding_top=pt,padding_bottom=pb,smoothing_step=ss)
draw_combined_graphs(path_to_super="noise\\super", path_to_anti="noise\\anti",source="noise",padding_top=pt,padding_bottom=pb,smoothing_step=ss)
draw_combined_graphs(path_to_super="policy\\super", path_to_anti="policy\\anti",source="policy",padding_top=pt,padding_bottom=pb,smoothing_step=ss)

calculate_smape_and_save(path_to_super="env\\super", path_to_anti="env\\anti", source="env")
calculate_smape_and_save(path_to_super="buffer\\super", path_to_anti="buffer\\anti", source="buffer")
calculate_smape_and_save(path_to_super="noise\\super", path_to_anti="noise\\anti", source="noise")
calculate_smape_and_save(path_to_super="policy\\super", path_to_anti="policy\\anti", source="policy")

combine_graphs()



