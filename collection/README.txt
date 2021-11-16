this folder contains framework for collecting prompts from command line. 
since prompt running is very slow, computation is saved in the "state" folder
instead of relying on main memory

etl.py loads in the target dataset (currently, just sst2)

collect_prompts.py prompts the user to provide prompts for the loaded dataset
