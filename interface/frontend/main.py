import json
import requests
import argparse
import textwrap
from multiprocessing import Manager, Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from datasets import get_dataset_infos
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import DjangoLexer

from tinydb import TinyDB, Query

from promptsource.session import _get_state
from promptsource.templates import DatasetTemplates, Template, TemplateCollection
from promptsource.utils import (
    get_dataset,
    get_dataset_confs,
    list_datasets,
    removeHyphen,
    renameDatasetColumn,
    render_features,
)


# add an argument for read-only
# At the moment, streamlit does not handle python script arguments gracefully.
# Thus, for read-only mode, you have to type one of the below two:
# streamlit run promptsource/app.py -- -r
# streamlit run promptsource/app.py -- --read-only
# Check https://github.com/streamlit/streamlit/issues/337 for more information.
parser = argparse.ArgumentParser(description="run app.py with args")
parser.add_argument("-r", "--read-only", action="store_true", help="whether to run it as read-only mode")

args = parser.parse_args()
if args.read_only:
    select_options = ["View data", "View prompts"]
    side_bar_title_prefix = "Promptsource (Read only)"
else:
    select_options = ["Data collection", "View data", "View prompts"]
    side_bar_title_prefix = "USC"

#
# Cache functions
#
get_dataset = st.cache(allow_output_mutation=True)(get_dataset)
get_dataset_confs = st.cache(get_dataset_confs)
list_datasets = st.cache(list_datasets)

def get_database():
    db = TinyDB('db.json') 
    return db
get_database = st.cache(allow_output_mutation=True)(get_database)

#
# Loads session state
#
state = _get_state()

#
# Initial page setup
#
st.set_page_config(page_title="Data collection", layout="wide")
st.sidebar.markdown(
    "<center>Quick and easy data collection by prompting language models</center>",
    unsafe_allow_html=True,
)
mode = st.sidebar.selectbox(
    label="Choose a mode",
    options=select_options,
    index=0,
    key="mode_select",
)
st.sidebar.title(f"{side_bar_title_prefix} 🌸 - {mode}")

#
# Adds pygments styles to the page.
#
st.markdown(
    "<style>" + HtmlFormatter(style="friendly").get_style_defs(".highlight") + "</style>", unsafe_allow_html=True
)


# Combining mode `Prompted dataset viewer` and `Sourcing` since the
# backbone of the interfaces is the same
assert mode in ["Data collection", "View data", "View prompts"], ValueError(
    f"`mode` ({mode}) should be in `[Data collection, View data]`"
)

#
# Loads dataset information
#

dataset_list = list_datasets()
ag_news_index = dataset_list.index("ag_news")

#
# Select a dataset - starts with ag_news
#
dataset_key = st.sidebar.selectbox(
    "Dataset",
    dataset_list,
    key="dataset_select",
    index=ag_news_index,
    help="Select the dataset to work on.",
)

#
# datapoint database
#
db = get_database()

#
# If a particular dataset is selected, loads dataset and template information
#
if dataset_key is not None:
    #
    # Check for subconfigurations (i.e. subsets)
    #
    configs = get_dataset_confs(dataset_key)
    conf_option = None
    if len(configs) > 0:
        conf_option = st.sidebar.selectbox("Subset", configs, index=0, format_func=lambda a: a.name)

    dataset = get_dataset(dataset_key, str(conf_option.name) if conf_option else None)
    splits = list(dataset.keys())
    index = 0
    if "train" in splits:
        index = splits.index("train")
    split = st.sidebar.selectbox("Split", splits, key="split_select", index=index)
    dataset = dataset[split]
    dataset = renameDatasetColumn(dataset)

    #
    # Loads template data
    #
    try:
        dataset_templates = DatasetTemplates(dataset_key, conf_option.name if conf_option else None)
    except FileNotFoundError:
        st.error(
            "Unable to find the prompt folder!\n\n"
            "We expect the folder to be in the working directory. "
            "You might need to restart the app in the root directory of the repo."
        )
        st.stop()

    #
    # Body of the app: display prompted examples in mode `Prompted dataset viewer`
    # or text boxes to create new prompts in mode `Sourcing`
    #

    if mode == "Data collection":
        from data_collection import main
        main(state, db, dataset, dataset_key, split, conf_option, dataset_templates)
    if mode == "View data":
        from view_data import main
        main(state, dataset, db)
    if mode == "View prompts":
        from view_prompts import main
        main(state, dataset, dataset_templates, db)

    #
    # sidebar
    #
    template_list = dataset_templates.all_template_names
    num_templates = len(template_list)
    st.sidebar.write(
        "No of prompts created for "
        + f"`{dataset_key + (('/' + conf_option.name) if conf_option else '')}`"
        + f": **{str(num_templates)}**"
    )

    num_datapoints = len(db)
    st.sidebar.write(
        "No of points collected for "
        + f"`{dataset_key + (('/' + conf_option.name) if conf_option else '')}`"
        + f": **{str(num_datapoints)}**"
    )

    st.sidebar.subheader("Dataset Schema")
    rendered_features = render_features(dataset.features)
    st.sidebar.write(rendered_features)


