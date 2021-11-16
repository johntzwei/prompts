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
    select_options = ["View data"]
    side_bar_title_prefix = "Promptsource (Read only)"
else:
    select_options = ["Data collection", "View data"]
    side_bar_title_prefix = "USC"

#
# Cache functions
#
get_dataset = st.cache(allow_output_mutation=True)(get_dataset)
get_dataset_confs = st.cache(get_dataset_confs)
list_datasets = st.cache(list_datasets)


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

WIDTH = 80


def show_jinja(t, width=WIDTH):
    wrap = textwrap.fill(t, width=width, replace_whitespace=False)
    out = highlight(wrap, DjangoLexer(), HtmlFormatter())
    st.write(out, unsafe_allow_html=True)


def show_text(t, width=WIDTH, with_markdown=False):
    wrap = [textwrap.fill(subt, width=width, replace_whitespace=False) for subt in t.split("\n")]
    wrap = "\n".join(wrap)
    if with_markdown:
        st.write(wrap, unsafe_allow_html=True)
    else:
        st.text(wrap)


# Combining mode `Prompted dataset viewer` and `Sourcing` since the
# backbone of the interfaces is the same
assert mode in ["Data collection", "View data"], ValueError(
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

    template_list = dataset_templates.all_template_names
    num_templates = len(template_list)
    st.sidebar.write(
        "No of prompts created for "
        + f"`{dataset_key + (('/' + conf_option.name) if conf_option else '')}`"
        + f": **{str(num_templates)}**"
    )

    st.sidebar.subheader("Dataset Schema")
    rendered_features = render_features(dataset.features)
    st.sidebar.write(rendered_features)


    #
    # Body of the app: display prompted examples in mode `Prompted dataset viewer`
    # or text boxes to create new prompts in mode `Sourcing`
    #
    LABEL_FIELD = 'label'
    LABEL_FILE = 'labels.jsonl'

    if mode == "Data collection":
        # write one example to main
        st.markdown("## Strategy: `%s`, label field: `%s`" % ('random', LABEL_FIELD))

        col1, col2 = st.beta_columns(2)
        with col1:
            request_next_point = st.button("Sample new point")

            strategy = lambda : np.random.randint(0, len(dataset)-1)
            idx = strategy()

            example = dataset[idx]
            example = removeHyphen(example)

            nolabel_example = example.copy()
            del(nolabel_example[LABEL_FIELD])
            st.write('idx: ', idx)
            st.write(nolabel_example)


        with col2:
            # annotation guidelines
            datapoint_guideline = """
            Data point annotation guidelines here.
            """
            st.markdown(datapoint_guideline)

            with st.form("new_datapoint_form"):
                selected_label = st.selectbox('What is the label of the data point?',
                        dataset.features[LABEL_FIELD].names)
                new_point_submitted = st.form_submit_button("Save label only")

            if new_point_submitted:
                # are file writes atomic?
                label_dict = {'idx' : idx, LABEL_FIELD : selected_label, 'annotator' : 'jw'}
                labels = open(LABEL_FILE, 'at')
                labels.write(json.dumps(label_dict) + '\n')
                labels.close()


        st.markdown('## Write a template!')
        template_guideline = """
        Template annotation guidelines here.
        """
        st.markdown(template_guideline)

        with st.form("try_template_form"):
            answer_choices = st.text_input(
                "Answer Choices",
                help="A Jinja expression for computing answer choices. "
                "Separate choices with a triple bar (|||).",
            )

            # Jinja
            jinja = st.text_area("Template", 
                    height=40,
                    help="Here's an example for ag_news: "
                    "What label best describes this news article?\n{{text}} ||| [predict] "
            )


            st.markdown('### API hostname: `%s`' % '10.136.17.32:8000/')
            if st.form_submit_button('Test template'):
                template = Template('test', jinja, "jw")
                applied_template = template.apply(example)[0]

                choices = answer_choices.split(' ||| ')

                st.write('Input:')
                st.write(applied_template)
                st.write(choices)
                st.write('Output:')

                import requests
                r = requests.get('http://10.136.17.32:8000/', 
                        params={'inputs' : applied_template, 'choices' : json.dumps(choices)})
                st.write(r.url)

                probs = json.loads(r.content)
                st.write(sorted(zip(choices, probs), key=lambda x: -x[1]))

                # save state
                state.jinja = jinja
                state.answer_choices = answer_choices


        st.markdown('## Save template')
        with st.form("save_template_form"):

            st.markdown('jinja: `%s`' % state.jinja)
            st.write('answer_choices: `%s`' % state.answer_choices)

            template = Template("no_name", "", "")

            new_template_name = st.text_input(
                "Template name",
                help="Choose a descriptive name for the template below."
            )

            # Metadata
            original_task = st.checkbox(
                "Original Task?",
                help="Prompt asks model to perform the original task designed for this dataset.",
            )
            choices_in_prompt = st.checkbox(
                "Choices in Template?",
                help="Prompt explicitly lists choices in the template for the output.",
            )

            new_template_submitted = st.form_submit_button("Save template")
            if new_template_submitted:
                if new_template_name in dataset_templates.all_template_names:
                    st.error(
                        f"A prompt with the name {new_template_name} already exists "
                        f"for dataset {state.templates_key}!"
                    )
                elif new_template_name == "":
                    st.error("Need to provide a prompt name!")
                elif state.jinja == None:
                    st.error("Test your prompt first!")
                else:
                    template = Template(new_template_name, jinja, "jw")
                    dataset_templates.add_template(template)


        #
        # Display dataset information
        #
        st.header("Dataset: " + dataset_key + " " + (("/ " + conf_option.name) if conf_option else ""))

        st.markdown(
            "*Homepage*: "
            + dataset.info.homepage
            + "\n\n*Dataset*: https://github.com/huggingface/datasets/blob/master/datasets/%s/%s.py"
            % (dataset_key, dataset_key)
        )

        md = """
        %s
        """ % (
            dataset.info.description.replace("\\", "") if dataset_key else ""
        )
        st.markdown(md)

#
# Must sync state at end
#
state.sync()
