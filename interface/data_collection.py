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



def main(state, db, dataset, dataset_key, split, conf_option, LABEL_FIELD = 'label'):
    # write one example to main
    st.markdown("## Strategy: `%s`, label field: `%s`" % ('random', LABEL_FIELD))

    col1, col2 = st.beta_columns(2)
    with col2:
        # annotation guidelines
        datapoint_guideline = """
        Data point annotation guidelines here.
        """
        st.markdown(datapoint_guideline)

        with st.form("new_datapoint_form"):
            selected_label = st.selectbox('What is the label of the data point?',
                    dataset.features[LABEL_FIELD].names)
            new_point_submitted = st.form_submit_button("Save label")

        if new_point_submitted:
            label_dict = {'split' : split, 'idx' : state.point_idx, 'label' : selected_label, 'annotator' : 'jw'}
            db.insert(label_dict)

            # refresh label
            state['point_idx'] = None

    with col1:
        request_next_point = st.button("Sample new point")

        if request_next_point or state['point_idx'] is None:
            strategy = lambda : np.random.randint(0, len(dataset)-1)
            idx = strategy()
            state.point_idx = idx
        else:
            idx = state.point_idx

        example = dataset[idx]
        example = removeHyphen(example)

        nolabel_example = example.copy()
        del(nolabel_example[LABEL_FIELD])
        st.write('idx: ', idx)
        st.write(nolabel_example)

    st.markdown('## Write a template!')
    template_guideline = """
    Template annotation guidelines here.
    """
    st.markdown(template_guideline)

    with st.form("try_template_form"):
        answer_choices = st.text_input(
            "Answer Choices",
            help="A Jinja expression for computing answer choices. "
            "Separate choices with spaces and a triple bar ( ||| ).",
        )

        # Jinja
        jinja = st.text_area("Template", 
                height=40,
                help="Here's an example: To which category does this article belong? {{text}}"
        )


        st.markdown('### API hostname: `%s`' % '10.136.17.32:8000/')
        test_submit_button = st.form_submit_button('Test template')

    if test_submit_button:
        template = Template('test', jinja, "jw", answer_choices=answer_choices)
        applied_template = template.apply(example)[0]

        choices = answer_choices.split(' ||| ')

        r = requests.get('http://10.136.17.32:8000/', 
                params={'inputs' : applied_template, 'choices' : json.dumps(choices)})
        probs = json.loads(r.content)

        st.write(r.url)

        col1, col2 = st.beta_columns(2)
        with col1:
            st.write('Input:')
            show_text(applied_template)
            st.write(choices)

        with col2:
            st.write('Output:')
            st.write(sorted(zip(choices, probs), key=lambda x: -x[1]))

        #
        # only display saving window if tested
        #
        st.markdown('## Save template')
        with st.form("save_template_form"):

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
                else:
                    template = Template(new_template_name, jinja, "jw", answer_choices=answer_choices)
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
