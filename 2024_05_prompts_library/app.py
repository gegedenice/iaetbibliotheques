import json
import streamlit as st
from streamlit_modal import Modal
import streamlit.components.v1 as components
from clients import OllamaClient, NvidiaClient, GroqClient

st.set_page_config(
    page_title="Prompts Library",
    layout="wide",
)

# Cache the header of the app to prevent re-rendering on each load
@st.cache_resource
def display_app_header():
    """Display the header of the Streamlit app."""
    st.title("Prompts Library")
    st.subheader("ChatBot with prompt templates")

# Display the header of the app
display_app_header()

# Some style
st.markdown(
    '<style>div[key^="edit-modal"] {top: 25px;}</style>', unsafe_allow_html=True
)


# UI sidebar ##########################################
def ui_llm(provider):
    if api_token := st.sidebar.text_input("Enter your API Key", type="password", key=f"API_{provider}"):
        provider_models = llm_providers[st.session_state["llm_provider"]](
            api_key=api_token
        ).list_models_names()
        if provider_models:
            llm = st.sidebar.radio(
                "Select your model", provider_models, key="llm"
            )
        else:
            st.sidebar.error("Ollama is not running, or there is a problem with the selected LLM provider")
    else:
        st.sidebar.warning("You must enter your API key")

st.sidebar.subheader("Models")

# LLM
llm_providers = {
    "Cloud Groq": GroqClient,
    "Cloud Nvidia": NvidiaClient,
    "Local Ollama": OllamaClient,
}
if llm_provider := st.sidebar.radio(
    "Choose your LLM Provider", llm_providers.keys(), key="llm_provider"
):
    ui_llm(st.session_state["llm_provider"])

# LLM parameters
st.sidebar.subheader("Parameters")
max_tokens = st.sidebar.number_input("Token numbers", value=1024, key="max_tokens")
temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="temperature"
)
top_p = st.sidebar.slider(
    "Top P", min_value=0.0, max_value=1.0, value=0.7, step=0.1, key="top_p"
)

# helpers functions ########################################

def edit_form(form_name, title=None, source=None, system=None, user=None):
    """
    Creates a form for editing a prompt template.

    Args:
        form_name: The name of the form.
        title: The title of the prompt template (optional).
        source: The source of the prompt template (optional).
        system: The system example instruction (optional).
        user: The user example instruction (optional).

    Returns:
        None
    """
    with st.form(form_name, clear_on_submit=False, border=True):
        new_title = st.text_input("Name", value=title)
        new_source = st.text_input("Source", value=source)
        new_system = st.text_area("System example instruction", value=system)
        new_user = st.text_area("User example instruction", value=user)
        if st.form_submit_button("Submit"):
            rec = {
                "title": new_title,
                "source": new_source,
                "messages": [
                    {"role": "system", "content": new_system},
                    {"role": "user", "content": new_user},
                ],
            }
            if title is not None:
                delete_prompt(title)
            add_prompt(rec)


def read_prompts_file():
    """
    Loads the prompts from the "prompts.json" file.

    Returns:
        A list of prompt templates.
    """
    prompts_file = open("prompts.json", encoding="utf-8")
    return json.load(prompts_file)


def add_prompt(rec):
    """
    Adds a new prompt to the "prompts.json" file.

    Args:
        rec: The new prompt to add.

    Returns:
        None
    """
    with open("prompts.json", "r", encoding="utf-8") as fp:
        listObj = json.load(fp)
    listObj.append(rec)
    print(listObj)
    with open("prompts.json", "w") as outfile:
        outfile.write(json.dumps(listObj, indent=4, sort_keys=True))
    st.rerun()


def edit_prompt(title):
    """
    Edits a prompt template.

    Args:
        title: The title of the prompt to edit.

    Returns:
        A dictionary containing the edited prompt information.
    """
    with open("prompts.json", "r", encoding="utf-8") as fp:
        listObj = json.load(fp)
    rec = [i for i in listObj if i["title"].strip() == title.strip()]
    rec_messages = rec[0]["messages"]
    return edit_form(
        "prompt_edit",
        title=title,
        source=[x["source"] for x in rec][0],
        system=[x["content"] for x in rec_messages if x["role"] == "system"][0],
        user=[x["content"] for x in rec_messages if x["role"] == "user"][0],
    )


def delete_prompt(title):
    """
    Removes a prompt template from the "prompts.json" file.

    Args:
        title: The title of the prompt to delete.
    """
    with open("prompts.json", "r", encoding="utf-8") as fp:
        listObj = json.load(fp)
    recs = [i for i in listObj if not (i["title"].strip() == title.strip())]
    with open("prompts.json", "w") as outfile:
        outfile.write(json.dumps(recs, indent=4, sort_keys=True))


def get_llm_response(system, prompt):
    """
    Generates a response from the selected LLM.

    Args:
        system: The system input from the user.
        prompt: The user prompt.

    Returns:
        The response from the LLM.
    """
    options = dict(
        max_tokens=st.session_state["max_tokens"],
        top_p=st.session_state["top_p"],
        temperature=st.session_state["temperature"],
    )
    return llm_providers[st.session_state["llm_provider"]](
        api_key=st.session_state[f"API_{st.session_state['llm_provider']}"],
        model=st.session_state["llm"],
    ).api_chat_completion(system, prompt, **options)


def generate(system, prompt):
    st.session_state.messages.append({"role": "system", "content": system})
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response = get_llm_response(
            system, prompt
        )
        st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def clear():
    for key in st.session_state.keys():
        del st.session_state[key]

# UI main #####################################################

tab1, tab2 = st.tabs(["Prompts Library", "Chatbot"])
with tab1:
    new_modal = Modal(
        "Add prompt",
        key="edit-modal",
    )
    if new_prompt_modal := st.button("➕ Add a prompt template"):
        new_modal.open()
    if new_modal.is_open():
        with new_modal.container():
            edit_form("prompt_add")
    prompts = read_prompts_file()
    grids = range(1, len(prompts) + 1)
    cols = st.columns([1, 1])
    wcol = 2
    for f, b in zip(prompts, grids):
        col = cols[b % wcol]
        with col:
            with st.expander(f["title"].upper()):
                if st.button(f"✔️ Select prompt {f['title'].upper()} and go to Chatbot tab", type="secondary"):
                        # can do better here
                        st.session_state["init_messages"] = f["messages"]
                        st.session_state.init_system = f["messages"][0]["content"]
                        st.session_state.init_user = f["messages"][1]["content"]
                edit_modal = Modal(
                    f"Edit prompt {f['title'].upper()}",
                    key=f"edit-modal_{f['title']}",
                )
                if edit_prompt_modal := st.button(
                    f"✏️ Edit {f['title'].upper()}", type="secondary"
                ):
                    edit_modal.open()
                if edit_modal.is_open():
                    with edit_modal.container():
                        edit_prompt(f["title"])
                st.write(f"Source : {f['source']}")
                st.markdown(f"- System : {f['messages'][0]['content']}")
                st.markdown(f"- User: {f['messages'][1]['content']}")
                st.divider()
                if st.button(f"❌ Delete prompt {f['title'].upper()}", type="primary"):
                    delete_prompt(f["title"])
                    st.rerun()
with tab2:
    # Clear chat history
    if st.button("Clear Chatbot history", type="secondary"):
        clear()
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # React to user input
    if "init_messages" in st.session_state:
        system = st.text_area(":blue[System]", key="init_system")
        prompt = st.text_area(":blue[User]", key="init_user")
    else:
        system = st.text_area(":blue[System]")
        prompt = st.text_area(":blue[User]")
    if st.button("Generate", type="primary"):
        generate(system, prompt)
