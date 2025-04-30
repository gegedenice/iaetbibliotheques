import dash
from dash import dcc, html, Input, Output, State, ctx, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import umap
import hdbscan
import sklearn.feature_extraction.text as text
from dash.exceptions import PreventUpdate
import json
from dotenv import load_dotenv
import helpers
from omeka_s_api_client import OmekaSClient, OmekaSClientError
from lancedb_client import LanceDBManager
import torch
import torch.nn.functional as F

# Load .env for credentials
load_dotenv()
_DEFAULT_PARSE_METADATA = (
    'dcterms:identifier','dcterms:type','dcterms:title', 'dcterms:description',
    'dcterms:creator','dcterms:publisher','dcterms:date','dcterms:spatial',
    'dcterms:format','dcterms:provenance','dcterms:subject','dcterms:medium',
    'bibo:annotates','bibo:content', 'bibo:locator', 'bibo:owner'
)

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True
server = app.server
manager = LanceDBManager()


french_stopwords = text.ENGLISH_STOP_WORDS.union([
    "alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir", "bon",
    "car", "ce", "cela", "ces", "ceux", "chaque", "ci", "comme", "comment", "dans",
    "des", "du", "dedans", "dehors", "depuis", "devrait", "doit", "donc", "dos",
    "début", "elle", "elles", "en", "encore", "essai", "est", "et", "eu", "fait",
    "faites", "fois", "font", "hors", "ici", "il", "ils", "je", "juste", "la", "le",
    "les", "leur", "là", "ma", "maintenant", "mais", "mes", "mine", "moins", "mon",
    "mot", "même", "ni", "nommés", "notre", "nous", "nouveaux", "ou", "où", "par",
    "parce", "parole", "pas", "personnes", "peut", "peu", "pièce", "plupart", "pour",
    "pourquoi", "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "sa",
    "sans", "ses", "seulement", "si", "sien", "son", "sont", "sous", "soyez", "sujet",
    "sur", "ta", "tandis", "tellement", "tels", "tes", "ton", "tous", "tout", "trop",
    "très", "tu", "valeur", "voie", "voient", "vont", "votre", "vous", "vu", "ça",
    "étaient", "état", "étions", "été", "être"
])

# -------------------- Layout --------------------
app.layout = html.Div([
    # Header
    dbc.NavbarSimple(
        children=[],
        brand="Omeka S Computer Vision Assistant",
        brand_href="/",
        color="light",
        dark=False,
        className="mb-4 shadow-sm border-bottom"
    ),

    # Main Container
    dbc.Container(fluid=True, children=[
        dbc.Row([
            # Left column - Controls
            dbc.Col(width=6, children=[
                dbc.Card([
                    dbc.CardHeader(html.H4("Data Loading and ploting", className="text-center")),
                    dbc.CardBody([

                        # Tabs
                        dcc.Tabs(id="data-tabs", value="api", children=[
                            dcc.Tab(label="Harvest data from Omeka S", value="omeka"),
                            dcc.Tab(label="Visualize existing collections", value="lance")
                        ]),

                        html.Div(id="data-tab-content"),

                        html.Br(),
                    ])
                ], className="mb-4 shadow-sm")
            ]),
            # Right column - Explanations
            dbc.Col(width=6, children=[
                dbc.Card([
                    dbc.CardHeader(
                        html.H4(
                            dbc.Button("Explanations", color="primary", id="explanation-toggle", n_clicks=0),
                            className="text-center"
                        )
                    ),
                    dbc.Collapse(
                        dbc.CardBody([
                            html.P("This application allows you to explore Omeka S collections through interactive visualization."),
    html.P("You can load data in two ways:"),
    html.P("1. From Omeka S: Connect to your Omeka S instance and select a collection to visualize."),
    html.P("2. From LanceDB: Load previously processed collections from the local database."),
    html.P("The visualization uses UMAP projection and topic clustering to create an interactive map of your collection."),
    html.P("You can explore items by hovering over points and search using semantic queries."),
                        ]),
                        id="explanation-collapse",
                        is_open=False
                    )
                ], className="mb-4 shadow-sm")
            ])
        ]),

        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.InputGroup([
                    dbc.Input(
                        id="search-input",
                        type="text",
                        placeholder="Search...",
                    ),
                    dbc.Button(
                        "Search", 
                        id="search-button", 
                        color="primary",
                        size="sm",
                    ),
                    dbc.Button(
                        "Clear", 
                        id="clear-button", 
                        color="secondary",
                        size="sm",
                    ),
                ], className="d-flex align-items-center")
            ], width={"size": 6, "offset": 3}),  # Center the input group and make it half width
        ], className="mb-3"), 
        dbc.Row([
        dbc.Col([
            html.Label("Number of results:", className="mb-0"),
            dcc.Slider(
                id="search-limit-slider",
                min=1,
                max=50,
                step=1,
                value=5,
                marks={i: str(i) for i in range(1, 51, 1)},
                className="mt-1"
            ),
        ], width={"size": 6, "offset": 3}),
    ], className="mb-3"),       
        html.Br(),
        # Central Visualization (like scatter plot, map etc.)  
        dbc.Row([
                html.Div([
                    dbc.Spinner(
                    id="loading-spinner",
                    type="grow",
                    color="primary",
                    fullscreen=False,
                    children=[
                         # Add a placeholder div
                        html.Div(
                            id="graph-placeholder",
                            children="Select a data source and load data to visualize",
                            style={
                                "height": "700px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "color": "#666",
                                "fontSize": "1.2rem",
                                "fontStyle": "italic",
                                "width": "900px"  # Set width to 70%
                            }
                        ),
                        dcc.Graph(
                        id="umap-graph", 
                        style={
                            "width": "900px",  # Set width to 70%
                            "height": "700px",
                            "display": "none"
                        },
                        config={
                            'scrollZoom': True,
                            'displayModeBar': True,
                            'modeBarButtonsToAdd': ['drawline']
                        }
                    )],
                ),
                    html.Div(id="point-details", 
                    style={
                        "width": "30%",  # Set width to 30%
                        "padding": "15px",
                        "borderLeft": "1px solid #ccc",
                        "overflowY": "auto", 
                        "height": "700px",
                        "minWidth": "250px",
                        "maxWidth": "30%"  # Match the width
                    }),
                ], 
                style={
                    "display": "flex", 
                    "flexDirection": "row",
                    "width": "100%",
                    "gap": "10px",
                    "justifyContent": "space-between"
                }),
            ]),
            html.Div(id="status"),
            dcc.Store(id="omeka-client-config", storage_type="session"),   
        ]),

    # Footer
    html.Footer([
        html.Hr(),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Img(src="./SmartBibl.IA_Solutions.png", height="50"),
                    html.Small([
                        html.Br(),
                        html.A("Géraldine Geoffroy", href="mailto:grldn.geoffroy@gmail.com", className="text-muted")
                    ])
                ]),
                dbc.Col([
                    html.H5("Code source"),
                    html.Ul([
                        html.Li(html.A("Github", href="https://github.com/gegedenice/openalex-explorer", className="text-muted", target="_blank"))
                    ])
                ]),
                dbc.Col([
                    html.H5("Ressources"),
                    html.Ul([
                        html.Li(html.A("Nomic Atlas", href="https://atlas.nomic.ai/", target="_blank", className="text-muted")),
                        html.Li(html.A("Model nomic-embed-text-v1.5", href="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5", target="_blank", className="text-muted")),
                        html.Li(html.A("Model nomic-embed-vision-v1.5", href="https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5", target="_blank", className="text-muted"))
                    ])
                ])
            ])
        ])
    ], className="mt-5 p-3 bg-light border-top")
])

# -------------------- UI Callbacks --------------------
# ------------------------------------------------------

##-------------------- Tabs Callbacks --------------------
@app.callback(
    Output("data-tab-content", "children"),
    Input("data-tabs", "value")
)
def render_tab_content(tab):
    if tab == "omeka":
        return html.Div([
            html.Div([
                html.H5("Harvest data from an Omeka S instance", className="mb-3"),
                # API URL input with full width
                dbc.InputGroup([
                    dbc.Input(
                        id="api-url",
                        value="https://your-omeka-instance.org",
                        type="url",
                        placeholder="Enter your Omeka S instance URL",
                        className="mb-2"
                    ),
                ]),
                # Buttons and dropdowns container
                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Load Item Sets",
                                id="load-sets",
                                color="link",
                                size="sm",
                                className="w-100 mb-2"
                            ),
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id="items-sets-dropdown",
                                placeholder="Select a collection",
                                className="mb-2"
                            ),
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(
                                id="table-name",
                                value="Enter a table name for data storage",
                                type="text",
                                placeholder="New table name",
                                className="mb-2"
                            ),
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Process Omeka Collection",
                                id="process-omeka",
                                color="success",
                                size="sm",
                                className="mt-2"
                            ),
                        ]),
                    ]),
                ], fluid=True, className="p-0"),
            ], className="p-3"),
        ], className="border rounded bg-white shadow-sm")
    elif tab == "lance":
        # Get tables at runtime
        tables = manager.list_tables()
        return html.Div([
            html.H5("From LanceDB", className="mb-3"),
            html.Div([
                dbc.RadioItems(
                    id="db-tables-radio",
                    options=[{"label": t, "value": t} for t in tables],
                    value=tables[0] if tables else None,
                    className="mb-3"
                ),
                dbc.Button("Display Table", id="load-data-db", color="success", size="sm", className="me-2"),
                dbc.Button("Drop Table", id="drop-data-db", color="danger", size="sm"),
            ]) if tables else html.P("No tables available in LanceDB", className="text-muted"),
        ], className="border rounded bg-white shadow-sm p-3")

    return html.Div("Invalid tab selected.")

# -------------------- Collpase callback --------------------
@app.callback(
    Output("explanation-collapse", "is_open"),
    Input("explanation-toggle", "n_clicks"),
    prevent_initial_call=True
)
def toggle_collapse(n):
    return n % 2 == 1

# -------------------- Graph placeholder Toggle callback --------------------
@app.callback(
    Output("graph-placeholder", "style"),
    Output("umap-graph", "style"),
    [Input("umap-graph", "figure")],
    prevent_initial_call=True
)
def toggle_graph_visibility(figure):
    if figure is None:
        return {"display": "flex"}, {"display": "none"}
    return {"display": "none"}, {
        "flex": 3,
        "width": "100%",
        "display": "block"
    }

# -------------------- Features Callbacks --------------------
# ------------------------------------------------------------

## -------------------- Load Omeka collections callback--------------------

@app.callback(
    Output("items-sets-dropdown", "options"),
    Output("omeka-client-config", "data"),
    Input("load-sets", "n_clicks"),
    State("api-url", "value"),
    prevent_initial_call=True
)
def load_item_sets(n_clicks, base_url):
    if n_clicks is None:  # Add this check
        raise PreventUpdate
    client = OmekaSClient(base_url, "...", "...", 50)
    try:
        item_sets = client.list_all_item_sets()
        options = [{"label": s.get('dcterms:title', [{}])[0].get('@value', 'N/A'), "value": s["o:id"]} for s in item_sets]
        return options, {
            "base_url": base_url,
            "key_identity": "...",
            "key_credential": "...",
            "default_per_page": 50
        }
    except Exception as e:
        return dash.no_update, dash.no_update

## -------------------- Load & Process Omeka items callback--------------------
@app.callback(
    Output("umap-graph", "figure"),
    Output("status", "children"),
    Input("process-omeka", "n_clicks"),  # Changed ID to match new button
    State("items-sets-dropdown", "value"),
    State("omeka-client-config", "data"),
    State("table-name", "value"),
    prevent_initial_call=True
)
def handle_omeka_data(n_clicks, item_set_id, client_config, table_name):
    if not n_clicks or not client_config:
        raise PreventUpdate

    client = OmekaSClient(
        base_url=client_config["base_url"],
        key_identity=client_config["key_identity"],
        key_credential=client_config["key_credential"]
    )
    
    df_omeka = harvest_omeka_items(client, item_set_id=item_set_id)
    items = df_omeka.to_dict(orient="records")
    records_with_text = [helpers.add_concatenated_text_field_exclude_keys(item, keys_to_exclude=['id','images_urls'], text_field_key='text', pair_separator=' - ') for item in items]
    df = helpers.prepare_df_atlas(pd.DataFrame(records_with_text), id_col='id', images_col='images_urls')
    
    text_embed = helpers.generate_text_embed(df['text'].tolist())
    img_embed = helpers.generate_img_embed(df['images_urls'].tolist())
    # Convert to tensors if needed
    text_tensor = torch.tensor(text_embed)
    img_tensor = torch.tensor(img_embed)

    # Average then normalize
    combined = (0.7 * text_tensor + 0.3 * img_tensor)
    normalized_embeddings = F.normalize(combined, p=2, dim=1)

    embeddings = normalized_embeddings.numpy()
    df["embeddings"] = embeddings.tolist()

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    umap_embeddings = reducer.fit_transform(embeddings)
    df["umap_embeddings"] = umap_embeddings.tolist()

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric="euclidean")
    cluster_labels = clusterer.fit_predict(umap_embeddings)
    df["Cluster"] = cluster_labels

    vectorizer = text.TfidfVectorizer(max_features=1000, stop_words=list(french_stopwords), lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(df["text"].astype(str).tolist())
    top_words = []
    for label in sorted(df["Cluster"].unique()):
        if label == -1:
            top_words.append("Noise")
            continue
        mask = (df["Cluster"] == label).to_numpy().nonzero()[0]
        cluster_docs = tfidf_matrix[mask]
        mean_tfidf = cluster_docs.mean(axis=0)
        mean_tfidf = np.asarray(mean_tfidf).flatten()
        top_indices = mean_tfidf.argsort()[::-1][:5]
        terms = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        top_words.append(", ".join(terms))
    cluster_name_map = {label: name for label, name in zip(sorted(df["Cluster"].unique()), top_words)}
    df["Topic"] = df["Cluster"].map(cluster_name_map)

    manager.initialize_table(table_name)
    manager.add_entry(table_name, df.to_dict(orient="records"))
    
    return create_umap_plot(df)

## -------------------- Load LanceDB data callback--------------------
@app.callback(
    Output("umap-graph", "figure", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("load-data-db", "n_clicks"),
    State("db-tables-radio", "value"), 
    prevent_initial_call=True
)
def handle_db_data(n_clicks, db_table):
    if not n_clicks or not db_table:
        raise PreventUpdate
        
    items = manager.get_content_table(db_table)
    df = pd.DataFrame(items)
    df = df.dropna(axis=1, how='all')
    df = df.fillna('')
    #umap_embeddings = np.array(df["umap_embeddings"].tolist())
    return create_umap_plot(df)

## -------------------- plotly Hover datapoint callback--------------------
@app.callback(
    Output("point-details", "children"),
    Input("umap-graph", "hoverData")
)
def show_point_details(hoverData):
    if not hoverData:
        return html.Div("🖱️ Hover a point to see more details.", style={"color": "#888"})
    id,item_id, img_url, title, desc = hoverData["points"][0]["customdata"]
    return html.Div([
        html.H4(title, style={"fontSize": "1.2rem"}),  # Reduced header size
        html.P(f"Item ID: {item_id}", style={"fontSize": "0.9rem", "color": "#666"}),  # Smaller text
        html.Img(src=img_url, style={
            "maxWidth": "300px",  # Fixed max width instead of 100%
            "height": "auto",     # Maintain aspect ratio
            "marginBottom": "10px",
            "borderRadius": "5px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)" 
        }),
        html.P(desc or "No description available.",
               style={"lineHeight": "1.6", "color": "#444", "fontSize": "0.9rem"})  # Smaller text
    ])

## -------------------- Search & filter datapoint callback--------------------    
@app.callback(
    Output("umap-graph", "figure", allow_duplicate=True),
    Input("search-button", "n_clicks"),
    Input("search-limit-slider", "value"),  # Add slider input
    State("search-input", "value"),
    State("db-tables-radio", "value"),
    State("umap-graph", "figure"),
    prevent_initial_call=True
)
def filter_points(n_clicks, limit, search_query, table, current_fig):
    # Get the trigger that caused the callback
    trigger = ctx.triggered_id
    
    # If slider changed but no search query exists, don't update
    if trigger == "search-limit-slider" and not search_query:
        return dash.no_update
        
    if not search_query:
        # Reset visibility of all points
        for trace in current_fig['data']:
            trace['visible'] = True
        return current_fig
        
    # Generate text embedding
    query_embed = helpers.generate_text_embed([f"search_query: {search_query}"]).tolist()  
    
    # Perform semantic search using the slider value
    matching = manager.semantic_search(
        table_name=table,
        query_embed=query_embed, 
        limit=limit  # Use the slider value
    )
    
    matching_ids = [item['id'] for item in json.loads(matching)]
    print(f"Searching for '{search_query}' with limit {limit}")
    print(f"Found {len(matching_ids)} matches")
    
    # Update visibility of points
    fig = go.Figure(current_fig)
    for trace in fig.data:
        point_ids = [point[0] for point in trace['customdata']]
        selected_indices = [i for i, id in enumerate(point_ids) if id in matching_ids]
        trace.update(
            selectedpoints=selected_indices,
            unselected=dict(marker=dict(opacity=0.1))
        )
    
    return fig

## -------------------- Clear search callback--------------------    
@app.callback(
    Output("umap-graph", "figure", allow_duplicate=True),
    Output("search-input", "value"),  # Clear the search input
    Input("clear-button", "n_clicks"),
    State("umap-graph", "figure"),
    prevent_initial_call=True
)
def clear_search(n_clicks, current_fig):
    if not n_clicks:
        raise PreventUpdate
        
    fig = go.Figure(current_fig)
    
    # Reset all points to visible and full opacity
    for trace in fig.data:
        trace.update(
            selectedpoints=None,
            unselected=None,
            opacity=0.8
        )
    
    return fig, ""  # Return cleared figure and empty search input

## -------------------- Drop table callback--------------------
@app.callback(
    Output("db-tables-dropdown", "options",allow_duplicate=True),  # Update dropdown options
    Output("status", "children",allow_duplicate=True),  # Show status message
    Input("drop-data-db", "n_clicks"),
    State("db-tables-radio", "value"),
    State("data-tabs", "value"),
    prevent_initial_call=True
)
def drop_db_data(n_clicks, db_table, current_tab):
    if not n_clicks or not db_table:
        raise PreventUpdate
        
    try:
        success = manager.drop_table(db_table)
        
        if success:
            # Re-render the entire tab content to show updated radio buttons
            return render_tab_content("lance"), f"Table '{db_table}' successfully deleted"
        else:
            return dash.no_update, f"Failed to delete table '{db_table}'"
            
    except Exception as e:
        print(f"Error dropping table: {str(e)}")
        return dash.no_update, f"Error: {str(e)}", dash.no_update

# -------------------- Utility --------------------
# -------------------------------------------------

def harvest_omeka_items(client, item_set_id=None, per_page=50):
    """
    Fetch and parse items from Omeka S.
    Args:
        client: OmekaSClient instance
        item_set_id: ID of the item set to fetch items from (optional)
        per_page: Number of items to fetch per page (default: 50)
    Returns:
        DataFrame containing parsed item data
    """
    print("\n--- Fetching and Parsing Multiple Items by colection---")
    try:
        # Fetch items
        items_list = client.list_all_items(item_set_id=item_set_id, per_page=per_page)
        print(f"Initial fetch: {len(items_list)} items")

        parsed_items_list = []
        for idx, item_raw in enumerate(items_list):
            try:
                print(f"\nProcessing item {idx + 1}/{len(items_list)}")
                if 'o:media' not in item_raw:
                    print(f"Skipping item {idx + 1}: No media found")
                    continue

                parsed = client.digest_item_data(item_raw, prefixes=_DEFAULT_PARSE_METADATA)
                if not parsed:
                    print(f"Skipping item {idx + 1}: Parsing failed")
                    continue

                # Debug media processing
                medias_id = [x["o:id"] for x in item_raw["o:media"]]
                print(f"Found {len(medias_id)} media items")
                
                medias_list = []
                for media_id in medias_id:
                    try:
                        media = client.get_media(media_id)
                        print(f"Media type: {media.get('o:media_type', 'unknown')}")
                        if "image" in media.get("o:media_type", ""):
                            url = media.get('o:original_url')
                            if url:
                                medias_list.append(url)
                            else:
                                print(f"No URL found for media {media_id}")
                    except Exception as e:
                        print(f"Error processing media {media_id}: {str(e)}")

                if medias_list:
                    parsed["images_urls"] = medias_list
                    parsed_items_list.append(parsed)
                    print(f"Added item with {len(medias_list)} images")
                else:
                    print(f"Skipping item {idx + 1}: No valid image URLs found")

            except Exception as e:
                print(f"Error processing item {idx + 1}: {str(e)}")
                print(f"Item raw data: {item_raw}")
                continue

        if not parsed_items_list:
            print("No valid items were parsed!")
            return None

        print(f"\nFinal results:")
        print(f"Total items processed: {len(items_list)}")
        print(f"Successfully parsed items: {len(parsed_items_list)}")
        
        df = pd.DataFrame(parsed_items_list)
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame shape: {df.shape}")
        return df

    except OmekaSClientError as e:
        print(f"Omeka client error: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return None
        
def create_umap_plot(df):
    coords = np.array(df["umap_embeddings"].tolist())
    fig = px.scatter(
        df,
        x=coords[:, 0],
        y=coords[:, 1],
        color="Topic",  # Start with top-level topics
        custom_data=[df["id"], df["item_id"], df["images_urls"], df["Title"], df["Description"]],
        hover_data=None,
        title="UMAP Projection with HDBSCAN Topics",
        color_discrete_sequence=px.colors.qualitative.D3,
        width=900,
        height=700,
    )
    # Update marker style
    fig.update_traces(
        marker=dict(
            size=12,  # Larger points
            opacity=0.8,  # Slight transparency
            line=dict(width=0),  # Remove borders
            symbol='circle'
        ),
        hoverinfo='none',  # Disable native hover
        hovertemplate=None
        #hovertemplate="<b>%{customdata[1]}</b><br><img src='%{customdata[0]}' height='150'><extra></extra>"
    )
    
    # Convert to a go.Figure object to access additional configuration
    fig = go.Figure(fig)
    
    # Update layout including scroll zoom
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=700,
        margin=dict(t=30, b=30, l=30, r=30),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0)'
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            fixedrange=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            fixedrange=False
        ),
        dragmode='pan',
        modebar_add=[
            'zoom',
            'pan',
            'zoomIn',
            'zoomOut',
            'resetScale'
        ],
    )
    
    return fig, f"Loaded {len(df)} items and projected into 2D."

if __name__ == "__main__":
    app.run(debug=True)
