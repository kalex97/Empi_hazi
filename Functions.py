import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from itertools import combinations

COLUMNS_STR = """    user_id
    public
    completion_percentage
    gender
    region
    last_login
    registration
    AGE
    body
    I_am_working_in_field
    spoken_languages
    hobbies
    I_most_enjoy_good_food
    pets
    body_type
    my_eyesight
    eye_color
    hair_color
    hair_type
    completed_level_of_education
    favourite_color
    relation_to_smoking
    relation_to_alcohol
    sign_in_zodiac
    on_pokec_i_am_looking_for
    love_is_for_me
    relation_to_casual_sex
    my_partner_should_be
    marital_status
    children
    relation_to_children
    I_like_movies
    I_like_watching_movie
    I_like_music
    I_mostly_like_listening_to_music
    the_idea_of_good_evening
    I_like_specialties_from_kitchen
    fun
    I_am_going_to_concerts
    my_active_sports
    my_passive_sports
    profession
    I_like_books
    life_style
    music
    cars
    politics
    relationships
    art_culture
    hobbies_interests
    science_technologies
    computers_internet
    education
    sport
    movies
    travelling
    health
    companies_brands
    more"""
COLUMNS_LIST = [col.strip() for col in COLUMNS_STR.split("\n")]


def select_relevant_profiles(all_profiles):
    """Select relevant profiles
    criteria:
    * is public
    * region is selected region
    * AGE specified
    * GENDER SPECIFIED
    """
    public_condition = all_profiles["public"] == 1
    age_condition = all_profiles["AGE"] > 14
    gender_condition = all_profiles["gender"].isin([0, 1])
    return all_profiles.loc[public_condition & age_condition & gender_condition]


def select_relevant_edges(all_edges, selected_ids):
    """Select relevant edges for those profiles that are relevant"""
    source_condition = all_edges["source"].isin(selected_ids)
    sink_condition = all_edges["sink"].isin(selected_ids)
    return all_edges.loc[source_condition & sink_condition]


def convert_edges_to_undirected(edges):
    """Convert edges to undirected, and keep only mutual connections"""
    undirected_edges = (
        edges.assign(
            smaller_id=lambda df: df[["source", "sink"]].min(axis=1),
            greater_id=lambda df: df[["source", "sink"]].max(axis=1),
        )
        .groupby(["smaller_id", "greater_id"])
        .agg({"source": "count"})
    )
    print(undirected_edges["source"].value_counts())
    return (
        undirected_edges.loc[undirected_edges["source"] == 2]
        .drop("source", axis=1)
        .reset_index()
    )


def remove_test_set_gender_and_age(nodes):
    """Remove the gender feature from a subset of the nodes for estimation"""
    # todo: the 40k  random can be adjusted if youre working with a subset
    test_profiles = np.random.choice(nodes["user_id"].unique(), 40000,
                                     replace=False)
    nodes["TRAIN_TEST"] = "TRAIN"
    test_condition = nodes["user_id"].isin(test_profiles)
    nodes.loc[test_condition, ["AGE", "gender"]] = np.nan
    nodes.loc[test_condition, ["TRAIN_TEST"]] = "TEST"

    return nodes


def load_and_select_profiles_and_edges(full="N"):
    """load and select relevant profiles, then filter and undirect edges"""
    print("loading profiles")
    # TODO: Add some functionality to only read a subset of the data!
    profiles = pd.read_csv(
        "data/soc-pokec-profiles.txt.gz",
        sep="\t",
        names=COLUMNS_LIST,
        index_col=False,
        usecols=["user_id", "public", "gender", "region", "AGE"],
    )
    print("loading edges")
    edges = pd.read_csv(
        "data/soc-pokec-relationships.txt.gz",
        sep="\t", names=["source", "sink"]
    )
    selected_profiles = select_relevant_profiles(profiles)
    selected_ids = selected_profiles["user_id"].unique()
    selected_edges = select_relevant_edges(edges, selected_ids)
    undirected_edges = convert_edges_to_undirected(selected_edges)
    nodes_with_edges = set(undirected_edges["smaller_id"].unique()).union(
        undirected_edges["greater_id"].unique()
    )
    print(f"Selected profiles: {len(selected_profiles)}")
    print(f"Nodes with edges: {len(nodes_with_edges)}")
    selected_profiles = selected_profiles[
        selected_profiles["user_id"].isin(nodes_with_edges)
    ]
    selected_profiles["AGE"] = selected_profiles["AGE"].clip(upper=50)
    if full == "N":
        selected_profiles = remove_test_set_gender_and_age(selected_profiles)

    return selected_profiles, undirected_edges


def create_graph_from_nodes_and_edges(nodes, edges):
    """Create a networkx graph object with all relevant features"""
    node_attributes = nodes.set_index("user_id").to_dict(orient="index")
    node_attributes_list = [
        (index, attr_dict) for index, attr_dict in node_attributes.items()
    ]
    G = nx.Graph()
    G.add_nodes_from(node_attributes_list)
    G.add_edges_from(edges.values.tolist())
    return G


def add_node_features_to_edges(nodes, edges):
    """Add features of nodes to edges in order to create heatmaps"""
    # TODO: column names could be nicer!
    edges_w_features = edges.merge(
        nodes[["user_id", "AGE", "gender"]].set_index("user_id"),
        how="left",
        left_on="smaller_id",
        right_index=True,
    )
    edges_w_features = edges_w_features.merge(
        nodes[["user_id", "AGE", "gender"]].set_index("user_id"),
        how="left",
        left_on="greater_id",
        right_index=True,
    )
    return edges_w_features


def plot_degree_distribution(G):
    """Plot a degree distribution of a graph
    TODO: log-log binning! To understand this better, check out
    networksciencebook.com"""
    plot_df = (
        pd.Series(dict(G.degree)).value_counts().sort_index().
        to_frame().reset_index()
    )
    plot_df.columns = ["k", "count"]
    plot_df["log_k"] = np.log(plot_df["k"])
    plot_df["log_count"] = np.log(plot_df["count"])
    fig, ax = plt.subplots()
    ax.scatter(plot_df["k"], plot_df["count"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.suptitle("Mutual Degree Distribution")
    ax.set_xlabel("k")
    ax.set_ylabel("count_k")


def plot_age_distribution_by_gender(nodes):
    """Plot a histogram where the color represents gender"""
    plot_df = nodes[["AGE", "gender"]].copy(deep=True).astype(float)
    plot_df["gender"] = plot_df["gender"].replace({0.0: "woman", 1.0: "man"})
    sns.histplot(data=plot_df, x="AGE", hue="gender", bins=np.arange(
        0, 45, 5) + 15).set_title("Age distribution by gender")


def plot_node_degree_by_gender(nodes, G):
    """Plot the average of node degree across age and gender"""
    nodes_w_degree = nodes.set_index("user_id").merge(
        pd.Series(dict(G.degree)).to_frame(),
        how="left",
        left_index=True,
        right_index=True,
    )
    nodes_w_degree = nodes_w_degree.rename({0: "degree"}, axis=1)
    plot_df = (
        nodes_w_degree.groupby(["AGE", "gender"]).agg(
            {"degree": "mean"}).reset_index()
    )
    plot_df["gender"] = plot_df["gender"].replace({0.0: "woman", 1.0: "man"})
    sns.lineplot(data=plot_df, x="AGE", y="degree", hue="gender").set_title(
        "Node degree by gender")


def plot_node_average_neighbor_degree_by_gender(nodes, G):
    """Plot the average of node average_neighbor_degree across age and gender"""
    nodes_w_average_neighbor_degree = nodes.set_index("user_id").merge(
        pd.Series(dict(nx.average_neighbor_degree(G))).to_frame(),
        how="left",
        left_index=True,
        right_index=True,
    )
    nodes_w_average_neighbor_degree = nodes_w_average_neighbor_degree.rename({
        0: "average_neighbor_degree"}, axis=1)
    plot_df = (
        nodes_w_average_neighbor_degree.groupby(["AGE", "gender"]).agg({
            "average_neighbor_degree": "mean"}).reset_index()
    )
    plot_df["gender"] = plot_df["gender"].replace({0.0: "woman", 1.0: "man"})
    sns.lineplot(
        data=plot_df, x="AGE", y="average_neighbor_degree", hue="gender").set_title(
            "Average neighbor degree by gender")


def plot_node_clustering_by_gender(nodes, G):
    """Plot the average of node clustering across age and gender"""
    nodes_w_clustering = nodes.set_index("user_id").merge(
        pd.Series(dict(nx.clustering(G))).to_frame(),
        how="left",
        left_index=True,
        right_index=True,
    )
    nodes_w_clustering = nodes_w_clustering.rename({0: "clustering"}, axis=1)
    plot_df = (
        nodes_w_clustering.groupby(["AGE", "gender"]).agg(
            {"clustering": "mean"}).reset_index()
    )
    plot_df["gender"] = plot_df["gender"].replace({0.0: "woman", 1.0: "man"})
    sns.lineplot(data=plot_df, x="AGE", y="clustering", hue="gender").set_title(
        "Node clustering by gender")


def plot_age_relations_heatmap(edges):
    """Plot a heatmap that represents the distribution of edges"""
    plot_df = edges.groupby(["AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    plot_df_w_w = plot_df.reset_index()
    plot_df_heatmap = plot_df_w_w.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    sns.heatmap(plot_df_heatmap_logged).set_title(
        "Heatmap of relations based on age")


def plot_age_relations_heatmapv2(edges, title):
    """Plot a heatmap that represents the distribution of edges"""
    plot_df = edges.groupby(["gender_x", "gender_y", "AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    plot_df_w_w = plot_df.reset_index()
    plot_df_heatmap = plot_df_w_w.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    sns.heatmap(plot_df_heatmap_logged).set_title(title)


def predictor_neighbor(G, node, profiles):
    neighbor_nodes = set()

    for n in G.neighbors(node):
        neighbor_nodes.add(n)
    m = 0
    f = 0

    for i in neighbor_nodes:
        if pd.notnull(profiles.loc[i]["gender"]):
            if profiles.loc[i]["gender"] == 0:
                f += 1
            else:
                m += 1
    if (m+f) == 0:
        prediction = "NA"
    else:
        prediction = round(m/(m+f))

    return(prediction)


def genderfilter(genders, edges):
    if genders == "FF":
        filtered_df = edges.loc[(edges['gender_x'] == 0) & (edges['gender_y'] == 0)]

    if genders == "FM" or genders == "MF":
        filtered_df = edges.loc[(edges['gender_x'] != edges['gender_y'])]

    if genders == "MM":
        filtered_df = edges.loc[(edges['gender_x'] == 1) & (edges['gender_y'] == 1)]

    return(filtered_df)


def plot_age_relations_heatmap_genderdiff(edges, y_title):
    plot_df = edges.groupby(["gender_x", "gender_y", "AGE_x", "AGE_y", "type"]).agg(
                            {"smaller_id": "count"})

    plot_df.reset_index(level="type", inplace=True)
    plot_df.reset_index(level="AGE_y", inplace=True)

    plot_df["gender_AGE"] = plot_df["AGE_y"]*plot_df["type"]

    plot_df_w_w = plot_df.reset_index()
    plot_df_heatmap = plot_df_w_w.pivot_table(
        index="gender_AGE", columns="AGE_x", values="smaller_id"
    ).fillna(0)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    sns.heatmap(plot_df_heatmap_logged).set_title(
        'Age relationship heatmap ("-" opposite gender)')
    plt.ylabel(y_title)


def neighbor_age_differences(G, node_number, profiles):

    neighbor_nodes = set()

    for n in G.neighbors(node_number):
        neighbor_nodes.add(n)

    ss_neighbor_age = []
    ds_neighbor_age = []

    for k in neighbor_nodes:

        if profiles.loc[node_number]["gender"] == profiles.loc[k]["gender"]:
            ss_neighbor_age.append(profiles.loc[k]["AGE"])

        else:
            ds_neighbor_age.append(profiles.loc[k]["AGE"])

    ss_neighbor_age = abs(ss_neighbor_age-profiles.loc[node_number]["AGE"])

    ds_neighbor_age = abs(ds_neighbor_age-profiles.loc[node_number]["AGE"])

    sl5, sl10, sl20 = 0, 0, 0
    dl5, dl10, dl20 = 0, 0, 0

    for y in ss_neighbor_age:
        if y < 5:
            sl5 += 1
        if y >= 5 and y < 10:
            sl10 += 1
        if y >= 10 and y < 20:
            sl20 += 1

    for x in ds_neighbor_age:
        if x < 5:
            dl5 += 1
        if x >= 5 and x < 10:
            dl10 += 1
        if x >= 10 and x < 20:
            dl20 += 1

    if len(neighbor_nodes) != 0:
        portion = np.divide([sl5, sl10, sl20, dl5, dl10, dl20], len(neighbor_nodes))

    else:
        portion(0, 0, 0, 0, 0, 0)

    return(portion)


def triangles(G, node):
    neighbors1 = set(G.neighbors(node))
    neighbors2 = set(G.neighbors(node))

    triangles = set()

    for neighbors1, neighbors2 in combinations(G.neighbors(node), 2):
        if G.has_edge(neighbors1, neighbors2):
            triangles.add((neighbors1, neighbors2))

    return(triangles)


def predictor_triangles(G, node, profiles):

    if nx.triangles(G, node) != 0:

        prediction = []
        for triangle in triangles(G, node):

            m = 0
            f = 0

            for neighbor in triangle:
                if pd.notnull(profiles.loc[neighbor]["gender"]):
                    if profiles.loc[neighbor]["gender"] == 0:
                        f += 1
                    else:
                        m += 1
                prediction.append(m/2)

        prediction = round(sum(prediction)/len(prediction))

    else:
        prediction = predictor_neighbor(G, node, profiles)

    return(prediction)


def acc_test(df):

    neighbor_acc = 0
    triangle_acc = 0

    for i in df.iterrows():

        if i[1]["original"] == i[1]["predicted_gender_neighbor"]:
            neighbor_acc += 1

        if i[1]["original"] == i[1]["predicted_gender_triangle"]:
            triangle_acc += 1

    neighbor_acc = neighbor_acc/len(df)
    triangle_acc = triangle_acc/len(df)

    return(print(f"accuracy with neighbors: {neighbor_acc}\n accuracy with triangles: {triangle_acc}"))


def portion_separator(G, df, full_profiles):

    df["age_diff"] = df.apply(lambda row: neighbor_age_differences(
            G, row.name, full_profiles), axis=1)

    df[['SL5', 'SL10', "SL20", 'DL5', 'DL10', "DL20"]] = pd.DataFrame(
        df["age_diff"].tolist(), index=df.index)

    age_groups = df.groupby(["AGE"]).agg({
     "SL5": "mean", "SL10": "mean", "SL20": "mean", "DL5": "mean",
     "DL10": "mean", "DL20": "mean"
     }).reset_index()

    return(age_groups)


def portion_plot(g):
    plt.plot(g["AGE"], g["SL5"], label="same, L5")
    plt.plot(g["AGE"], g["SL10"], label="same, L10")
    plt.plot(g["AGE"], g["SL20"], label="same, L20")
    plt.plot(g["AGE"], g["DL5"], label="opposite, L5")
    plt.plot(g["AGE"], g["DL10"], label="opposite, L10")
    plt.plot(g["AGE"], g["DL20"], label="opposite, L20")
    plt.xlabel('AGE')
    plt.ylabel('portion')
    plt.title("Portion of friends gender and age")
    plt.legend()
    plt.show()
