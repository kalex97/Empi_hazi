import pandas as pd
import Functions as fn


s_profiles, s_edges = fn.load_and_select_profiles_and_edges()
full_profiles, full_edges = fn.load_and_select_profiles_and_edges("Y")

s_profiles.set_index("user_id", drop=False, inplace=True)
full_profiles.set_index("user_id", drop=False, inplace=True)

network = fn.create_graph_from_nodes_and_edges(s_profiles, s_edges)

test_set_id = s_profiles[
    s_profiles.TRAIN_TEST == 'TEST']["user_id"].values.tolist()

prediction = pd.DataFrame(test_set_id, columns=['user_id'])
prediction.set_index("user_id", drop=False, inplace=True)

original = full_profiles["gender"]

prediction["original"] = original

prediction["predicted_gender_neighbor"] = prediction.apply(
    lambda row: fn.predictor_neighbor(
        network, row['user_id'], s_profiles), axis=1)

prediction["predicted_gender_triangle"] = prediction.apply(
    lambda row: fn.predictor_triangles(
        network, row['user_id'], s_profiles), axis=1)

fn.acc_test(prediction)
