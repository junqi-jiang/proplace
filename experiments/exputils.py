import numpy as np
import pandas as pd
import proplace
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm
from scipy.spatial import ConvexHull
from rbr.rbr import RBR
from proplace.clfutils import HiddenPrints
from alibi.explainers import cfproto
from carla.recourse_methods import Wachter, Roar


def normalised_l1(xp, x):
    return np.sum(np.abs(xp - x)) / (xp.shape[0])


def get_test_data(clf, m2s, d):
    utildataset = proplace.UtilDataset(len(d.columns) - 1, d.X1.values.shape[1],
                                       proplace.build_dataset_feature_types(d.columns, d.ordinal_features,
                                                                            d.discrete_features,
                                                                            d.continuous_features), d.feat_var_map)
    nodes = proplace.build_inn_nodes(clf, clf.n_layers_)
    clf_class0 = d.X1_train.values[clf.predict(d.X1_train.values) == 0]
    yhats = clf.predict(clf_class0).reshape(1, -1)
    for m in m2s:
        yhats = np.concatenate((yhats, (m.predict(clf_class0)).reshape(1, -1)))
    yhats = yhats.transpose()
    idxs = []
    for i, point_res in enumerate(yhats):
        if np.sum(point_res) == 0:
            idxs.append(i)
    clf_class0 = clf_class0[idxs]
    np.random.seed(10)
    test_xs_vals = clf_class0[np.random.choice(range(clf_class0.shape[0]), 50)]
    test_xs = pd.DataFrame(data=test_xs_vals, columns=d.X1.columns)
    test_xs_carla = pd.DataFrame(
        data=np.concatenate((test_xs_vals, (clf.predict(test_xs)).reshape(1, -1).transpose()), axis=1),
        columns=d.columns)
    return test_xs_vals, test_xs, test_xs_carla, utildataset, nodes


def eval_ces(clf, m2s, ces, d, utildataset, test_xs_vals, delta=0.02, val_only=False):
    print("============================")
    print(np.sum(clf.predict(ces) == 1) / len(ces), ">>> validity")
    avg_val_m2 = 0
    for m in m2s:
        avg_val_m2 += np.sum(m.predict(ces) == 1) / len(ces)
    avg_val_m2 /= len(m2s)
    print(avg_val_m2, ">>> robustness - M2 validity")
    if val_only:
        return []
    nodes = proplace.build_inn_nodes(clf, clf.n_layers_)
    weights, biases = proplace.build_inn_weights_biases(clf, clf.n_layers_, delta, nodes)
    inn_delta = proplace.Inn(clf.n_layers_, delta, nodes, weights, biases)
    avg_l1 = 0
    avg_lof = 0
    avg_delta_validity = 0
    avg_bound = 0
    valid_len = len(ces)
    avg_lof_novel = 0
    if not val_only:
        lof_novel = LocalOutlierFactor(n_neighbors=10, novelty=True)
        lof_novel.fit(d.X1[d.y1.values == 1].values)
    for i, ce in tqdm(enumerate(ces)):
        if np.isnan(ce).any():
            valid_len -= 1
            continue
        if not val_only:
            # lof novelty
            lof_novel_score = 0 if lof_novel.predict(ce.reshape(1, -1))[0] == -1 else 1
            avg_lof_novel += lof_novel_score
            # lof wrt X1 class 1
            lof = LocalOutlierFactor(n_neighbors=10)
            lof.fit(np.concatenate((ce.reshape(1, -1), (d.X1[d.y1.values == 1]).values), axis=0))
            avg_lof += -1 * lof.negative_outlier_factor_[0]
        # robustness
        this_solver = proplace.OptSolver(utildataset, inn_delta, 1, ce, mode=1, M=10000, x_prime=ce, delta=delta)
        found, bound = this_solver.compute_inn_bounds()
        avg_delta_validity += found
        if bound is not None:
            avg_bound += bound
        avg_l1 += normalised_l1(ce, test_xs_vals[i])
    if valid_len == 0:
        print("No valid counterfactuals found")
        return []
    print(avg_delta_validity / valid_len, avg_bound / valid_len, ">>> robustness - delta validity, avg bound")
    print(avg_l1 / valid_len, ">>> l1 cost")
    print(avg_lof_novel / valid_len, ">>> percentage of inliers dataset class 1 points")
    print(avg_lof / valid_len, ">>> avg lof dataset class 1 points")
    print("============================")
    return [np.sum(clf.predict(ces) == 1) / len(ces), avg_delta_validity / valid_len, avg_val_m2, avg_l1 / valid_len,
            avg_lof_novel / valid_len, avg_lof / valid_len]


def run_rnnce(d, clf, nodes, utildataset, delta, test_xs_vals):
    X1_class1_clf = d.X1.values[clf.predict(d.X1) == 1]
    weights, biases = proplace.build_inn_weights_biases(clf, clf.n_layers_, delta, nodes)
    inn_delta = proplace.Inn(clf.n_layers_, delta, nodes, weights, biases)
    valids = []
    for i, x in tqdm(enumerate(X1_class1_clf)):
        lb_solver = proplace.OptSolver(utildataset, inn_delta, 1, x, mode=1, M=10000, x_prime=x)
        found, lb = lb_solver.compute_inn_bounds()
        if found == 1:
            valids.append(i)
    X_class1_clf_robust = X1_class1_clf[valids]
    from sklearn.neighbors import KDTree

    treer = KDTree(X_class1_clf_robust, leaf_size=40)
    idxs = np.array(treer.query(test_xs_vals)[1]).flatten()
    return X_class1_clf_robust[idxs], treer, X_class1_clf_robust


def run_proplace(clf, nodes, utildataset, delta, treer, X_class1_clf_robust, test_xs_vals, k=20):
    weights, biases = proplace.build_inn_weights_biases(clf, clf.n_layers_, delta, nodes)
    inn_delta_0 = proplace.Inn(clf.n_layers_, delta, nodes, weights, biases)
    idxs = np.array(treer.query(test_xs_vals, k=k)[1])
    milpadvplaus_ces = []
    for i, x in tqdm(enumerate(test_xs_vals)):
        # get convex hull
        nns = X_class1_clf_robust[idxs[i]]  # shape (k,D)
        nns = np.concatenate((test_xs_vals[i].reshape(1, -1), nns))  # shape (k+1, D)
        hull = ConvexHull(nns, qhull_options='QJ')
        nns = nns[hull.vertices]
        this_solver = proplace.OptSolverRC4(utildataset, inn_delta_0, 1, x, mode=0, eps=0.01, M=1000, delta=delta,
                                            nns=nns)
        milpadvplaus_ces.append(this_solver.run())
    return milpadvplaus_ces
    # milpadvplaus_scores = eval_ces(milpadvplaus_ces, 0.01)


def run_milpr(clf, d, nodes, utildataset, delta_target, test_xs_vals):
    weights_tar, biases_tar = proplace.build_inn_weights_biases(clf, clf.n_layers_, delta_target, nodes)
    inn_delta_tar = proplace.Inn(clf.n_layers_, delta_target, nodes, weights_tar, biases_tar)

    milpr_ces = []
    for i, x in tqdm(enumerate(test_xs_vals)):
        weights0, biases0 = proplace.build_inn_weights_biases(clf, clf.n_layers_, 0, nodes)
        inn_delta_0 = proplace.Inn(clf.n_layers_, 0, nodes, weights0, biases0)
        # start iteration
        eps = 0.02
        this_ce = proplace.OptSolver(utildataset, inn_delta_0, 1, x, mode=0, eps=eps, M=10000).compute_counterfactual()
        while eps < 100:  # the value before sigmoid equals 5
            target_solver = proplace.OptSolver(utildataset, inn_delta_tar, 1, x, mode=1, eps=eps, x_prime=this_ce)
            if target_solver.compute_inn_bounds()[0] == 1:
                break
            this_ce = proplace.OptSolver(utildataset, inn_delta_0, 1, x, mode=0, eps=eps,
                                         M=10000).compute_counterfactual()
            eps += 0.2
        milpr_ces.append(this_ce)
    return milpr_ces


def run_rbr(clf, d, random_state, test_xs_vals):
    # hyperparameters
    delta_plus = 1.0
    epsilon_op = 1.0    # optimistic: epsilon 0
    epsilon_pe = 1.0    # pessimistic: epsilon 1
    num_samples = 200
    rbr_ces = []
    for idx, x0 in tqdm(enumerate(test_xs_vals)):
        # x_ar, report = method.generate_recourse(x0, model, random_state, params)
        arg = RBR(clf, d.X1_train.values, num_cfacts=1000, max_iter=500, random_state=random_state, device='cpu')
        x_ar = arg.fit_instance(x0, num_samples, 0.2, delta_plus, 1.0, epsilon_op, epsilon_pe, None)
        rbr_ces.append(x_ar)
    return rbr_ces


def run_roar(clf, test_xs_carla, roar_params):
    roar = Roar(clf, roar_params)
    roar_ces = roar.get_counterfactuals(test_xs_carla).values
    return roar_ces


def counterfactual_stability(clf, xp):
    gaussian_samples = np.random.normal(xp, 0.1, (1000, len(xp)))
    model_scores = clf.predict_proba(gaussian_samples)[:, 1]
    return np.mean(model_scores) - np.std(model_scores)


def robx(clf, xplain, xrob, threshold=0.5):
    opt = xplain
    if counterfactual_stability(clf, opt) > threshold:
        return opt
    for a in np.arange(0.05, 1.01, 0.05):
        xp = (1 - a) * xplain + a * xrob
        if counterfactual_stability(clf, xp) > threshold:
            return xp
    return xrob


def run_robx(clf, d, test_xs_vals, tao=None):
    # NN+RobX
    from sklearn.neighbors import KDTree
    X1_class1_clf = d.X1.values[clf.predict(d.X1) == 1]
    X_class1_clf_robust = X1_class1_clf
    tree = KDTree(X_class1_clf_robust, leaf_size=40)
    idxs = np.array(tree.query(test_xs_vals)[1]).flatten()
    tree_ces = X_class1_clf_robust[idxs]

    if tao is None:
        tao = 0.5
    valids_cs = []
    for i, x in tqdm(enumerate(X1_class1_clf)):
        if counterfactual_stability(clf, x) >= tao:
            valids_cs.append(i)
    X1_class1_clf_stable = X1_class1_clf[valids_cs]
    print(X1_class1_clf_stable.shape)
    from sklearn.neighbors import KDTree
    cecestree = KDTree(X1_class1_clf_stable, leaf_size=40)
    nn1 = cecestree.query(test_xs_vals, k=1)[1]
    robx_ces = []
    for i, x in tqdm(enumerate(test_xs_vals)):
        opt_ce = None
        opt_l1 = 100
        for xrob in X1_class1_clf_stable[nn1[i]]:  # stable NN
            this_ce = robx(clf, tree_ces[i], xrob, tao)
            if normalised_l1(this_ce, x) < opt_l1:
                opt_ce = this_ce
                opt_l1 = normalised_l1(this_ce, x)
        robx_ces.append(opt_ce)
    return robx_ces


def run_proto_r(clf, d, utildataset, test_xs_vals, nodes, delta_target=0.03, kap=0.1, theta=10.):
    weights_tar, biases_tar = proplace.build_inn_weights_biases(clf, clf.n_layers_, delta_target, nodes)
    inn_delta_tar = proplace.Inn(clf.n_layers_, delta_target, nodes, weights_tar, biases_tar)
    data_point = np.array(d.X1.values[1])
    shape = (1,) + data_point.shape[:]
    predict_fn = lambda x: clf.predict_proba(x)
    cat_var = {}
    for idx in utildataset.feature_types:
        if utildataset.feature_types[idx] != proplace.DataType.CONTINUOUS_REAL:
            for varidx in utildataset.feat_var_map[idx]:
                cat_var[varidx] = 2
    CEs = []
    for i, x in tqdm(enumerate(test_xs_vals)):
        #kaps = np.concatenate((np.arange(kap, 1.01, 0.3), np.array([1])))
        #kaps = kaps[1:]
        kaps = [0.1, 0.8, 1.0]
        kaps = kaps[1:]
        # make sure the method is at least as good as the non-robust one: using the same default settings first
        best_cf = run_proto_robust_one(d, x, predict_fn, shape, cat_var, theta=theta, kap=kap)
        if best_cf is None:
            best_bound = -10000
        else:
            # test for Delta robustness
            target_solver = proplace.OptSolver(utildataset, inn_delta_tar, 1, x, mode=1, eps=0.01, x_prime=best_cf)
            found, bound = target_solver.compute_inn_bounds()
            if found == 1:
                CEs.append(best_cf)
                continue
            best_bound = bound if bound is not None else -10000
        for kappa in kaps:
            with HiddenPrints():
                this_cf = run_proto_robust_one(d, x, predict_fn, shape, cat_var, theta=theta, kap=kappa)
            if this_cf is None:
                continue
            target_solver = proplace.OptSolver(utildataset, inn_delta_tar, 1, x, mode=1, eps=0.01, x_prime=this_cf)
            found, bound = target_solver.compute_inn_bounds()
            if bound is None:
                continue
            if bound >= best_bound:
                best_cf = this_cf
                best_bound = bound
                if found == 1:
                    break
        CEs.append(best_cf)
    return CEs


def run_proto_robust_one(d, x, predict_fn, shape, cat_var, theta=10., kap=0.1):
    cf = cfproto.CounterFactualProto(predict_fn, shape, use_kdtree=True, feature_range=(
        np.array(d.X1.values.min(axis=0)).reshape(1, -1), np.array(d.X1.values.max(axis=0)).reshape(1, -1)),
                                     cat_vars=cat_var,
                                     ohe=False,theta=theta, kappa=kap, c_steps=10)
    cf.fit(d.X1.values)
    this_point = x
    with HiddenPrints():
        explanation = cf.explain(this_point.reshape(1, -1), Y=None, target_class=None, k=20, k_type='mean',
                                 threshold=0., verbose=True, print_every=100, log_every=100)
    if explanation is None:
        return None
    if explanation["cf"] is None:
        return None
    proto_cf = explanation["cf"]["X"]
    proto_cf = proto_cf[0]
    return np.array(proto_cf)


def run_wachter(model, test_xs_carla, wach_params=None):
    if wach_params is None:
        wach_params = {
            "lr": 0.01,
            "lambda_": 0.01,
            "n_iter": 100,
            "t_max_min": 0.5,
            "norm": 1,
            "clamp": False,
            "loss_type": "MSE",
            "y_target": [1],
            "binary_cat_features": False,
        }
    wach = Wachter(model, wach_params)
    wach_ces = wach.get_counterfactuals(test_xs_carla).values
    return wach_ces
