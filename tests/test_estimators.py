"""Tests de non-régression : les estimateurs KhiopsFast* doivent reproduire exactement khiops.sklearn,
avec le déploiement par le runner Khiops ("khiops") comme avec khiops-kni ("kni")."""

import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from khiops import core as kh
from khiops.sklearn import KhiopsClassifier, KhiopsEncoder, KhiopsRegressor
from scipy import sparse
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from khiops_fast import KhiopsFastClassifier, KhiopsFastEncoder, KhiopsFastRegressor, shutdown_kni_pool
from khiops_fast import kni_deploy

SAMPLES = kh.get_samples_dir()
DEPLOYMENTS = ["khiops", "kni", "kni_parallel"]


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    """Makes kni_parallel really use its processes on the small test datasets"""
    monkeypatch.setattr(kni_deploy, "MIN_ROWS_PER_CHUNK", 10)


def deploy(deployment):
    return {"deployment": deployment, "n_jobs": 2 if deployment == "kni_parallel" else None}


def read(*parts):
    return pd.read_csv(os.path.join(SAMPLES, *parts), sep="\t")


def assert_same_frame(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert a.shape == b.shape


@pytest.fixture(scope="module")
def adult():
    df = read("Adult", "Adult.txt").sample(3000, random_state=0)
    X, y = df.drop(columns=["class", "Label"], errors="ignore"), df["class"]
    return train_test_split(X, y, test_size=0.3, random_state=0)


@pytest.fixture(scope="module")
def iris():
    df = read("Iris", "Iris.txt")
    X, y = df.drop(columns=["Class"]), df["Class"]
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


@pytest.fixture(scope="module")
def accidents():
    main = read("Accidents", "Accidents.txt").sample(2000, random_state=0)
    vehicles = read("Accidents", "Vehicles.txt")
    vehicles = vehicles[vehicles["AccidentId"].isin(main["AccidentId"])]
    users = read("Accidents", "Users.txt")
    users = users[users["AccidentId"].isin(main["AccidentId"])]
    return main, vehicles, users


def star(main, vehicles):
    return {"main_table": (main.drop(columns=["Gravity"]), ["AccidentId"]),
            "additional_data_tables": {"Vehicles": (vehicles, ["AccidentId", "VehicleId"])}}


def snowflake(main, vehicles, users):
    X = star(main, vehicles)
    X["additional_data_tables"]["Vehicles/Users"] = (users, ["AccidentId", "VehicleId"])
    return X


@pytest.fixture(scope="module")
def star_models(accidents):
    main, vehicles, _ = accidents
    X, y = star(main, vehicles), main["Gravity"]
    return X, KhiopsClassifier(n_trees=0).fit(X, y), KhiopsFastClassifier(n_trees=0).fit(X, y)


# --- API ------------------------------------------------------------------------------------


def test_public_api():
    for cls, name in ((KhiopsFastClassifier, "KhiopsFastClassifier"), (KhiopsFastRegressor, "KhiopsFastRegressor"),
                      (KhiopsFastEncoder, "KhiopsFastEncoder")):
        est = cls()
        assert type(est).__name__ == name and repr(est).startswith(name)
        assert est.get_params()["deployment"] == "kni" and est.get_params()["n_jobs"] is None
        assert clone(est.set_params(deployment="khiops")).get_params()["deployment"] == "khiops"
    for fast, ref in ((KhiopsFastClassifier, KhiopsClassifier), (KhiopsFastRegressor, KhiopsRegressor),
                      (KhiopsFastEncoder, KhiopsEncoder)):
        assert set(fast().get_params()) == set(ref().get_params()) | {"deployment", "n_jobs"}


@pytest.mark.parametrize("n_jobs", [0, 1.5, "2", True])
def test_invalid_n_jobs(iris, n_jobs):
    X_train, _, y_train, _ = iris
    with pytest.raises(ValueError, match="n_jobs"):
        KhiopsFastClassifier(n_jobs=n_jobs).fit(X_train, y_train)


def test_resolve_n_jobs():
    available = len(os.sched_getaffinity(0))
    assert kni_deploy.resolve_n_jobs(None) == 1
    assert kni_deploy.resolve_n_jobs(-1) == available
    assert kni_deploy.resolve_n_jobs(-2) == max(1, available - 1)
    assert kni_deploy.resolve_n_jobs(10**6) == available


def test_invalid_deployment(iris):
    X_train, X_test, y_train, _ = iris
    with pytest.raises(ValueError, match="deployment"):
        KhiopsFastClassifier(deployment="gpu").fit(X_train, y_train)
    clf = KhiopsFastClassifier().fit(X_train, y_train).set_params(deployment="gpu")
    with pytest.raises(ValueError, match="deployment"):
        clf.predict(X_test)


# --- mono-table --------------------------------------------------------------------------------


@pytest.mark.parametrize("deployment", DEPLOYMENTS)
@pytest.mark.parametrize("dataset", ["adult", "iris"])
def test_classifier_same_as_khiops(dataset, deployment, request):
    X_train, X_test, y_train, y_test = request.getfixturevalue(dataset)
    ref = KhiopsClassifier().fit(X_train, y_train)
    fast = KhiopsFastClassifier(**deploy(deployment)).fit(X_train, y_train)
    assert list(fast.classes_) == list(ref.classes_)
    assert fast.n_features_in_ == ref.n_features_in_
    np.testing.assert_array_equal(fast.predict(X_test), ref.predict(X_test))
    np.testing.assert_array_equal(fast.predict_proba(X_test), ref.predict_proba(X_test))
    assert fast.score(X_test, y_test) == ref.score(X_test, y_test)


@pytest.mark.parametrize("deployment", DEPLOYMENTS)
def test_classifier_numpy_and_sparse(iris, deployment):
    X_train, X_test, y_train, _ = iris
    for convert in (lambda X: X.to_numpy(), lambda X: sparse.csr_matrix(X.to_numpy())):
        ref = KhiopsClassifier().fit(convert(X_train), y_train)
        fast = KhiopsFastClassifier(**deploy(deployment)).fit(convert(X_train), y_train)
        np.testing.assert_array_equal(fast.predict(convert(X_test)), ref.predict(convert(X_test)))
        np.testing.assert_array_equal(fast.predict_proba(convert(X_test)), ref.predict_proba(convert(X_test)))


@pytest.mark.parametrize("deployment", DEPLOYMENTS)
def test_regressor_same_as_khiops(adult, deployment):
    X_train, X_test, _, _ = adult
    y_train, y_test = X_train["age"], X_test["age"]
    X_train, X_test = X_train.drop(columns=["age"]), X_test.drop(columns=["age"])
    ref = KhiopsRegressor().fit(X_train, y_train)
    fast = KhiopsFastRegressor(**deploy(deployment)).fit(X_train, y_train)
    np.testing.assert_array_equal(fast.predict(X_test), ref.predict(X_test))
    assert fast.score(X_test, y_test) == ref.score(X_test, y_test)


@pytest.mark.parametrize("deployment", DEPLOYMENTS)
@pytest.mark.parametrize("transform_type", ["part_id", "part_label", "dummies", "conditional_info"])
def test_encoder_same_as_khiops(adult, deployment, transform_type):
    X_train, X_test, y_train, _ = adult
    params = {"transform_type_categorical": transform_type, "transform_type_numerical": transform_type}
    ref = KhiopsEncoder(**params).fit(X_train, y_train)
    fast = KhiopsFastEncoder(**deploy(deployment), **params).fit(X_train, y_train)
    assert_same_frame(fast.transform(X_test), ref.transform(X_test))


def test_encoder_fit_transform(adult):
    X_train, _, y_train, _ = adult
    ref = KhiopsEncoder().fit_transform(X_train, y_train)
    assert_same_frame(KhiopsFastEncoder().fit_transform(X_train, y_train), ref)


# --- multi-table -------------------------------------------------------------------------------


@pytest.mark.parametrize("deployment", DEPLOYMENTS)
def test_classifier_star_same_as_khiops(star_models, deployment):
    X, ref, fast = star_models
    fast.set_params(**deploy(deployment))
    np.testing.assert_array_equal(fast.predict(X), ref.predict(X))
    np.testing.assert_array_equal(fast.predict_proba(X), ref.predict_proba(X))


@pytest.mark.parametrize("deployment", ["kni", "kni_parallel"])
def test_classifier_star_unsorted_and_missing_secondary(star_models, accidents, deployment):
    _, ref, fast = star_models
    main, vehicles, _ = accidents
    main = main.sample(frac=1, random_state=1)
    vehicles = vehicles[~vehicles["AccidentId"].isin(main["AccidentId"].iloc[:200])].sample(frac=1, random_state=2)
    X = star(main, vehicles)
    expected = ref.predict_proba(X)
    np.testing.assert_array_equal(fast.set_params(**deploy(deployment)).predict_proba(X), expected)


@pytest.mark.parametrize("deployment", DEPLOYMENTS)
def test_snowflake_same_as_khiops(accidents, deployment):
    main, vehicles, users = accidents
    X, y = snowflake(main, vehicles, users), main["Gravity"]
    ref = KhiopsClassifier(n_trees=0).fit(X, y)
    fast = KhiopsFastClassifier(n_trees=0, **deploy(deployment)).fit(X, y)
    np.testing.assert_array_equal(fast.predict_proba(X), ref.predict_proba(X))
    ref_enc = KhiopsEncoder().fit(X, y)
    fast_enc = KhiopsFastEncoder(**deploy(deployment)).fit(X, y)
    assert_same_frame(fast_enc.transform(X), ref_enc.transform(X))


# --- robustesse ---------------------------------------------------------------------------------


def test_threads_are_serialized(adult):
    X_train, X_test, y_train, _ = adult
    clf = KhiopsFastClassifier(deployment="kni").fit(X_train, y_train)
    expected = clf.predict_proba(X_test)
    chunks = np.array_split(np.arange(len(X_test)), 8)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda idx: clf.predict_proba(X_test.iloc[idx]), chunks))
    np.testing.assert_array_equal(np.vstack(results), expected)


def test_pickle_and_switch_after_fit(iris):
    X_train, X_test, y_train, _ = iris
    clf = pickle.loads(pickle.dumps(KhiopsFastClassifier(deployment="khiops").fit(X_train, y_train)))
    expected = clf.predict_proba(X_test)
    assert clf.deployment == "khiops"
    np.testing.assert_array_equal(clf.set_params(deployment="kni").predict_proba(X_test), expected)
    np.testing.assert_array_equal(clf.set_params(deployment="kni_parallel", n_jobs=2).predict_proba(X_test), expected)


# --- kni_parallel --------------------------------------------------------------------------------


def test_parallel_small_input_stays_in_process(iris, monkeypatch):
    X_train, X_test, y_train, _ = iris
    monkeypatch.setattr(kni_deploy, "MIN_ROWS_PER_CHUNK", 5000)
    shutdown_kni_pool()
    clf = KhiopsFastClassifier(deployment="kni_parallel", n_jobs=4).fit(X_train, y_train)
    expected = clf.set_params(deployment="kni").predict_proba(X_test)
    np.testing.assert_array_equal(clf.set_params(deployment="kni_parallel").predict_proba(X_test), expected)
    assert kni_deploy._pool is None


def test_parallel_pool_reused_and_resized(adult):
    X_train, X_test, y_train, _ = adult
    clf = KhiopsFastClassifier(deployment="kni_parallel", n_jobs=2).fit(X_train, y_train)
    expected = clf.predict_proba(X_test)
    pool = kni_deploy._pool
    np.testing.assert_array_equal(clf.predict_proba(X_test), expected)
    assert kni_deploy._pool is pool and kni_deploy._pool_size == 2
    np.testing.assert_array_equal(clf.set_params(n_jobs=3).predict_proba(X_test), expected)
    assert kni_deploy._pool is not pool and kni_deploy._pool_size == 3
    shutdown_kni_pool()
    assert kni_deploy._pool is None


def test_parallel_worker_error_is_raised_and_pool_survives(adult):
    X_train, X_test, y_train, _ = adult
    clf = KhiopsFastClassifier(deployment="kni_parallel", n_jobs=2).fit(X_train, y_train)
    expected = clf.predict_proba(X_test)
    bad = X_test.copy()
    # A 9 MB field exceeds the maximum KNI record length (8 MB) in the last chunk only
    bad.iloc[-1, bad.columns.get_loc("occupation")] = "x" * (9 * 1024 * 1024)
    with pytest.raises(RuntimeError, match="chunk 1"):
        clf.predict_proba(bad)
    np.testing.assert_array_equal(clf.predict_proba(X_test), expected)
