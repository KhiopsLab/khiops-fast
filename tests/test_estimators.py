"""Tests de non-régression : les estimateurs KhiopsFast* doivent reproduire exactement khiops.sklearn."""

import os

import numpy as np
import pandas as pd
import pytest
from khiops import core as kh
from khiops.sklearn import KhiopsClassifier, KhiopsEncoder, KhiopsRegressor
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from khiops_fast import KhiopsFastClassifier, KhiopsFastEncoder, KhiopsFastRegressor

SAMPLES = kh.get_samples_dir()


def read(*parts):
    return pd.read_csv(os.path.join(SAMPLES, *parts), sep="\t")


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
    X = {"main_table": (main.drop(columns=["Gravity"]), ["AccidentId"]),
         "additional_data_tables": {"Vehicles": (vehicles, ["AccidentId", "VehicleId"])}}
    return X, main["Gravity"]


def test_public_api():
    for cls, name in ((KhiopsFastClassifier, "KhiopsFastClassifier"), (KhiopsFastRegressor, "KhiopsFastRegressor"),
                      (KhiopsFastEncoder, "KhiopsFastEncoder")):
        est = cls()
        assert type(est).__name__ == name and repr(est).startswith(name)
        assert clone(est).get_params() == est.get_params()
    assert set(KhiopsFastClassifier().get_params()) == set(KhiopsClassifier().get_params())
    assert set(KhiopsFastRegressor().get_params()) == set(KhiopsRegressor().get_params())
    assert set(KhiopsFastEncoder().get_params()) == set(KhiopsEncoder().get_params())


@pytest.mark.parametrize("dataset", ["adult", "iris"])
def test_classifier_same_as_khiops(dataset, request):
    X_train, X_test, y_train, y_test = request.getfixturevalue(dataset)
    ref = KhiopsClassifier().fit(X_train, y_train)
    fast = KhiopsFastClassifier().fit(X_train, y_train)
    assert list(fast.classes_) == list(ref.classes_)
    assert fast.n_features_in_ == ref.n_features_in_
    np.testing.assert_array_equal(fast.predict(X_test), ref.predict(X_test))
    np.testing.assert_allclose(fast.predict_proba(X_test), ref.predict_proba(X_test))
    assert fast.score(X_test, y_test) == pytest.approx(ref.score(X_test, y_test))


def test_classifier_multitable_same_as_khiops(accidents):
    X, y = accidents
    ref = KhiopsClassifier(n_trees=0).fit(X, y)
    fast = KhiopsFastClassifier(n_trees=0).fit(X, y)
    np.testing.assert_array_equal(fast.predict(X), ref.predict(X))
    np.testing.assert_allclose(fast.predict_proba(X), ref.predict_proba(X))


def test_regressor_same_as_khiops(adult):
    X_train, X_test, _, _ = adult
    y_train, y_test = X_train["age"], X_test["age"]
    X_train, X_test = X_train.drop(columns=["age"]), X_test.drop(columns=["age"])
    ref = KhiopsRegressor().fit(X_train, y_train)
    fast = KhiopsFastRegressor().fit(X_train, y_train)
    np.testing.assert_allclose(fast.predict(X_test), ref.predict(X_test))
    assert fast.score(X_test, y_test) == pytest.approx(ref.score(X_test, y_test))


def test_encoder_same_as_khiops(adult):
    X_train, X_test, y_train, _ = adult
    ref = KhiopsEncoder(transform_type_categorical="part_id", transform_type_numerical="part_id").fit(X_train, y_train)
    fast = KhiopsFastEncoder(transform_type_categorical="part_id", transform_type_numerical="part_id").fit(X_train, y_train)
    ref_out, fast_out = ref.transform(X_test), fast.transform(X_test)
    assert fast_out.shape == ref_out.shape
    np.testing.assert_array_equal(np.asarray(fast_out), np.asarray(ref_out))
    np.testing.assert_array_equal(np.asarray(fast.fit_transform(X_train, y_train)),
                                  np.asarray(ref.fit_transform(X_train, y_train)))
