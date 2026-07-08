# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np


class TestCase(unittest.TestCase):
    def _GetNdArray(self, a):
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        return a

    def assertAllEqual(self, a, b):
        """Assert two array-likes have identical values (NaN == NaN for floats).

        a: expected array-like.
        b: actual array-like.
        """
        a = self._GetNdArray(a)
        b = self._GetNdArray(b)
        self.assertEqual(
            a.shape, b.shape,
            "Shape mismatch: expected %s, got %s." % (a.shape, b.shape))
        same = (a == b)

        if a.dtype == np.float32 or a.dtype == np.float64:
            same = np.logical_or(same, np.logical_and(np.isnan(a),
                                                      np.isnan(b)))
        if not np.all(same):
            diff = np.logical_not(same)
            if a.ndim:
                x = a[np.where(diff)]
                y = b[np.where(diff)]
                print("not equal where = ", np.where(diff))
            else:
                # np.where is broken for scalars
                x, y = a, b
            print("not equal lhs = ", x)
            print("not equal rhs = ", y)
            np.testing.assert_array_equal(a, b)

    def assertAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg: str = ""):
        """Assert two array-likes (or flat dicts of same) are close within tol.

        a, b: expected/actual array-like or flat dict; both must be dicts or neither.
        rtol, atol: relative / absolute tolerance.
        Raises ValueError if only one of a, b is a dict. No nested dicts.
        """
        is_a_dict = isinstance(a, dict)
        if is_a_dict != isinstance(b, dict):
            raise ValueError(f"Can't compare dict to non-dict, %s vs %s. {msg}" %
                             (a, b))
        if is_a_dict:
            self.assertCountEqual(a.keys(),
                                  b.keys(),
                                  msg=f"mismatched keys, expected %s, got %s. {msg}" % 
                                  (a.keys(), b.keys()))
            for k in a:
                self._assertArrayLikeAllClose(a[k],
                                              b[k],
                                              rtol=rtol,
                                              atol=atol,
                                              msg=f"%s: expected %s, got %s. {msg}" %
                                              (k, a, b))
        else:
            self._assertArrayLikeAllClose(a, b, rtol=rtol, atol=atol, msg=msg)

    def _assertArrayLikeAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
        a = self._GetNdArray(a)
        b = self._GetNdArray(b)
        self.assertEqual(
            a.shape, b.shape,
            "Shape mismatch: expected %s, got %s." % (a.shape, b.shape))
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            # Tolerance is atol + rtol*|b|; print the elements that violate it.
            cond = np.logical_or(
                np.abs(a - b) > atol + rtol * np.abs(b),
                np.isnan(a) != np.isnan(b))
            if a.ndim:
                x = a[np.where(cond)]
                y = b[np.where(cond)]
                print("not close where = ", np.where(cond))
            else:
                # np.where is broken for scalars
                x, y = a, b
            print("not close lhs = ", x)
            print("not close rhs = ", y)
            print("not close dif = ", np.abs(x - y))
            print("not close tol = ", atol + rtol * np.abs(y))
            print("dtype = %s, shape = %s" % (a.dtype, a.shape))
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


def params_grid(*params):
    """Cartesian product of the given param lists, as a list of combinations."""
    size = len(params)
    length = 1
    for p in params:
        length *= len(p)
    sizes = [len(p) for p in params]
    counter = [0] * size
    total = []
    for i in range(length):
        total.append([0] * size)
    for i in range(length):
        for j in range(size):
            total[i][j] = params[j][counter[j]]
        counter[size - 1] += 1
        for c in range(size - 1, -1, -1):
            if (counter[c] == sizes[c] and c > 0):
                counter[c - 1] += 1
                counter[c] = 0
    return total


def generate_sparse_data(shape,
                         num_points,
                         num_channels,
                         integer=False,
                         data_range=(-1, 1),
                         with_dense=True,
                         dtype=np.float32,
                         shape_scale = 1):
    """Build random sparse voxel data for a batch (uses global np.random state).

    shape: dense spatial shape. num_points: per-batch point counts.
    num_channels: feature width. integer/data_range/dtype: value sampling.
    with_dense: also emit a dense tensor. shape_scale: coord subsampling stride.
    Returns dict with 'features', 'indices' (int32, batch col last), and
    optionally 'features_dense'.
    """
    dense_shape = shape
    ndim = len(dense_shape)
    num_points = np.array(num_points)
    batch_size = len(num_points)
    batch_indices = []
    coors_total = np.stack(np.meshgrid(*[np.arange(0, s // shape_scale) for s in shape]),
                           axis=-1)
    coors_total = coors_total.reshape(-1, ndim) * shape_scale
    for i in range(batch_size):
        np.random.shuffle(coors_total)
        inds_total = coors_total[:num_points[i]]
        inds_total = np.pad(inds_total, ((0, 0), (0, 1)),
                            mode="constant",
                            constant_values=i)
        batch_indices.append(inds_total)
    if integer:
        sparse_data = np.random.randint(data_range[0],
                                        data_range[1],
                                        size=[num_points.sum(),
                                              num_channels]).astype(dtype)
    else:
        sparse_data = np.random.uniform(data_range[0],
                                        data_range[1],
                                        size=[num_points.sum(),
                                              num_channels]).astype(dtype)

    res = {
        "features": sparse_data.astype(dtype),
    }
    if with_dense:
        dense_data = np.zeros([batch_size, num_channels, *dense_shape],
                              dtype=sparse_data.dtype)
        start = 0
        for i, inds in enumerate(batch_indices):
            for j, ind in enumerate(inds):
                dense_slice = (i, slice(None), *ind[:-1])
                dense_data[dense_slice] = sparse_data[start + j]
            start += len(inds)
        res["features_dense"] = dense_data.astype(dtype)
    batch_indices = np.concatenate(batch_indices, axis=0)
    res["indices"] = batch_indices.astype(np.int32)
    return res
