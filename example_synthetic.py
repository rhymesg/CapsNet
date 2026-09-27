# Synthetic smoke checks for the traffic CapsNet implementation.
# See README.md#citation and docs/running.md#synthetic-example.
import argparse

import numpy as np

from MinMaxScaler import MinMaxScaler


def check_scaler():
    speeds = np.array([[10.0, 20.0], [30.0, 50.0]])
    scaler = MinMaxScaler(speeds, feature_range=(0, 1))
    scaled = scaler.scale(speeds)
    np.testing.assert_allclose(scaled, [[0.0, 0.25], [0.5, 1.0]])
    np.testing.assert_allclose(scaler.scale_inverse(scaled), speeds)
    print('Scaler check passed: range [0, 1], inverse round trip.')


def check_networks():
    import tensorflow as tf
    import main_capsnet as model

    model.batch_size = 1
    model.time_step_in = 10
    model.time_step_out = 1
    model.num_sensor = 8
    model.num_data_train = 16
    model.num_data_eval = 16
    rng = np.random.RandomState(7)
    model.dataset_train = rng.uniform(0, 1, size=(16, 8))
    model.dataset_eval = rng.uniform(0, 1, size=(16, 8))
    model.get_dataset_image(shuffle=False)
    x_batch, y_batch = model.get_batch(0, train=True)
    x_eval, y_eval = model.get_batch(0, train=False)
    assert x_batch.shape == x_eval.shape == (1, 10, 8, 1)
    assert y_batch.shape == y_eval.shape == (1, 8)
    np.testing.assert_allclose(x_batch[0, :, :, 0], model.dataset_train[:10])
    np.testing.assert_allclose(y_batch[0], model.dataset_train[10])
    np.testing.assert_allclose(y_eval[0], model.dataset_eval[10])

    for network in (model.CapsNet, model.CNN_Ma):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            tf.set_random_seed(7)
            inputs = tf.placeholder(tf.float32, shape=x_batch.shape)
            targets = tf.placeholder(tf.float32, shape=y_batch.shape)
            predictions = network(inputs)
            loss = tf.sqrt(tf.losses.mean_squared_error(targets, predictions))
            update = tf.train.AdamOptimizer(0.0005).minimize(loss)
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                session.run(update, {inputs: x_batch, targets: y_batch})
                result, value = session.run(
                    [predictions, loss], {inputs: x_eval, targets: y_eval})
            assert result.shape == (1, 8)
            assert np.isfinite(result).all() and np.isfinite(value)
            print('{} check passed: predictions (1, 8), finite loss.'.format(
                network.__name__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scaler-only', action='store_true',
                        help='check only NumPy scaling; no TensorFlow required')
    args = parser.parse_args()
    check_scaler()
    if not args.scaler_only:
        check_networks()


if __name__ == '__main__':
    main()
