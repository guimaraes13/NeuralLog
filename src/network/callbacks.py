"""
File to store useful callback implementations.
"""
import logging

from tensorflow.keras.callbacks import Callback

logger = logging.getLogger()


def format_metrics_for_epoch(metrics, epoch=None):
    """
    Formats the metrics on `metrics` for the given epoch.

    :param metrics: a dictionary with the metrics names and values.
    The value can be either the metric itself or a list of values for each
    epochs
    :type metrics: dict[str, float], dict[str, list[float]]
    :param epoch: the epoch of the correspondent value, in case the value be
    :type epoch: int
    :return: the formatted message
    :rtype: str
    """
    message = ""
    for k, v in metrics.items():
        if epoch is None:
            message += " - {}: {:.5f}".format(k, v)
        else:
            message += " - {}: {:.5f}".format(k, v[epoch])
    return message


class EpochLogger(Callback):
    """
    Callback to log the measures of the model after each epoch.
    """

    def __init__(self, number_of_epochs):
        """
        Callback to log the measures of the model after each epoch.

        :param number_of_epochs: the total number of epochs of the training
        :type number_of_epochs: int or None
        """
        super(Callback, self).__init__()
        self.number_of_epochs = str(number_of_epochs) or "Unknown"

    def on_epoch_end(self, epoch, logs=None):
        """
        Method called on the end of each epoch.

        :param epoch: the number of the epoch
        :type epoch: int
        :param logs: a dict of data from other callbacks
        :type logs: dict
        """
        logger.info("Epochs %d/%s%s", epoch + 1, self.number_of_epochs,
                    format_metrics_for_epoch(logs))
