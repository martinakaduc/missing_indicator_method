from __future__ import absolute_import
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout
from keras.layers.wrappers import TimeDistributed
from mimic3models.keras_utils import ExtendMask, GetTimestep, LastTimestep
from keras.layers.merge import Multiply


class Network(Model):
    def __init__(
        self,
        dim,
        batch_norm,
        dropout,
        rec_dropout,
        partition,
        ihm_pos,
        target_repl=False,
        depth=1,
        input_dim=76,
        **kwargs
    ):

        print("==> not used params in network class:", kwargs.keys())

        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth

        # Input layers and masking
        X = Input(shape=(None, input_dim), name="X")
        mX = Masking()(X)

        # Masks
        ihm_M = Input(shape=(1,), name="ihm_M")
        decomp_M = Input(shape=(None,), name="decomp_M")
        los_M = Input(shape=(None,), name="los_M")

        inputs = [X, ihm_M, decomp_M, los_M]

        # Main part of the network
        for i in range(depth):
            mX = LSTM(
                units=dim,
                activation="tanh",
                return_sequences=True,
                recurrent_dropout=rec_dropout,
                dropout=dropout,
            )(mX)
        L = mX

        if dropout > 0:
            L = Dropout(dropout)(L)

        # Output modules
        outputs = []

        # ihm output

        # NOTE: masking for ihm prediction works this way:
        #   if ihm_M = 1 then we will calculate an error term
        #   if ihm_M = 0, our prediction will be 0 and as the label
        #   will also be 0 then error_term will be 0.
        if target_repl > 0:
            ihm_seq = TimeDistributed(Dense(1, activation="sigmoid"), name="ihm_seq")(L)
            ihm_y = GetTimestep(ihm_pos)(ihm_seq)
            ihm_y = Multiply(name="ihm_single")([ihm_y, ihm_M])
            outputs += [ihm_y, ihm_seq]
        else:
            ihm_seq = TimeDistributed(Dense(1, activation="sigmoid"))(L)
            ihm_y = GetTimestep(ihm_pos)(ihm_seq)
            ihm_y = Multiply(name="ihm")([ihm_y, ihm_M])
            outputs += [ihm_y]

        # decomp output
        decomp_y = TimeDistributed(Dense(1, activation="sigmoid"))(L)
        decomp_y = ExtendMask(name="decomp", add_epsilon=True)([decomp_y, decomp_M])
        outputs += [decomp_y]

        # los output
        if partition == "none":
            los_y = TimeDistributed(Dense(1, activation="relu"))(L)
        else:
            los_y = TimeDistributed(Dense(10, activation="softmax"))(L)
        los_y = ExtendMask(name="los", add_epsilon=True)([los_y, los_M])
        outputs += [los_y]

        # pheno output
        if target_repl:
            pheno_seq = TimeDistributed(
                Dense(25, activation="sigmoid"), name="pheno_seq"
            )(L)
            pheno_y = LastTimestep(name="pheno_single")(pheno_seq)
            outputs += [pheno_y, pheno_seq]
        else:
            pheno_seq = TimeDistributed(Dense(25, activation="sigmoid"))(L)
            pheno_y = LastTimestep(name="pheno")(pheno_seq)
            outputs += [pheno_y]

        super(Network, self).__init__(inputs=inputs, outputs=outputs)

    def say_name(self):
        return "{}.n{}{}{}{}.dep{}".format(
            "k_lstm",
            self.dim,
            ".bn" if self.batch_norm else "",
            ".d{}".format(self.dropout) if self.dropout > 0 else "",
            ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
            self.depth,
        )
