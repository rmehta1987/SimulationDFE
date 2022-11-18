from .base import Base

class SplitFactorCoupling(Base):
    def __init__(self, c_in, factor, height, width, args):
        super().__init__()
        self.n_channels = args.n_channels
        self.kernel = 3
        self.input_channel = c_in
        self.round_approx = args.round_approx

        if args.variable_type == 'discrete':
            self.round = BackRound(
                args, inverse_bin_width=2**args.n_bits)
        else:
            self.round = None

        self.split_idx = c_in - (c_in // factor)

        self.nn = NN(
            args=args,
            c_in=self.split_idx,
            c_out=c_in - self.split_idx,
            height=height,
            width=width,
            kernel=self.kernel,
            nn_type=args.coupling_type)

    def forward(self, z, ldj, reverse=False):
        z1 = z[:, :self.split_idx, :, :]
        z2 = z[:, self.split_idx:, :, :]

        t = self.nn(z1)

        if self.round is not None:
            t = self.round(t)

        if not reverse:
            z2 = z2 + t
        else:
            z2 = z2 - t

        z = torch.cat([z1, z2], dim=1)

        return z, ldj


class Coupling(Base):
    def __init__(self, c_in, height, width, args):
        super().__init__()

        if args.split_quarter:
            factor = 4
        elif args.splitfactor > 1:
            factor = args.splitfactor
        else:
            factor = 2

        self.coupling = SplitFactorCoupling(
            c_in, factor, height, width, args=args)

    def forward(self, z, ldj, reverse=False):
        return self.coupling(z, ldj, reverse)