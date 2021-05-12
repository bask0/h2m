"""
Hydrological hybrid model.
"""
# ipython - - tune.py - -tune false
import torch
from torch import jit
from torch import nn

# Anotation types
from typing import List, Dict, Tuple
from torch import Tensor

from models.layers import MultiLinear


class HybridModel(torch.jit.ScriptModule):
    """Hydrological model.

    """

    def __init__(
            self,
            num_features: int,
            static_hidden_size: int,
            static_num_layers: int,
            static_enc_size: int,
            static_dropout: float,
            lstm_hidden_size: int,
            task_hidden_size: int,
            task_num_layers: int,
            task_dropout: float) -> None:

        super(HybridModel, self).__init__()

        self.num_hidden = lstm_hidden_size

        self.prediction = False

        self.device = 'gpu'

        self.static_enc = MultiLinear(
            input_size=30,
            hidden_size=static_hidden_size,
            output_size=static_enc_size,
            num_layers=static_num_layers,
            dropout=static_dropout,
            dropout_last=False,
            activation=nn.LeakyReLU(),
            activation_last=True)

        # + 3 because 3 physical state variables: CWD, GW, SWE
        self.lstm_cell = nn.LSTMCell(
            num_features + 3 + static_enc_size, lstm_hidden_size)

        self.fc_multitask = MultiLinear(
            input_size=lstm_hidden_size,
            hidden_size=task_hidden_size,
            output_size=5,
            num_layers=task_num_layers,
            dropout=task_dropout,
            activation=nn.Tanh())

        self.qb_frac = nn.Parameter(torch.zeros(1))
        self.sscale = nn.Parameter(torch.ones(1))

    def to_device(self, device: str) -> None:
        """Move model to given device and adds device attribute.

        Parameters
        ----------
        device: str
            The device to move model to, i.e. 'cpu' or 'cuda'.
        """
        # Remember device.
        self.device = device
        # Move model to device.
        self.to(device)

    @jit.script_method
    def logit_mix(self, val: torch.Tensor, start: float, t: int) -> torch.Tensor:
        # Mix given values from start to val using logit function.
        # Approaches 'val' at around t = 700.
        mixed = (val - start) / \
            (1 + torch.exp(-0.012 * (torch.tensor(t) - 700))) + start
        return mixed

    @jit.script_method
    def forward(
            self,
            x: torch.Tensor,
            prec: torch.Tensor,
            rn: torch.Tensor,
            tair: torch.Tensor,
            static: torch.Tensor,
            h_t: torch.Tensor,
            c_t: torch.Tensor,
            cwd_t: torch.Tensor,
            gw_t: torch.Tensor,
            swe_t: torch.Tensor,
            epoch: int) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward run.

        Parameters
        ----------
        x        The time feature time-series (seq x batch x num_features)
        prec     Precipitation (seq x batch x 1)
        rn       Net radiation (seq x batch x 1)
        tair     Air temperature (seq x batch x 1)
        static   Static variables (batch x 30)
        h_t      LSTM state (h)
        c_t      LSTM state (c)
        cwd_t    CWD initial state
        gw_t     GW initial state
        swe_t    SWE initial state
        epoch    The current epoch
        """

        # CONVENTIONS
        # _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
        #
        # * The tensors are SEQUENCE FIRST: <seq_len, num_batches, num_features>
        # * In var_t, `t` is variable `var` at time `t`. The same variable w/o
        #   `t` refers to the entire time-series
        # * All hydrological variables have shape <seq_len, batch_size, 1>
        # * All hydrological variables at time t have shape <batch_size, 1>
        # * As jit.script is used to optimize the graph, only pyrorch ops
        #   are allowed.

        # INITIALIZE OUTPUTS
        # _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

        # FLUXES (targets)
        #
        ef:           List[torch.Tensor] = []  # evapotranspiration
        et:           List[torch.Tensor] = []  # evapotranspiration
        q:            List[torch.Tensor] = []  # total runoff

        # STATES (targets)
        swe:          List[torch.Tensor] = []  # snow water equivalent
        tws:          List[torch.Tensor] = []  # total water storage

        # STATES (internal)
        cwd:          List[torch.Tensor] = []  # Cumulative water deficit (~soil moisture)
        gw:           List[torch.Tensor] = []  # Groundwater

        # FLUXES (internal)
        winp:         List[torch.Tensor] = []  # Water input (rain + snowmelt)
        smrec:        List[torch.Tensor] = []  # Soilmoisture recharrge
        gwrec:        List[torch.Tensor] = []  # Groiundwater recharge
        qb:           List[torch.Tensor] = []  # Base runoff (groundwater)
        qf:           List[torch.Tensor] = []  # Quick runoff (top layer)
        cwd_overflow: List[torch.Tensor] = []  # Soil overflow, added to groundwater
        smelt:        List[torch.Tensor] = []  # Snow ablation
        sacc:         List[torch.Tensor] = []  # Snow accumulation

        # FACTORS (internal)
        qf_frac:      List[torch.Tensor] = []  # Fraction of input water -> qf
        smrec_frac:   List[torch.Tensor] = []  # Fraction of input water -> sm
        gwrec_frac:   List[torch.Tensor] = []  # Fraction of input water -> gw
        snowc:        List[torch.Tensor] = []  # tair_t * snowc_t -> smelt

        qb_frac:      torch.Tensor = torch.empty(0)  # Base runoff.

        # INITIALIZE STATES
        # _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

        # Static encoding.
        stat_enc = self.static_enc(static)

        # TIME LOOP
        for t, (x_t, prec_t, rn_t, tair_t) in enumerate(zip(x.unbind(0), prec.unbind(0), rn.unbind(0), tair.unbind(0))):

            # Temperature in degree C.
            tairc_t = tair_t - 273.15

            # RECURRENT LAYER
            # _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

            # Takes meteovars, physical states, static encodings, updates LSTM states.
            h_t, c_t = self.lstm_cell(
                torch.cat((x_t, cwd_t, gw_t, swe_t, stat_enc), dim=1), (h_t, c_t))

            # MULTITASK LAYERS
            # _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

            # Multitask layer, maps LSTM state c to physical params.
            fc_out_t = self.fc_multitask(h_t)

            # Split physical params into wfrac (n=3) and other params (n=2).
            wfrac_t_, single_vars_t = fc_out_t.chunk(2, dim=-1)

            # Normalize wfrac (water input fractions): >=0 and sum = 1.
            wfrac_t_ = torch.nn.functional.softplus(wfrac_t_)
            wfrac_t = wfrac_t_ / wfrac_t_.sum(-1, keepdim=True)
            qf_frac_t, smrec_frac_t, gwrec_frac_t = wfrac_t.chunk(3, dim=-1)

            # Split remaining params into evaporative fraction and snow melt coefficient (degree-day).
            ef_t, snowc_t = single_vars_t.chunk(2, dim=-1)

            # Snow melt coefficient >= 0.
            snowc_t = torch.nn.functional.softplus(snowc_t)

            # Evaporative fraction 0-1.
            ef_t = torch.sigmoid(ef_t)

            # HYDROLOGICAL EQUATIONS
            # _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

            # ET ==============================================================

            # Rn (W m-2) * 0.0864 -> LE (MJ m-2 d-1)
            # LE (MJ m-2 d-1) / 2.45 -> ET (mm d-1)
            et_t = ef_t * torch.relu(rn_t) * 0.0864 / 2.45

            # SNOW ============================================================

            # True if tair <= 0.
            is_snow = torch.less_equal(tairc_t, 0.) * 1.0
            # Snow is precip for tair <= 0. Bias-correct snow accumulation.
            sacc_t = prec_t * is_snow * self.sscale
            # Rain is precip if tair > 0.
            rain_t = prec_t * (1.0 - is_snow)
            # Snow melt is positive tair (in degree C) times snow melt coefficient, <= current snow.
            smelt_t = torch.min(swe_t, snowc_t * torch.relu(tairc_t))
            # Snow is past snow plus accumulation minus melt.
            swe_t = swe_t + sacc_t - smelt_t

            # WATER INPUT PARTITIONING ========================================

            # The total water input, precipitation + snow melt.
            winp_t = rain_t + smelt_t

            # The input is partitioned into fast runoff, soil moisture recharge and
            # ground water recharge.
            qf_t = winp_t * qf_frac_t
            smrec_t = winp_t * smrec_frac_t
            gwrec_t = winp_t * gwrec_frac_t

            # The base runoff is a function of the past groundwater, qb_frac is a global parameter.
            qb_frac = torch.sigmoid(self.qb_frac) * 0.01
            qb_t = gw_t * qb_frac

            # STATE UPDATE ========================================

            # Cummulated water deficid is the deficit of water, it has the same dynamics
            # as soil moisture but its upper bound is 0.
            cwd_t = cwd_t + smrec_t - et_t

            # Cummulated water deficit cannot be larger than 0; overflow is added to
            # groundwater. TODO: Change to fast runoff?
            cwd_overflow_t = torch.nn.functional.softplus(cwd_t)
            cwd_t = cwd_t - cwd_overflow_t

            # Groundwater is the past groundwater minus base runoff, plus groundwater
            # recharge and soil overflow.
            gw_t = gw_t - qb_t + gwrec_t + cwd_overflow_t

            # Base and fast runoff sum to total runoff.
            q_t = qb_t + qf_t

            # Total water storage is the sum of all states, the mean is subtracted later to get anomalies.
            tws_t = swe_t + gw_t + cwd_t

            # SAVE VARIABLES ========================================

            # Store values
            et += [et_t]
            q += [q_t]
            tws += [tws_t]

            swe += [swe_t]
            cwd += [cwd_t]
            gw += [gw_t]

            # Only needed if prediction.
            if self.prediction:
                ef += [ef_t]
                smelt += [smelt_t]
                winp += [winp_t]
                qf_frac += [qf_frac_t]
                smrec_frac += [smrec_frac_t]
                gwrec_frac += [gwrec_frac_t]
                smrec += [smrec_t]
                gwrec += [gwrec_t]
                qb += [qb_t]
                qf += [qf_t]
                cwd_overflow += [cwd_overflow_t]
                sacc += [sacc_t]
                snowc += [snowc_t]

        # Subtract mean of TWS to center around 0.
        tws = torch.stack(tws, dim=1)
        tws -= tws.mean(dim=(1, 2), keepdim=True)

        if self.prediction:
            ret_vars = {
                'tws':                  tws,
                'swe':                  torch.stack(swe, dim=1),
                'ef':                   torch.stack(ef, dim=1),
                'et':                   torch.stack(et, dim=1),
                'q':                    torch.stack(q, dim=1),
                'winp':                 torch.stack(winp, dim=1),
                'smrec':                torch.stack(smrec, dim=1),
                'gwrec':                torch.stack(gwrec, dim=1),
                'qb':                   torch.stack(qb, dim=1),
                'qf':                   torch.stack(qf, dim=1),
                'cwd':                  torch.stack(cwd, dim=1),
                'cwd_overflow':         torch.stack(cwd_overflow, dim=1),
                'gw':                   torch.stack(gw, dim=1),
                'qf_frac':              torch.stack(qf_frac, dim=1),
                'smrec_frac':           torch.stack(smrec_frac, dim=1),
                'gwrec_frac':           torch.stack(gwrec_frac, dim=1),
                'sacc':                 torch.stack(sacc, dim=1),
                'smelt':                torch.stack(smelt, dim=1),
                'snowc':                torch.stack(snowc, dim=1),
                'stat_enc':             stat_enc,
                'qb_frac':              qb_frac,
                'sscale':               self.sscale
            }

        else:
            ret_vars = {
                'tws':  tws,
                'swe':  torch.stack(swe, dim=1),
                'et':   torch.stack(et, dim=1),
                'q':    torch.stack(q, dim=1),
                'gw':   torch.stack(gw, dim=1),
                'cwd':  torch.stack(cwd, dim=1)
            }

        return ret_vars, h_t, c_t, cwd_t, gw_t, swe_t

    def predict(self, mode=True):
        """Sets the module in prediction mode.

        Sets the attribute 'self.prediction' which is used internally
        to change certain model behaviour, e.g. returning the internal
        variables (like qb) in addition to the target varaibles.

        Returns:
            Module: self
        """

        self.prediction = mode
        return self
