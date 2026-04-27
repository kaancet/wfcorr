import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
from temporaldata import Interval, RegularTimeSeries, IrregularTimeSeries


@dataclass(init=False)
class MovementEvents(RegularTimeSeries):
    """"""

    def __init__(self, *, sampling_rate, domain=None, domain_start=0, **kwargs):
        super().__init__(sampling_rate=sampling_rate, domain=domain, domain_start=domain_start, **kwargs)

    @classmethod
    def from_pandas(cls, in_df: pd.DataFrame) -> "MovementEvents":
        """_summary_

        Args:
            in_df (_type_): _description_

        Returns:
            Event: _description_
        """

        ts = in_df.index.to_numpy()
        event_names = in_df.columns.to_list()

        sampling_rate = round(1 / np.mean(np.diff(ts)), 1)

        return cls(
            sampling_rate=sampling_rate,
            domain=Interval(0, ts[-1]),
            **{{c: in_df[c].values for c in event_names}},
        )


@dataclass
class LickEvents(IrregularTimeSeries):
    """"""

    @classmethod
    def from_pandas(cls, in_df: pd.DataFrame) -> "LickEvents":
        """_summary_

        Args:
            in_df (pd.DataFrame): _description_

        Returns:
            LickEvents: _description_
        """

        ts = in_df["licks"].explode().values.astype(float)

        return cls(
            timestamps=ts,
            domain=Interval(0, ts[-1]),
        )


@dataclass
class StimEvents(Interval):
    """"""

    @classmethod
    def from_pandas(cls, in_df: pd.DataFrame) -> "StimEvents":
        """_summary_

        Args:
            in_df (pd.DataFrame): _description_

        Returns:
            StimEvents: _description_
        """

        temp_dict = defaultdict(list)
        for i in in_df["trial_timing"].to_list():
            for k, v in i.items():
                temp_dict[k].append(v)

        starts = np.array(temp_dict["photoindicator"])
        ends = np.array(temp_dict["photoindicator_off"])

        return cls(
            start=starts,
            end=ends,
            stimulus=in_df["stimulus"].values,
            outcome=in_df["outcome"].values,
            timekeys=["start", "end"],
        )


@dataclass
class FeedbackEvents(IrregularTimeSeries):
    """ """

    @classmethod
    def from_pandas(cls, in_df: pd.DataFrame) -> "FeedbackEvents":
        """_summary_

        Args:
            in_df (pd.DataFrame): _description_

        Returns:
            FeedbackEvents: _description_
        """

        fb_list = []
        for i in in_df["trial_timing"].to_list():
            for k, v in i.items():
                if k in ["reward", "punishment"]:
                    fb_list.append((k, v))

        ts = np.array([t for _, t in fb_list])

        return cls(
            timestamps=ts,
            domain=Interval(0, ts[-1]),
            feedback=np.array([f for f, _ in fb_list]),
        )
