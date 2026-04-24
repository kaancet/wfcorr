import pickle
from pathlib import Path
from typing import Optional
from temporaldata import Data
from dataclasses import dataclass
from datetime import datetime as dt

from event import MovementEvents, LickEvents, StimEvents, FeedbackEvents
from region import RegionActivity


ANALYSIS_PATH = Path("/mnt/raid/data/WF_analysis/")


@dataclass
class Session(Data):
    """"""

    session_date: dt
    mouse_id: str

    movements: MovementEvents
    licks: LickEvents
    stims: StimEvents
    feedback: FeedbackEvents

    regions: Optional[list[RegionActivity]]

    @classmethod
    def from_name(cls, session_name: str) -> "Session":
        """_summary_

        Args:
            session_path (str): _description_

        Returns:
            Session: _description_
        """

        # for now semi hard-coded for expediency
        curated_path = ANALYSIS_PATH / session_name / "curation"

        movement_path = curated_path / "movement_data_synched_resampled_merged.pkl"
        behavior_path = curated_path / "behaviour_imaging_synched.pkl"

        with open(str(movement_path), "rb") as mf:
            mobj = pickle.load(mf)

        with open(str(behavior_path), "rb") as bf:
            bobj = pickle.load(bf)

        # get the date
        date_str = session_name.split("_")[0]
        date = dt.strptime(date_str, "%y%m%d")

        mouse_id = session_name.split("_")[1]

        return cls(
            session_date=date,
            mouse_id=mouse_id,
            movements=MovementEvents.from_dataframe(mobj),
            licks=LickEvents.from_dataframe(bobj),
            stims=StimEvents.from_dataframe(bobj),
            feedback=FeedbackEvents.from_dataframe(bobj),
        )

    def add_region(self, region: RegionActivity) -> None:
        self.regions.append(region)
