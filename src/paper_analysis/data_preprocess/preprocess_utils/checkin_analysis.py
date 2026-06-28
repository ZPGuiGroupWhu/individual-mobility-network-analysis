import warnings
from datetime import timedelta
import numpy as np
import pandas as pd

def create_activity_flag_checkin(staypoints, method='time_threshold', time_threshold=15.0,
                                 activity_column_name='is_activity'):
    """
        Add a flag whether or not a staypoint is considered an activity based on a time threshold.

        Parameters
        ----------
        staypoints: Staypoints

        method: {'time_threshold'}, default = 'time_threshold'
            - 'time_threshold' : All staypoints with a duration greater than the time_threshold are considered an activity.

        time_threshold : float, default = 15 (minutes)
            The time threshold for which a staypoint is considered an activity in minutes. Used by method 'time_threshold'

        activity_column_name : str , default = 'is_activity'
            The name of the newly created column that holds the activity flag.

        Returns
        -------
        staypoints : Staypoints
            Original staypoints with the additional activity column
        """
    if method == "time_threshold":
        staypoints[activity_column_name] = (staypoints["tracked_at"].shift(-1) - staypoints["tracked_at"]) > timedelta(
            minutes=time_threshold)
    else:
        raise ValueError(f"Method {method} not known for creating activity flag.")

    return staypoints


def location_identifier_checkin(staypoints, method='FREQ', pre_filter=True, **pre_filter_kwargs):
    """Assign "home" and "work" activity label for each user with different methods.

        Parameters
        ----------
        staypoints : Staypoints
            Staypoints with column "location_id".

        method : {'FREQ', 'OSNA'}, default "FREQ"
            'FREQ': Generate an activity label per user by assigning the most visited location the label "home"
            and the second most visited location the label "work". The remaining locations get no label.
            'OSNA': Use weekdays data divided in three time frames ["rest", "work", "leisure"]. Finds most popular
            home location for timeframes "rest" and "leisure" and most popular "work" location for "work" timeframe.

        pre_filter : bool, default True
            Prefiltering the staypoints to exclude locations with not enough data.
            The filter function can also be accessed via `pre_filter_locations`.

        pre_filter_kwargs : dict
            Kwargs to hand to `pre_filter_locations` if used. See function for more informations.

        Returns
        -------
        sp: Staypoints
            With additional column `purpose` assigning one of three activity labels {'home', 'work', None}.

        Note
        ----
        The methods are adapted from [1]. The algorithms count the distinct hours at a
        location as the home location is derived from geo-tagged tweets.

        References
        ----------
        [1] Chen, Qingqing, and Ate Poorthuis. 2021.
        "Identifying Home Locations in Human Mobility Data: An Open-Source R Package for Comparison and Reproducibility".
        International Journal of Geographical Information Science 0 (0): 1-24.
        https://doi.org/10.1080/13658816.2021.1887489.
        """
    sp = staypoints.copy()
    if "location_id" not in sp.columns:
        raise KeyError(
            (
                "To derive activity labels the Staypoints must have a column "
                f"named 'location_id' but it has [{', '.join(sp.columns)}]"
            )
        )
    if pre_filter:
        f = pre_filter_locations(sp, **pre_filter_kwargs)
    else:
        f = pd.Series(np.full(len(sp.index), True), index=sp.index)

    if method == "FREQ":
        method_val = freq_method(sp[f], "home", "work")
    elif method == "OSNA":
        method_val = osna_method(sp[f])
    else:
        raise ValueError(f"Method {method} does not exist.")
    sp.loc[f, "purpose"] = method_val["purpose"]

    return sp


def pre_filter_locations(
        staypoints,
        agg_level="user",
        thresh_sp=10,
        thresh_loc=10,
        thresh_sp_at_loc=10,
):
    """Filter locations and user out that have not enough data to do a proper analysis.

    To disable a specific filter parameter set it to zero.

    Parameters
    ----------
    staypoints : Staypoints
        Staypoints with the column "location_id".

    agg_level: {"user", "dataset"}, default "user"
        The level of aggregation when filtering locations. 'user' : locations are filtered per-user;
        'dataset' : locations are filtered over the whole dataset.

    thresh_sp : int, default 10
        Minimum staypoints a user must have to be included.

    thresh_loc : int, default 10
        Minimum locations a user must have to be included.

    thresh_sp_at_loc : int, default 10
        Minimum number of staypoints at a location must have to be included.

    Returns
    -------
    total_filter: pd.Series
        Boolean series containing the filter as a mask.
    """

    sp = staypoints.copy()

    # filtering users
    user = sp.groupby("user_id").nunique()
    user_sp = user["tracked_at"] >= thresh_sp  # every staypoint should have a started_at -> count
    user_loc = user["location_id"] >= thresh_loc

    user_filter_agg = user_sp & user_loc
    user_filter_agg.rename("user_filter", inplace=True)  # rename for merging
    user_filter = pd.merge(sp["user_id"], user_filter_agg, left_on="user_id", right_index=True)["user_filter"]

    # filtering locations
    # sp["duration"] = sp["finished_at"] - sp["started_at"]
    if agg_level == "user":
        groupby_loc = ["user_id", "location_id"]
    elif agg_level == "dataset":
        groupby_loc = ["location_id"]
    else:
        raise ValueError(f"Unknown agg_level '{agg_level}' use instead {{'user', 'dataset'}}.")

    loc = sp.groupby(groupby_loc).agg({"tracked_at": "count"})
    # loc.columns = loc.columns.droplevel(0)  # remove possible multi-index
    # loc.rename(columns={"min": "started_at", "max": "finished_at", "sum": "duration"}, inplace=True)
    # period for maximal time span first visit - last visit.
    # duration for effective time spent at location summed up.

    loc_sp = loc["tracked_at"] >= thresh_sp_at_loc

    loc_filter_agg = loc_sp
    loc_filter_agg.rename("loc_filter", inplace=True)  # rename for merging
    loc_filter = pd.merge(sp[groupby_loc], loc_filter_agg, how="left", left_on=groupby_loc, right_index=True)[
        "loc_filter"]

    total_filter = user_filter & loc_filter

    return total_filter


def freq_method(staypoints, *labels):
    """Generate an activity label per user.

    Assigning the most visited location the label "home" and the second most visited location the label "work".
    The remaining locations get no label.

    Labels can also be given as arguments.

    Parameters
    ----------
    staypoints : Staypoints
        Staypoints with the column "location_id".

    labels : collection of str, default ("home", "work")
        Labels in decreasing time of activity.

    Returns
    -------
    sp: Staypoints
        The input staypoints with additional column "purpose".

    """
    sp = staypoints.copy()
    if not labels:
        labels = ("home", "work")
    for name, group in sp.groupby("user_id"):
        if "checkin_count" not in group.columns:
            group["checkin_count"] = 1
        # pandas keeps inner order of groups
        sp.loc[sp["user_id"] == name, "purpose"] = _freq_transform(group, *labels)
    if "purpose" not in sp.columns:  # if empty sp
        sp["purpose"] = None
    return sp


def _freq_transform(group, *labels):
    """Transform function that assigns the longest (duration) visited locations the labels in order.

    Parameters
    ----------
    group : pd.DataFrame
        Should have columns "location_id" and "duration".

    Returns
    -------
    pd.Series
        dtype : object
    """
    group_agg = group.groupby("location_id").agg({"checkin_count": "count"})
    group_agg["purpose"] = _freq_assign(group_agg["checkin_count"], *labels)
    group_merge = pd.merge(
        group["location_id"], group_agg["purpose"], how="left", left_on="location_id", right_index=True
    )
    return group_merge["purpose"]


def _freq_assign(checkin_count, *labels):
    """Assign k labels to k most checkins the rest is `None`.

    Parameters
    ----------
    checkin_count : pd.Series

    Returns
    -------
    np.array
        dtype : object
    """
    kth = (-checkin_count).argsort()[: len(labels)]  # if inefficient use partial sort.
    label_array = np.full(len(checkin_count), fill_value=None)
    labels = labels[: len(kth)]  # if provided with more labels than entries.
    label_array[kth] = labels
    return label_array


def osna_method(staypoints):
    """Find "home" location for timeframes "rest" and "leisure" and "work" location for "work" timeframe.

    Use weekdays data divided in three time frames ["rest", "work", "leisure"] to generate location labels.
    "rest" + "leisure" locations are weighted together. The location with the most checkins is assigned "home" label.
    The most "work" location is assigned "work" label.

    Parameters
    ----------
    staypoints : Staypoints
        Staypoints with the column "location_id".

    Returns
    -------
    Staypoints
        The input staypoints with additional column "purpose".

    Note
    ----
    The method is adapted from [1].
    When "home" and "work" label overlap, the method selects the "work" location by the 2nd highest score.

    References
    ----------
    [1] Efstathiades, Hariton, Demetris Antoniades, George Pallis, and Marios Dikaiakos. 2015.
    "Identification of Key Locations Based on Online Social Network Activity".
    In https://doi.org/10.1145/2808797.2808877.

    """
    sp_in = staypoints  # no copy --> used to join back later.
    sp = sp_in.copy()
    sp["checkin_count"] = 1.0

    sp["label"] = sp["tracked_at"].apply(_osna_label_timeframes)
    sp.loc[sp["label"] == "rest", "checkin_count"] *= 0.739  # weight given in paper
    sp.loc[sp["label"] == "leisure", "checkin_count"] *= 0.358  # weight given in paper

    groups_map = {
        "rest": "home",
        "leisure": "home",
        "work": "work",
    }  # weekends aren't included in analysis!
    # groupby user, location and label.
    groups = ["user_id", "location_id", sp["label"].map(groups_map)]

    sp_agg = sp.groupby(groups)["checkin_count"].sum()
    if sp_agg.empty:
        print("Got empty table in the osna method, check if the dates lie in weekends.")
        sp_in["purpose"] = pd.NA
        return sp_in

    # create a pivot table -> labels "home" and "work" as columns. ("user_id", "location_id" still in index.)
    sp_pivot = sp_agg.unstack()
    # get index of maximum for columns "work" and "home"
    # looks over locations to find maximum for columns
    # use fillna such that idxmax raises no error on columns with only NaT
    sp_idxmax = sp_pivot.fillna(0).groupby(["user_id"]).idxmax()

    # preset dtype to avoid upcast (float64 -> object) in pandas (and the corresponding error)
    sp_pivot["purpose"] = None
    # assign empty index to idx_work/idx_home to have a default behavior for the intersection later
    idx_work = idx_home = pd.Index([])
    if "work" in sp_pivot.columns:
        # first get all index of max entries (of work) that are not NaT
        idx_work = sp_pivot.loc[sp_idxmax["work"], "work"].dropna().index
        # set them to the corresponding purpose (work)
        sp_pivot.loc[idx_work, "purpose"] = "work"

    if "home" in sp_pivot.columns:
        # get all index of max entries (of home) that are not NaT
        idx_home = sp_pivot.loc[sp_idxmax["home"], "home"].dropna().index
        # set them to the corresponding purpose (home overrides work!)
        sp_pivot.loc[idx_home, "purpose"] = "home"

    # if override happened recalculate work
    overlap = idx_home.intersection(idx_work)
    if not overlap.empty:
        # remove overlap -> must take another location as everything is NaT on maximum
        sp_pivot.loc[overlap, "work"] = None
        sp_idxmax = sp_pivot["work"].fillna(0).groupby(["user_id"]).idxmax()
        idx_work = sp_pivot.loc[sp_idxmax, "work"].dropna().index
        sp_pivot.loc[idx_work, "purpose"] = "work"

    # now join it back together
    sel = sp_in.columns != "purpose"  # no overlap with older "purpose"
    return pd.merge(
        sp_in.loc[:, sel],
        sp_pivot["purpose"],
        how="left",
        left_on=["user_id", "location_id"],
        right_index=True,
    )


def _osna_label_timeframes(dt, weekend=[5, 6], start_rest=2, start_work=8, start_leisure=19):
    """Help function to assign "weekend", "rest", "work", "leisure"."""
    if dt.weekday() in weekend:
        return "weekend"
    if start_rest <= dt.hour < start_work:
        return "rest"
    if start_work <= dt.hour < start_leisure:
        return "work"
    return "leisure"
